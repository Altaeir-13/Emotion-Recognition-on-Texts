import pandas as pd
import torch
import numpy as np
import os
import gc
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold

# ==========================================
# 1. CONFIGURAÇÕES NUCLEARES (V17 - K-FOLD)
# ==========================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
N_FOLDS = 5                     # O padrão ouro para Cross-Validation
PER_DEVICE_BATCH_SIZE = 16      
GRADIENT_ACCUMULATION = 2       
EPOCHS = 4            
LEARNING_RATE = 3e-5  

BASE_DIR = os.getcwd()
OUTPUT_DIR_BASE = os.path.join(BASE_DIR, "resultado_v17_fold")
SUBMISSION_FILE = "submissao_final_v17_kfold.csv"

# ==========================================
# 2. FUNÇÕES DE CARGA E UTILITÁRIOS
# ==========================================
label_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def find_file(filename):
    if os.path.exists(filename): return filename
    possible_paths = [os.path.join("Nivel de Competicao", filename), os.path.join(BASE_DIR, filename)]
    for p in possible_paths:
        if os.path.exists(p): return p
    return None

def load_base_data(path):
    real_path = find_file(path)
    if not real_path:
        print(f"❌ ERRO: {path} não encontrado.")
        return pd.DataFrame()
    print(f"📂 Lendo Base: {real_path}...")
    df = pd.read_csv(real_path)
    df = df.dropna(subset=['text'])
    for col in label_cols:
        if col not in df.columns: df[col] = 0
    df['labels'] = df[label_cols].values.tolist()
    return df.reset_index(drop=True)

def apply_reality_injection(df):
    """Aplica o Oversampling V16 apenas nos dados de treino do fold atual"""
    print("💉 [Fold] Injetando realidade (Oversampling)...")
    if 'id' in df.columns:
        mask_real = ~df['id'].astype(str).str.startswith('sintetico')
    else:
        mask_real = pd.Series([True] * len(df))

    # Filtra exemplos REAIS de classes difíceis
    df_real_disgust = df[mask_real & (df['disgust'] == 1)]
    df_real_fear    = df[mask_real & (df['fear'] == 1)]
    
    extras = []
    # Multiplica x15 (Agressivo para furar a bolha)
    if not df_real_disgust.empty: extras.append(pd.concat([df_real_disgust] * 15))
    if not df_real_fear.empty:    extras.append(pd.concat([df_real_fear] * 15))
    
    if extras:
        df_extras = pd.concat(extras)
        df = pd.concat([df, df_extras])
        print(f"   -> +{len(df_extras)} linhas reais injetadas.")
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.targets = dataframe.labels
        self.max_len = max_len
    def __len__(self): return len(self.text)
    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = torch.nn.Sigmoid()(torch.Tensor(preds)).numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_true = p.label_ids
    return {'f1_macro': f1_score(y_true, y_pred, average='macro')}

# ==========================================
# 3. PREPARAÇÃO
# ==========================================
train_file = "Balanced With Sintetic Data.csv"
if not find_file(train_file): train_file = "Balanced With Sintetic Data V5.csv"

# Carrega dataset COMPLETO (será dividido pelo KFold)
full_train_df = load_base_data(train_file)
val_df_external = load_base_data("dev.csv") # Dev externo para calibração final
test_df = load_base_data("test.csv")

if full_train_df.empty: exit()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
val_external_set = EmotionDataset(val_df_external, tokenizer)
test_set = EmotionDataset(test_df, tokenizer)

# Acumuladores para Ensemble
# Vamos somar as probabilidades de cada fold aqui
test_probs_sum = np.zeros((len(test_df), 6))
val_external_probs_sum = np.zeros((len(val_df_external), 6))

# ==========================================
# 4. LOOP NUCLEAR (K-FOLD)
# ==========================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"\n☢️ INICIANDO PROTOCOLO K-FOLD ({N_FOLDS} Folds) ☢️")

for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_df)):
    print(f"\n{'='*40}")
    print(f"🔄 FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*40}")
    
    # 4.1. Split e Injeção
    fold_train_df = full_train_df.iloc[train_idx].copy()
    fold_val_df   = full_train_df.iloc[val_idx].copy() # Validação interna do fold
    
    # Aplica Oversampling SÓ no treino (não vaza para validação)
    fold_train_df = apply_reality_injection(fold_train_df)
    
    print(f"   -> Treino: {len(fold_train_df)} | Validação Interna: {len(fold_val_df)}")
    
    train_set = EmotionDataset(fold_train_df, tokenizer)
    eval_set  = EmotionDataset(fold_val_df, tokenizer)
    
    # 4.2. Setup Modelo e Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6, problem_type="multi_label_classification"
    )
    model.to(device)
    
    fold_output_dir = f"{OUTPUT_DIR_BASE}_{fold+1}"
    
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=0.05,            
        warmup_ratio=0.1,             
        lr_scheduler_type="cosine",   
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        dataloader_num_workers=0
    )
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_set,
        eval_dataset=eval_set, tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 4.3. Treinar
    trainer.train()
    
    # 4.4. Inferência (Ensemble Building)
    print(f"🔮 [Fold {fold+1}] Gerando previsões para Ensemble...")
    
    # Prevê no Teste
    test_preds = trainer.predict(test_set)
    test_probs = torch.nn.Sigmoid()(torch.tensor(test_preds.predictions)).numpy()
    test_probs_sum += test_probs # Acumula
    
    # Prevê no Dev Externo (para calibração final)
    val_preds = trainer.predict(val_external_set)
    val_probs = torch.nn.Sigmoid()(torch.tensor(val_preds.predictions)).numpy()
    val_external_probs_sum += val_probs # Acumula
    
    # 4.5. Limpeza (Libera VRAM para o próximo fold)
    del model, trainer, test_preds, val_preds
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ Fold {fold+1} concluído. Memória limpa.")

# ==========================================
# 5. ENSEMBLE E CALIBRAGEM FINAL
# ==========================================
print("\n🎉 K-FOLD CONCLUÍDO! Calculando Ensemble...")

# Média das probabilidades
avg_test_probs = test_probs_sum / N_FOLDS
avg_val_probs  = val_external_probs_sum / N_FOLDS

# Labels reais do Dev Externo
val_labels = np.array(val_df_external['labels'].tolist())

print("--- OTIMIZAÇÃO DE THRESHOLDS (VIA ENSEMBLE) ---")
best_thresholds = {}
for i, label in enumerate(label_cols):
    best_t = 0.5
    best_f1 = 0
    
    # Grid Search na Média do Dev
    for t in np.arange(0.1, 0.95, 0.05):
        score = f1_score(val_labels[:, i], (avg_val_probs[:, i] >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    # Trava de Segurança V16 (Mantida)
    if best_f1 < 0.05:
        print(f"⚠️ {label}: F1 zerado no Ensemble. Forçando 0.20.")
        best_t = 0.20
        
    best_thresholds[label] = best_t
    print(f"✅ {label.ljust(10)} | Melhor Corte: {best_t:.2f} | F1 Val: {best_f1:.4f}")

# ==========================================
# 6. GERAÇÃO FINAL
# ==========================================
print("\n--- GERANDO CSV FINAL ---")
final_preds = np.zeros(avg_test_probs.shape)

for i, label in enumerate(label_cols):
    final_preds[:, i] = (avg_test_probs[:, i] >= best_thresholds[label]).astype(int)

# Contagem de Neutros
neutros = 0
for k in range(len(final_preds)):
    if np.sum(final_preds[k]) == 0: neutros += 1
print(f"ℹ️ Neutros finais: {neutros}")

sub_df = pd.DataFrame(final_preds.astype(int), columns=label_cols)
if 'id' in test_df.columns: sub_df['id'] = test_df['id']
cols = ['id'] + label_cols if 'id' in sub_df.columns else label_cols
sub_df = sub_df[cols]

sub_df.to_csv(SUBMISSION_FILE, index=False)
print("="*50)
print(f"☢️ ARQUIVO NUCLEAR GERADO: {SUBMISSION_FILE}")
print("="*50)
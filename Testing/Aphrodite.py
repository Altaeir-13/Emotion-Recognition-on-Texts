import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import gc
import shutil
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# ==========================================
# 1. CONFIGURAÇÕES GOD MODE (V20)
# ==========================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
N_FOLDS = 5                     
PER_DEVICE_BATCH_SIZE = 16      
GRADIENT_ACCUMULATION = 2       
EPOCHS = 4            
LEARNING_RATE = 3e-5  

BASE_DIR = os.getcwd()
OUTPUT_DIR_BASE = os.path.join(BASE_DIR, "resultado_v20_god")
SUBMISSION_FILE = "submissao_final_v20_god.csv"

# ==========================================
# 2. CARGA DE DADOS
# ==========================================
label_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def find_file(filename):
    if os.path.exists(filename): return filename
    possible_paths = [
        os.path.join("Testing", filename),
        os.path.join(BASE_DIR, filename)
    ]
    for p in possible_paths:
        if os.path.exists(p): return p
    return None

def load_data(path):
    real_path = find_file(path)
    if not real_path:
        print(f"❌ ERRO CRÍTICO: {path} não encontrado.")
        return pd.DataFrame()
    
    print(f"📂 Lendo: {real_path}...")
    df = pd.read_csv(real_path)
    df = df.dropna(subset=['text'])
    for col in label_cols:
        if col not in df.columns: df[col] = 0
    df['labels'] = df[label_cols].values.tolist()
    return df.reset_index(drop=True)

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
# 3. CUSTOM TRAINER (WEIGHTED LOSS)
# ==========================================
class WeightedTrainer(Trainer):
    def __init__(self, pos_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Garante que os pesos estejam no mesmo dispositivo do modelo
        self.pos_weights = pos_weights.to(self.args.device) if pos_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # AQUI ESTÁ O SEGREDO DO V20:
        # Usamos BCEWithLogitsLoss com pos_weight calculado dinamicamente
        # Isso força o modelo a valorizar os '1's (positivos) muito mais que os '0's.
        if self.pos_weights is not None:
            # Move pesos para o device correto se necessário (segurança extra)
            if self.pos_weights.device != logits.device:
                self.pos_weights = self.pos_weights.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 4. PREPARAÇÃO
# ==========================================
print("\n--- ETAPA 1: CARGA DE DADOS (V5) ---")
train_file = "Balanced With Sintetic Data V5.csv"
if not find_file(train_file):
    print("⚠️ V5 não achado. Usando V1...")
    train_file = "Balanced With Sintetic Data.csv"

full_train_df = load_data(train_file)
val_df_external = load_data("dev.csv") 
test_df = load_data("test.csv")

if full_train_df.empty: exit()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
val_external_set = EmotionDataset(val_df_external, tokenizer)
test_set = EmotionDataset(test_df, tokenizer)

test_probs_sum = np.zeros((len(test_df), 6))
val_external_probs_sum = np.zeros((len(val_df_external), 6))

# ==========================================
# 5. LOOP NUCLEAR (K-FOLD + WEIGHTED LOSS)
# ==========================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"\n☢️ INICIANDO GOD MODE ({N_FOLDS} Folds) + WEIGHTED LOSS ☢️")

for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_df)):
    print(f"\n{'='*40}")
    print(f"🔄 FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*40}")
    
    fold_train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    fold_val_df   = full_train_df.iloc[val_idx].reset_index(drop=True)
    
    # --- CÁLCULO DINÂMICO DOS PESOS (WEIGHTED LOSS) ---
    print("⚖️ Calculando pesos para equilibrar classes neste Fold...")
    # Fórmula: (Negativos / Positivos)
    # Ex: Se tenho 400 'Não' e 100 'Sim', o peso do 'Sim' deve ser 4.
    pos_counts = fold_train_df[label_cols].sum().values
    total_counts = len(fold_train_df)
    neg_counts = total_counts - pos_counts
    
    # Evita divisão por zero com np.maximum(1)
    weights = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float)
    print(f"   -> Pesos calculados: {weights.numpy().round(2)}")
    
    train_set = EmotionDataset(fold_train_df, tokenizer)
    eval_set  = EmotionDataset(fold_val_df, tokenizer)
    
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
    
    # Usando o WeightedTrainer em vez do Trainer padrão
    trainer = WeightedTrainer(
        pos_weights=weights,
        model=model, args=training_args, train_dataset=train_set,
        eval_dataset=eval_set, tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    print(f"🔮 [Fold {fold+1}] Somando votos...")
    test_preds = trainer.predict(test_set)
    test_probs = torch.nn.Sigmoid()(torch.tensor(test_preds.predictions)).numpy()
    test_probs_sum += test_probs 
    
    val_preds = trainer.predict(val_external_set)
    val_probs = torch.nn.Sigmoid()(torch.tensor(val_preds.predictions)).numpy()
    val_external_probs_sum += val_probs
    
    del model, trainer, test_preds, val_preds
    torch.cuda.empty_cache()
    gc.collect()
    try: shutil.rmtree(fold_output_dir)
    except: pass

# ==========================================
# 6. ENSEMBLE E CALIBRAGEM FINAL
# ==========================================
print("\n🎉 PROCESSAMENTO CONCLUÍDO! Otimizando...")

avg_test_probs = test_probs_sum / N_FOLDS
avg_val_probs  = val_external_probs_sum / N_FOLDS
val_labels = np.array(val_df_external['labels'].tolist())

best_thresholds = {}
for i, label in enumerate(label_cols):
    best_t = 0.5
    best_f1 = 0
    
    # Com Weighted Loss, as probabilidades tendem a ser mais altas.
    # O Grid Search vai achar o ponto de corte ideal naturalmente.
    for t in np.arange(0.1, 0.95, 0.05):
        score = f1_score(val_labels[:, i], (avg_val_probs[:, i] >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    # Trava de Segurança (agora menos necessária, mas mantida por prudência)
    if best_f1 < 0.05:
        print(f"⚠️ {label}: F1 zerado. Forçando 0.20.")
        best_t = 0.20
        
    best_thresholds[label] = best_t
    print(f"✅ {label.ljust(10)} | Melhor Corte: {best_t:.2f} | F1 Val: {best_f1:.4f}")

# ==========================================
# 7. GERAÇÃO FINAL
# ==========================================
final_preds = np.zeros(avg_test_probs.shape)
for i, label in enumerate(label_cols):
    final_preds[:, i] = (avg_test_probs[:, i] >= best_thresholds[label]).astype(int)

sub_df = pd.DataFrame(final_preds.astype(int), columns=label_cols)
if 'id' in test_df.columns: sub_df['id'] = test_df['id']
cols = ['id'] + label_cols if 'id' in sub_df.columns else label_cols
sub_df = sub_df[cols]

sub_df.to_csv(SUBMISSION_FILE, index=False)
print("="*50)
print(f"🏆 ARQUIVO GOD MODE GERADO: {SUBMISSION_FILE}")
print("="*50)
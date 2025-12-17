import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score

# --- 1. Configurações de Elite ---
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Batch Size: Se der erro de memória (OOM), diminua para 8 ou 4.
# Se tiver memória sobrando, aumente para 32.
BATCH_SIZE = 16  

# Epochs: Com 500k linhas balanceadas, 3 épocas costumam ser o suficiente.
# O EarlyStopping vai parar antes se o modelo convergir rápido.
EPOCHS = 4       
LEARNING_RATE = 2e-5

# --- CAMINHOS (Ajustado conforme seu log) ---
# Caminho relativo baseado na sua estrutura de pastas
TRAIN_PATH = "Nivel de Competicao/Golden Standard.csv" 
DEV_PATH   = "Datasets/Original/dev.csv" 
TEST_PATH  = "Datasets/Original/test.csv"

OUTPUT_DIR = "./resultados_final_gold"
MODEL_SAVE_DIR = "./modelo_final_gold"

# --- 2. Setup Hardware ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM Total: {vram:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️ GPU não detectada. Treino será lento.")

# --- 3. Carregamento e Tratamento ---
label_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def load_data(path, is_train=False):
    if not os.path.exists(path):
        print(f"❌ ERRO CRÍTICO: Arquivo não encontrado: {path}")
        # Tenta procurar na raiz se falhar na pasta
        filename = os.path.basename(path)
        if os.path.exists(filename):
            print(f"   ↳ Arquivo encontrado na raiz: {filename}. Usando ele.")
            return pd.read_csv(filename)
        return pd.DataFrame()
    
    print(f"Carregando: {path}...")
    df = pd.read_csv(path)
    
    # Tratamento de segurança (Sanity Check em tempo de execução)
    df = df.dropna(subset=['text'])
    
    # Garante colunas numéricas
    for col in label_cols:
        if col not in df.columns: df[col] = 0
            
    # Cria a coluna de lista de labels
    df['labels'] = df[label_cols].values.tolist()
    
    # Para o treino, embaralhar é essencial
    if is_train:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return df

train_df = load_data(TRAIN_PATH, is_train=True)
val_df = load_data(DEV_PATH)
test_df = load_data(TEST_PATH)

if train_df.empty:
    print("🚨 ABORTANDO: Dataset de treino vazio ou não encontrado.")
    exit()

print(f"📊 Dataset de Treino Final: {len(train_df)} linhas")

# --- 4. Dataset Class ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.targets = dataframe.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        # Limpeza básica de espaços extras
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

train_set = EmotionDataset(train_df, tokenizer)
val_set = EmotionDataset(val_df, tokenizer)
test_set = EmotionDataset(test_df, tokenizer)

# --- 5. Modelo ---
# Multi-label classification (Sigmoid + BCE Loss)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)
model.to(device)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds)).numpy()
    
    # Threshold de 0.4 é o padrão ouro para multi-label balanceado
    y_pred = (probs >= 0.4).astype(int)
    y_true = p.label_ids
    
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred)
    }

# --- 6. Treinamento ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=torch.cuda.is_available(), # Mixed Precision para acelerar
    save_total_limit=1,
    dataloader_num_workers=0 # Mantém 0 no Windows para evitar bugs de processo
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n🚀 INICIANDO TREINO (DATASET GOLD STANDARD) 🚀")
print("Isso vai demorar algumas horas. Vá tomar um café.")
trainer.train()

# --- 7. Finalização ---
print("Salvando modelo...")
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)

print("Gerando previsões finais no Test Set...")
preds = trainer.predict(test_set)
probs = torch.nn.Sigmoid()(torch.tensor(preds.predictions)).numpy()

# Pós-processamento: Threshold 0.35 (Levemente agressivo) + Forçar Escolha
THRESHOLD = 0.35
y_pred_final = (probs >= THRESHOLD).astype(int)

forced_count = 0
for i in range(len(y_pred_final)):
    if np.sum(y_pred_final[i]) == 0:
        # Se nenhuma classe atingiu o threshold, pega a maior probabilidade
        y_pred_final[i][np.argmax(probs[i])] = 1
        forced_count += 1

print(f"Fallback 'Forçar Escolha' aplicado em {forced_count} linhas.")

# Gerar CSV de submissão
sub_df = pd.DataFrame(y_pred_final, columns=label_cols)
if 'id' in test_df.columns:
    sub_df['id'] = test_df['id']
    sub_df = sub_df[['id'] + label_cols]

output_file = "submissao_final_gold.csv"
sub_df.to_csv(output_file, index=False)
print(f"✅ FINALIZADO! Arquivo de submissão gerado: {output_file}")
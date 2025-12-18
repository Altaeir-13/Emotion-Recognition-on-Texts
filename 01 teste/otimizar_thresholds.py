import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import f1_score, classification_report

# --- Configurações ---
MODEL_PATH = "C:\\Users\\handi\\Documents\\Estudo\\AI\\01 teste\\modelo_final_gold"  # Onde o treino salvou o modelo
DATA_PATH = "C:\\Users\\handi\\Documents\\Estudo\\AI\\01 teste\\"                    # Onde estão dev.csv e test.csv
BATCH_SIZE = 32

# --- Funções Auxiliares ---
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text'])
    label_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for col in label_cols:
        if col not in df.columns: df[col] = 0
    df['labels'] = df[label_cols].values.tolist()
    return df, label_cols

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

# --- 1. Carregar Modelo e Dados ---
print("⏳ Carregando modelo e dados...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Carrega Validação (para achar o threshold ideal) e Teste (para gerar submissão)
val_df, label_cols = load_data(os.path.join(DATA_PATH, "dev.csv"))
test_df, _ = load_data(os.path.join(DATA_PATH, "test.csv"))

val_set = EmotionDataset(val_df, tokenizer)
test_set = EmotionDataset(test_df, tokenizer)

trainer = Trainer(model=model)

# --- 2. Gerar Previsões (Logits) ---
print("🔮 Prevendo no conjunto de Validação...")
val_preds = trainer.predict(val_set)
val_probs = torch.nn.Sigmoid()(torch.tensor(val_preds.predictions)).numpy()
val_labels = val_preds.label_ids

print("🔮 Prevendo no conjunto de Teste...")
test_preds = trainer.predict(test_set)
test_probs = torch.nn.Sigmoid()(torch.tensor(test_preds.predictions)).numpy()

# --- 3. Otimização de Threshold ---
print("\n--- 📐 OTIMIZANDO THRESHOLDS ---")
best_thresholds = {}
best_f1_scores = {}

# Testa thresholds de 0.10 a 0.90 para CADA classe individualmente
for i, label in enumerate(label_cols):
    best_t = 0.5
    best_f1 = 0
    
    # Grid search simples
    for t in np.arange(0.1, 0.95, 0.05):
        # Cria previsões binárias usando o threshold t
        preds_bin = (val_probs[:, i] >= t).astype(int)
        # Calcula F1 apenas para essa classe (Binário)
        score = f1_score(val_labels[:, i], preds_bin)
        
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    best_thresholds[label] = best_t
    best_f1_scores[label] = best_f1
    print(f"✅ {label.ljust(10)} | Melhor Threshold: {best_t:.2f} | F1: {best_f1:.4f}")

# --- 4. Relatório Final de Performance (Validação) ---
print("\n📊 Performance Estimada com Thresholds Otimizados:")
final_val_preds = np.zeros(val_probs.shape)
for i, label in enumerate(label_cols):
    t = best_thresholds[label]
    final_val_preds[:, i] = (val_probs[:, i] >= t).astype(int)

print(classification_report(val_labels, final_val_preds, target_names=label_cols, zero_division=0))

# --- 5. Aplicar no Teste e Salvar ---
print("💾 Gerando arquivo de submissão final...")
final_test_preds = np.zeros(test_probs.shape)

for i, label in enumerate(label_cols):
    t = best_thresholds[label]
    final_test_preds[:, i] = (test_probs[:, i] >= t).astype(int)

# Fallback para linhas zeradas (Se mesmo com threshold baixo não marcou nada)
forced_count = 0
for k in range(len(final_test_preds)):
    if np.sum(final_test_preds[k]) == 0:
        # Pega a classe com maior probabilidade bruta
        idx_max = np.argmax(test_probs[k])
        final_test_preds[k][idx_max] = 1
        forced_count += 1

print(f"⚠️ Fallback aplicado em {forced_count} linhas (forçado a maior probabilidade).")

# Salva CSV
sub_df = pd.DataFrame(final_test_preds.astype(int), columns=label_cols)
if 'id' in test_df.columns:
    sub_df['id'] = test_df['id']
    sub_df = sub_df[['id'] + label_cols]

sub_df.to_csv("submissao_otimizada.csv", index=False)
print("🚀 Sucesso! Arquivo 'submissao_otimizada.csv' pronto.")
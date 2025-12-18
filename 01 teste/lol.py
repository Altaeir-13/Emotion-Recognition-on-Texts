import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# --- Configurações ---
# Caminho absoluto para evitar erros
MODEL_PATH = "C:\\Users\\handi\\Documents\\Estudo\\AI\\01 teste\\modelo_final_gold"
TEST_DATA_PATH = "C:\\Users\\handi\\Documents\\Estudo\\AI\\01 teste\\test.csv"
OUTPUT_FILE = "submissao_final_ajustada_v9.csv"

# --- 🎯 THRESHOLDS DEFINITIVOS ---
# Baseados na otimização Bayesiana anterior + Correção Manual (Disgust)
THRESHOLDS = {
    'anger': 0.30,
    'disgust': 0.25,  # 🔧 MANUAL FIX: Reduzido de 0.50 (falha) para 0.25 (agressivo)
    'fear': 0.20,
    'joy': 0.10,
    'sadness': 0.50,
    'surprise': 0.25
}

# --- Funções Auxiliares ---
def load_data(path):
    df = pd.read_csv(path)
    # Garante que as colunas existem (mesmo que vazias) para manter estrutura
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
print("⏳ Carregando modelo final...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
trainer = Trainer(model=model)

print(f"📂 Carregando dataset de teste: {TEST_DATA_PATH}")
test_df, label_cols = load_data(TEST_DATA_PATH)
test_set = EmotionDataset(test_df, tokenizer)

# --- 2. Inferência ---
print("🔮 Gerando probabilidades brutas (logits)...")
test_preds = trainer.predict(test_set)
# Aplica Sigmoid para converter logits em probabilidades (0.0 a 1.0)
test_probs = torch.nn.Sigmoid()(torch.tensor(test_preds.predictions)).numpy()

# --- 3. Aplicação dos Thresholds Híbridos ---
print("\n--- 📐 APLICANDO THRESHOLDS DEFINITIVOS ---")
for label, t in THRESHOLDS.items():
    print(f"   -> {label.ljust(10)}: {t}")

final_test_preds = np.zeros(test_probs.shape)

for i, label in enumerate(label_cols):
    t = THRESHOLDS[label]
    # Aplica o corte
    final_test_preds[:, i] = (test_probs[:, i] >= t).astype(int)

# --- 4. Fallback (Forçar Escolha) ---
forced_count = 0
for k in range(len(final_test_preds)):
    if np.sum(final_test_preds[k]) == 0:
        # Se zerou tudo, pega a maior probabilidade (mesmo que baixa)
        idx_max = np.argmax(test_probs[k])
        final_test_preds[k][idx_max] = 1
        forced_count += 1

print(f"\n⚠️ Fallback aplicado em {forced_count} linhas (onde nenhuma classe atingiu o threshold).")

# --- 5. Salvar ---
sub_df = pd.DataFrame(final_test_preds.astype(int), columns=label_cols)
if 'id' in test_df.columns:
    sub_df['id'] = test_df['id']
    sub_df = sub_df[['id'] + label_cols]

sub_df.to_csv(OUTPUT_FILE, index=False)
print(f"🚀 SUCESSO! Arquivo pronto para envio: {OUTPUT_FILE}")
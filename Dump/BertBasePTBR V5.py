import pandas as pd
import torch
import numpy as np
import os
import gc
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score

# ==========================================
# 1. CONFIGURAÇÕES V18 (DATASET V5 - SUPERIOR)
# ==========================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
PER_DEVICE_BATCH_SIZE = 16      
GRADIENT_ACCUMULATION = 2       
EPOCHS = 4            
LEARNING_RATE = 3e-5  

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "resultado_v18_v5")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "modelo_final_v18_v5")
SUBMISSION_FILE = "submissao_final_v18_v5.csv"

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

def load_data(path, is_train=False):
    real_path = find_file(path)
    if not real_path:
        print(f"❌ ERRO: {path} não encontrado.")
        return pd.DataFrame()
    
    print(f"📂 Lendo: {real_path}...")
    df = pd.read_csv(real_path)
    df = df.dropna(subset=['text'])
    for col in label_cols:
        if col not in df.columns: df[col] = 0
    df['labels'] = df[label_cols].values.tolist()
    
    if is_train:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

print("\n--- ETAPA 1: CARGA V5 ---")
# Agora buscamos explicitamente o V5
train_file = "Balanced With Sintetic Data V5.csv"
if not find_file(train_file):
    print("⚠️ V5 não achado. Usando anterior (mas recomendo fortemente o V5)...")
    train_file = "Balanced With Sintetic Data.csv"

train_df = load_data(train_file, is_train=True)
val_df = load_data("dev.csv")
test_df = load_data("test.csv")

if train_df.empty: exit()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

train_set = EmotionDataset(train_df, tokenizer)
val_set = EmotionDataset(val_df, tokenizer)
test_set = EmotionDataset(test_df, tokenizer)

# ==========================================
# 3. TREINAMENTO
# ==========================================
print("\n--- ETAPA 2: TREINO V18 ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=6, problem_type="multi_label_classification"
)
model.to(device)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = torch.nn.Sigmoid()(torch.Tensor(preds)).numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_true = p.label_ids
    return {'f1_macro': f1_score(y_true, y_pred, average='macro')}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
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
    eval_dataset=val_set, tokenizer=tokenizer, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.save_model(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)

# ==========================================
# 4. CALIBRAGEM AUTOMÁTICA
# ==========================================
print("\n--- ETAPA 3: CALIBRAGEM ---")
val_preds_output = trainer.predict(val_set)
val_probs = torch.nn.Sigmoid()(torch.tensor(val_preds_output.predictions)).numpy()
val_labels = val_preds_output.label_ids

best_thresholds = {}
for i, label in enumerate(label_cols):
    best_t = 0.5
    best_f1 = 0
    for t in np.arange(0.1, 0.95, 0.05):
        score = f1_score(val_labels[:, i], (val_probs[:, i] >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t = t
    
    # Com o V5, esperamos que o modelo seja menos "tímido". 
    # Mas a trava de segurança ainda é prudente para o Dev pequeno.
    if best_f1 < 0.05:
        print(f"⚠️ {label}: F1 zerado. Forçando 0.20.")
        best_t = 0.20
    
    best_thresholds[label] = best_t
    print(f"✅ {label.ljust(10)} | Melhor Corte: {best_t:.2f} | F1 Val: {best_f1:.4f}")

# ==========================================
# 5. GERAÇÃO FINAL
# ==========================================
print("\n--- ETAPA 4: GERAÇÃO FINAL ---")
test_preds_output = trainer.predict(test_set)
test_probs = torch.nn.Sigmoid()(torch.tensor(test_preds_output.predictions)).numpy()
final_preds = np.zeros(test_probs.shape)

for i, label in enumerate(label_cols):
    final_preds[:, i] = (test_probs[:, i] >= best_thresholds[label]).astype(int)

# Contagem de Neutros
neutros = 0
for k in range(len(final_preds)):
    if np.sum(final_preds[k]) == 0: neutros += 1
print(f"ℹ️ Linhas mantidas como NEUTRAS: {neutros}")

sub_df = pd.DataFrame(final_preds.astype(int), columns=label_cols)
if 'id' in test_df.columns: sub_df['id'] = test_df['id']
cols = ['id'] + label_cols if 'id' in sub_df.columns else label_cols
sub_df = sub_df[cols]

sub_df.to_csv(SUBMISSION_FILE, index=False)
print("="*50)
print(f"✅ ARQUIVO V18 (DATASET V5) PRONTO: {SUBMISSION_FILE}")
print("="*50)
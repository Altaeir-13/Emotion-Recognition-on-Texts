import pandas as pd
import os
import ftfy  # Importe a biblioteca aqui

def converter_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Carregando arquivos...")
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    go1 = pd.read_csv(os.path.join(base_dir, 'goemotions_1_pt.csv'))
    go2 = pd.read_csv(os.path.join(base_dir, 'goemotions_2_pt.csv'))
    go3 = pd.read_csv(os.path.join(base_dir, 'goemotions_3_pt.csv'))

    # Concatenar os arquivos do GoEmotions
    go_full = pd.concat([go1, go2, go3], ignore_index=True)

    # --- CORREÇÃO DE ENCODING AQUI ---
    print("Corrigindo textos quebrados (ftfy)...")
    # Aplica a correção na coluna 'texto' original do GoEmotions
    # Verifica se é string antes de aplicar para evitar erro em valores nulos
    go_full['texto'] = go_full['texto'].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
    
    # Se o train.csv também tiver erros, descomente a linha abaixo:
    # train_df['text'] = train_df['text'].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
    # ---------------------------------

    mapping = {
        'joy': ['joy', 'admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude', 'love', 'optimism', 'pride', 'relief'],
        'sadness': ['sadness', 'disappointment', 'embarrassment', 'grief', 'remorse'],
        'anger': ['anger', 'annoyance', 'disapproval'],
        'fear': ['fear', 'nervousness'],
        'surprise': ['surprise', 'confusion', 'curiosity', 'realization'],
        'disgust': ['disgust']
    }

    go_processed = pd.DataFrame()
    go_processed['text'] = go_full['texto']

    for target_col, source_cols in mapping.items():
        cols_present = [c for c in source_cols if c in go_full.columns]
        if cols_present:
            go_processed[target_col] = go_full[cols_present].max(axis=1)
        else:
            go_processed[target_col] = 0

    for col in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
        if col not in go_processed.columns:
            go_processed[col] = 0

    # Gerar IDs continuos
    last_id_num = 2226 
    start_id = last_id_num + 1
    new_ids = [f"ptbr_train_track_a_{i:05d}" for i in range(start_id, start_id + len(go_processed))]
    go_processed.insert(0, 'id', new_ids)

    columns_order = ['id', 'text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    train_final = train_df[columns_order].copy()
    go_final = go_processed[columns_order].copy()

    combined_df = pd.concat([train_final, go_final], ignore_index=True)

    combined_df.to_csv(os.path.join(base_dir, 'dataset_unificado.csv'), index=False)
    print(f"Arquivo 'dataset_unificado.csv' criado com sucesso!")

if __name__ == "__main__":
    converter_datasets()
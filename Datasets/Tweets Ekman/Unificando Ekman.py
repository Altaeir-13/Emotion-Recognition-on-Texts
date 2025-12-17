import pandas as pd
import os

def unificar_tudo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Nomes dos arquivos (certifique-se que estão na mesma pasta)
    file_main = os.path.join(base_dir, 'Treino+Goemotions.csv')
    file_ekman = os.path.join(base_dir, 'tweets_ekman_formatado.csv')
    output_file = os.path.join(base_dir, 'dataset_completo_final.csv')

    print("Carregando datasets...")
    # Carrega o dataset principal (Treino + GoEmotions)
    try:
        df_main = pd.read_csv(file_main)
        print(f"Dataset Principal carregado: {len(df_main)} linhas.")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_main}' não encontrado.")
        return

    # Carrega o dataset Ekman formatado
    try:
        df_ekman = pd.read_csv(file_ekman)
        print(f"Dataset Ekman carregado: {len(df_ekman)} linhas.")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_ekman}' não encontrado.")
        return

    # 1. Descobrir o último ID do dataset principal
    # Pega a última linha e extrai o ID
    last_id_str = df_main['id'].iloc[-1]
    # Assume formato 'ptbr_train_track_a_XXXXX'
    try:
        last_id_num = int(last_id_str.split('_')[-1])
        print(f"Último ID encontrado no principal: {last_id_num} ({last_id_str})")
    except Exception as e:
        print(f"Erro ao ler o último ID: {e}")
        return

    # 2. Atualizar os IDs do Ekman para continuar a sequência
    print("Atualizando IDs do Ekman...")
    start_id = last_id_num + 1
    count_ekman = len(df_ekman)
    
    # Gera novos IDs sequenciais
    new_ids = [f"ptbr_train_track_a_{i:05d}" for i in range(start_id, start_id + count_ekman)]
    df_ekman['id'] = new_ids

    # 3. Concatenar (Unir) os dois datasets
    print("Unindo os arquivos...")
    # Garante que as colunas estejam na mesma ordem
    cols = ['id', 'text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    df_final = pd.concat([df_main[cols], df_ekman[cols]], ignore_index=True)

    # 4. Salvar o resultado final
    df_final.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"SUCESSO! Arquivo '{output_file}' criado.")
    print(f"Total de linhas no dataset final: {len(df_final)}")
    print(f"ID Inicial: {df_final['id'].iloc[0]}")
    print(f"ID Final:   {df_final['id'].iloc[-1]}")
    print("-" * 30)

if __name__ == "__main__":
    unificar_tudo()
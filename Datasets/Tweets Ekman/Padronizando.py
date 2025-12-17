import pandas as pd
import os
import ftfy

def corrigir_ekman_robusto():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'tweets_ekman.csv')
    output_file = os.path.join(base_dir, 'tweets_ekman_formatado.csv')
    unificado_path = os.path.join(base_dir, 'dataset_unificado.csv')

    # Sentimentos válidos para validação
    valid_sentiments = {'feliz', 'nojo', 'triste', 'medo', 'raiva'}
    
    data = []
    
    print("Lendo arquivo linha a linha para corrigir erros de aspas...")
    
    # Tenta abrir com utf-8, fallback para latin-1 se der erro
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
         with open(input_file, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Tenta separar pela última vírgula (padrão esperado)
        # rsplit divide da direita para a esquerda
        parts = line.rsplit(',', 1)
        
        found = False
        text_cand = ""
        sentiment_cand = ""

        # Verifica se a separação por vírgula funcionou
        if len(parts) == 2:
            if parts[1].strip().lower() in valid_sentiments:
                text_cand, sentiment_cand = parts
                found = True
        
        # Se não funcionou, tenta ponto e vírgula (algumas linhas parecem usar ;)
        if not found:
            parts = line.rsplit(';', 1)
            if len(parts) == 2:
                if parts[1].strip().lower() in valid_sentiments:
                    text_cand, sentiment_cand = parts
                    found = True
        
        if found:
            # Limpeza do texto: remove aspas soltas que causaram o erro
            text = text_cand.strip().strip('"').strip("'")
            # Corrige problemas de encoding (mojibake)
            text = ftfy.fix_text(text)
            
            data.append({'text': text, 'sentiment': sentiment_cand.strip().lower()})
        else:
            # Ignora linhas que não tenham sentimento válido no final (ex: cabeçalho)
            pass

    df = pd.DataFrame(data)
    print(f"Total de tweets recuperados corretamente: {len(df)}")
    
    # --- Mapeamento para formato Train ---
    target_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for col in target_cols:
        df[col] = 0

    df.loc[df['sentiment'] == 'feliz', 'joy'] = 1
    df.loc[df['sentiment'] == 'nojo', 'disgust'] = 1
    df.loc[df['sentiment'] == 'triste', 'sadness'] = 1
    df.loc[df['sentiment'] == 'medo', 'fear'] = 1
    df.loc[df['sentiment'] == 'raiva', 'anger'] = 1
    
    # --- Geração de IDs ---
    last_id_num = 2226 # Padrão caso não ache o unificado
    
    if os.path.exists(unificado_path):
        try:
            # Lê a última linha do arquivo unificado para pegar o último ID
            with open(unificado_path, 'rb') as f:
                f.seek(0, 2) 
                size = f.tell()
                f.seek(max(size - 1024, 0)) # Pega o finalzinho do arquivo
                last_lines = f.readlines()
                last_line = last_lines[-1].decode().strip()
                
                # Formato esperado: ptbr_train_track_a_XXXXX,...
                last_id_str = last_line.split(',')[0]
                if 'ptbr_train_track_a_' in last_id_str:
                     last_id_num = int(last_id_str.split('_')[-1])
        except Exception:
            print("Aviso: Não foi possível ler o último ID do arquivo unificado. Usando sequência padrão.")
    
    start_id = last_id_num + 1
    new_ids = [f"ptbr_train_track_a_{i:05d}" for i in range(start_id, start_id + len(df))]
    df.insert(0, 'id', new_ids)
    
    # Salvar
    final_cols = ['id', 'text'] + target_cols
    df[final_cols].to_csv(output_file, index=False)
    print(f"Arquivo corrigido salvo em: {output_file}")

if __name__ == "__main__":
    corrigir_ekman_robusto()
import pandas as pd
from sklearn.metrics import f1_score
import os

def avaliar_modelo(caminho_ground_truth, caminho_predicoes):
    # Verificação básica se os arquivos existem antes de tentar ler
    if not os.path.exists(caminho_ground_truth):
        print(f"ERRO: Arquivo não encontrado: {caminho_ground_truth}")
        return
    if not os.path.exists(caminho_predicoes):
        print(f"ERRO: Arquivo não encontrado: {caminho_predicoes}")
        return

    # 1. Carregar os arquivos
    try:
        df_gt = pd.read_csv(caminho_ground_truth)
        df_pred = pd.read_csv(caminho_predicoes)
    except Exception as e:
        print(f"ERRO ao ler os arquivos CSV: {e}")
        return
    
    # 2. Definir as colunas de emoção (ordem importa para o scikit-learn)
    colunas_emocao = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    # 3. Alinhamento dos dados (Garantir que estamos comparando os mesmos IDs na mesma ordem)
    # Ordenamos ambos pelo ID e resetamos o índice para garantir a correspondência linha a linha
    if 'id' in df_gt.columns and 'id' in df_pred.columns:
        df_gt = df_gt.sort_values('id').reset_index(drop=True)
        df_pred = df_pred.sort_values('id').reset_index(drop=True)
        
        # Verificação de segurança: checar se os IDs batem
        if not df_gt['id'].equals(df_pred['id']):
            print("ERRO: Os IDs do arquivo de submissão não correspondem exatamente ao Ground Truth.")
            print("Verifique se há linhas faltando ou IDs incorretos.")
            return
    else:
        print("AVISO: Coluna 'id' não encontrada em um dos arquivos. Assumindo que as linhas já estão na mesma ordem.")

    # 4. Extrair as matrizes de valores (0 e 1)
    # Verifica se as colunas existem
    missing_cols = [c for c in colunas_emocao if c not in df_gt.columns or c not in df_pred.columns]
    if missing_cols:
        print(f"ERRO: As seguintes colunas de emoção estão faltando: {missing_cols}")
        return

    y_real = df_gt[colunas_emocao].values
    y_pred = df_pred[colunas_emocao].values
    
    # 5. Calcular as métricas
    # average='macro' calcula o F1 para cada coluna e tira a média simples (Métrica Oficial)
    macro_f1 = f1_score(y_real, y_pred, average='macro')
    
    # average='micro' calcula o F1 global contando total de tp, fp, fn (útil para visão geral)
    micro_f1 = f1_score(y_real, y_pred, average='micro')
    
    # Calcular F1 por classe individualmente
    f1_por_classe = f1_score(y_real, y_pred, average=None)
    
    # 6. Exibir Resultados
    print("="*40)
    print(f"RESULTADO DA AVALIAÇÃO (SemEval-2025 Task 11)")
    print("="*40)
    print(f"Macro F1-Score (OFICIAL): {macro_f1:.5f}")
    print(f"Micro F1-Score          : {micro_f1:.5f}")
    print("-" * 40)
    print("Desempenho por Emoção (F1-Score):")
    for emocao, score in zip(colunas_emocao, f1_por_classe):
        print(f" - {emocao.ljust(10)}: {score:.5f}")
    print("="*40)

# --- Execução Principal ---
if __name__ == "__main__":
    # Usamos o 'r' antes das aspas para indicar raw string e evitar erros com as barras invertidas do Windows
    arquivo_gabarito = r"C:\Users\handi\Documents\Estudo\AI\Ground Truth.csv"
    arquivo_submissao = r"C:\Users\handi\Documents\Estudo\AI\submissao_otimizada.csv"
    
    print(f"Lendo gabarito de: {arquivo_gabarito}")
    print(f"Lendo submissão de: {arquivo_submissao}")
    print("-" * 40)
    
    avaliar_modelo(arquivo_gabarito, arquivo_submissao)
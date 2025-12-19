import pandas as pd
import random
import numpy as np
import os
import unicodedata

# --- 1. Configurações ---
INPUT_FILE = 'Datasets/All Organic Data.csv'
OUTPUT_FILE = 'Data Generator/Balanced With Sintetic Data V3.csv'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ALVO_MAXIMO = 85549

TARGETS = {
    "anger": ALVO_MAXIMO,
    "disgust": ALVO_MAXIMO,
    "fear": ALVO_MAXIMO,
    "joy": ALVO_MAXIMO,
    "sadness": ALVO_MAXIMO,
    "surprise": ALVO_MAXIMO
}

# --- 2. Vocabulário & Conectivos ---

giras_inicio = ["mano", "vei", "mds", "putz", "nossa", "gente", "na moral", "jesus", "ave"]
giras_fim = ["slk", "tenso", "bizarro", "foda", "demais", "intankavel", "socorro", "tlgd"]

# Conectivos para fundir frases (A mágica do Multilabel)
conectivos_adversativos = ["mas", "porem", "so que", "mas ai", "entretanto"]
conectivos_aditivos = ["e ainda", "e pra piorar", "e tambem", "e alem disso", "e fiquei"]
conectivos_causais = ["porque", "por causa disso", "dai", "entao"]

# Matriz de Emoções Compatíveis (Para não gerar "Feliz e Triste" sem sentido)
# Se a emoção principal for X, a secundária pode ser Y ou Z.
pares_compativeis = {
    "fear": ["sadness", "surprise", "anger"],       # Medo pode vir com tristeza (perda), surpresa (susto) ou raiva (ameaça)
    "disgust": ["anger", "sadness"],                # Nojo traz raiva (indignação) ou tristeza (decepção)
    "anger": ["disgust", "sadness", "fear"],        # Raiva de algo nojento, ou raiva triste (traição)
    "sadness": ["anger", "fear", "disgust"],        # Tristeza com raiva, medo do futuro...
    "surprise": ["joy", "fear", "anger"],           # Surpresa boa ou ruim
    "joy": ["surprise", "sadness"]                  # Alegria com surpresa (ganhar algo) ou agridoce (raro mas existe)
}

# --- 3. Templates (Curtos para facilitar a fusão) ---

# MEDO
medo_curtos = [
    "quase fui assaltado", "tem um cara estranho ali", "ouvi tiros na rua", 
    "fiquei preso no elevador", "vi um vulto", "minha pressao caiu de susto",
    "fui seguido", "tremi de pavor", "panico total"
]
# NOJO
nojo_curtos = [
    "tinha uma barata na comida", "cheiro de podre insuportavel", "pisei no esgoto",
    "vi um rato morto", "banheiro imundo", "me deu ansia de vomito",
    "que nojo dessa atitude", "falsidade me enjoa"
]
# RAIVA
raiva_curtos = [
    "internet caiu de novo", "o onibus nao passa", "fui cobrado errado",
    "meu chefe gritou comigo", "quebrou meu celular", "furei o pneu",
    "falaram mal de mim", "vontade de xingar tudo"
]
# TRISTEZA
tristeza_curtos = [
    "perdi meu dinheiro", "briguei com minha mae", "reprovei na prova",
    "me sinto sozinho", "vontade de chorar", "dia horrivel hoje",
    "saudade de quem ja foi", "coracao partido"
]
# SURPRESA
surpresa_curtos = [
    "nao esperava por essa", "ganhei um sorteio", "achava que era mentira",
    "fiquei de boca aberta", "nunca vi isso antes", "chocado com a noticia"
]
# ALEGRIA (Para misturas)
alegria_curtos = [
    "finalmente consegui", "dia maravilhoso", "passei na prova",
    "ganhei um presente", "to muito feliz", "que alivio"
]

templates_map = {
    "fear": medo_curtos, "disgust": nojo_curtos, "anger": raiva_curtos,
    "sadness": tristeza_curtos, "surprise": surpresa_curtos, "joy": alegria_curtos
}

# --- 4. Funções ---

def normalizar_unicode(txt):
    return unicodedata.normalize("NFKC", txt)

def sujar_texto(txt, intensidade):
    txt = txt.lower()
    if random.random() < 0.3 * intensidade:
        txt = txt.replace("que", "q").replace("voce", "vc").replace("muito", "mt")
    return normalizar_unicode(txt)

def gerar_frase_simples(emotion):
    lista = templates_map.get(emotion, ["frase generica"])
    base = random.choice(lista)
    # Adiciona contexto aleatório às vezes para não ficar só frase curta
    contextos = ["ontem", "hoje cedo", "no trabalho", "em casa", "na rua"]
    if random.random() < 0.3:
        base = f"{base} {random.choice(contextos)}"
    return base

def gerar_frase_multilabel(emo_primaria, emo_secundaria):
    # Pega uma frase de cada emoção
    parte1 = gerar_frase_simples(emo_primaria)
    parte2 = gerar_frase_simples(emo_secundaria)
    
    conectivo = random.choice(conectivos_aditivos + conectivos_causais)
    
    # Estrutura: [Gíria opcional] + Frase 1 + Conectivo + Frase 2
    frase = f"{parte1} {conectivo} {parte2}"
    
    if random.random() < 0.2:
        frase = f"{random.choice(giras_inicio)} {frase}"
    
    return frase

# --- 5. Execução ---

print(f"--- Balanceamento V12 MULTILABEL (Alvo: {ALVO_MAXIMO}) ---")

if not os.path.exists(INPUT_FILE):
    print("❌ Arquivo original não encontrado!")
    exit()

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['text'])
cols_labels = list(TARGETS.keys())
df['total_labels'] = df[cols_labels].sum(axis=1)
df = df[df['total_labels'] > 0] 
df = df.drop(columns=['total_labels'])

# Carrega amostras reais para o modo "mutação" (ainda útil)
amostras_reais = {emo: df[df[emo] == 1]['text'].tolist() for emo in TARGETS}

current_counts = df[list(TARGETS.keys())].sum()
new_rows = []

print("Gerando dados com suporte a múltiplas emoções...")

for emotion_target, target_val in TARGETS.items():
    current = current_counts.get(emotion_target, 0)
    needed = max(0, int(target_val - current))

    print(f"[{emotion_target.upper()}] Falta gerar: {needed}")

    if needed > 0:
        checkpoint = max(1, needed // 10)
        
        for i in range(needed):
            if i % checkpoint == 0 and i > 0: print(".", end="", flush=True)

            intensidade = random.random()
            
            # DECISÃO: Vai ser Single-Label ou Multi-Label?
            # 30% de chance de ser Multilabel (enriquece o dataset)
            eh_multilabel = random.random() < 0.30
            
            row = {col: 0 for col in cols_labels}
            texto_final = ""
            
            if eh_multilabel:
                # Escolhe uma emoção secundária compatível
                possiveis = pares_compativeis.get(emotion_target, [])
                if possiveis:
                    emo_secundaria = random.choice(possiveis)
                    texto_final = gerar_frase_multilabel(emotion_target, emo_secundaria)
                    
                    # AQUI ESTÁ A CORREÇÃO: Marca as DUAS emoções
                    row[emotion_target] = 1
                    row[emo_secundaria] = 1
                else:
                    # Fallback se não tiver par (ex: joy as vezes fica sozinho)
                    texto_final = gerar_frase_simples(emotion_target)
                    row[emotion_target] = 1
            else:
                # Geração Single-Label (Simples ou Mutação)
                if len(amostras_reais[emotion_target]) > 0 and random.random() < 0.4:
                    base = random.choice(amostras_reais[emotion_target])
                    texto_final = sujar_texto(base, intensidade) # Apenas sujeira leve
                else:
                    texto_final = gerar_frase_simples(emotion_target)
                    if random.random() < 0.2: texto_final += f" {random.choice(giras_fim)}"
                
                row[emotion_target] = 1

            # Sujeira final e atribuição
            texto_final = sujar_texto(texto_final, intensidade)
            row['text'] = texto_final
            new_rows.append(row)

        print(" Feito!")

# --- 6. Salvamento ---
print("\nSalvando DataFrame Multilabel...")
synthetic_df = pd.DataFrame(new_rows)

if not synthetic_df.empty:
    synthetic_df['id'] = [f'sint_multi_v12_{i}' for i in range(len(synthetic_df))]
    # Reorganiza colunas
    cols_finais = ['id', 'text'] + cols_labels
    synthetic_df = synthetic_df[cols_finais]

final_df = pd.concat([df, synthetic_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ DATASET V12 PRONTO: {OUTPUT_FILE}")
print(f"Total de linhas: {len(final_df)}")
print("\n--- Contagem Final de Labels (Note que a soma será maior que o nº de linhas) ---")
print(final_df[cols_labels].sum())
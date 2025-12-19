import pandas as pd
import random
import numpy as np
import os
import unicodedata

# --- 1. Configurações ---
INPUT_FILE = 'Datasets/All Organic Data.csv'
OUTPUT_FILE = 'Data Generator/Balanced With Sintetic Data V4.csv'

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

# --- 2. Vocabulário Rico & Contraste ---

# Informais / Gírias
giras_inicio = [
    "mano", "vei", "mds", "putz", "nossa", "gente", "na moral", "jesus", 
    "ave", "oxe", "vixe", "caraca", "pqp", "meu deus", "aff"
]
giras_fim = [
    "slk", "tenso", "bizarro", "foda", "demais", "intankavel", "socorro", 
    "tlgd", "cringe", "paia", "de cria", "tankei foi nada", "deu ruim"
]

# Formais (Para o contraste)
inicio_formal = [
    "gostaria de saber porque", "é inadmissivel que", "solicito providencias pois",
    "estou indignado com", "lamentavel a postura de", "inacreditavel que",
    "por gentileza alguem me explica", "é um absurdo que"
]

# Conectivos Mistos
conectivos = [
    "e ainda por cima", "mas ai", "porem", "entretanto", "so que", 
    "e pra ferrar tudo", "dai", "resultado:", "conclusao:"
]

# --- 3. Templates Expandidos (Situacionais) ---

# MEDO (Violência, Saúde, Sobrenatural, Fobia)
medo_templates = [
    "quase fui assaltado no ponto", "tem um cara estranho me olhando", 
    "ouvi tiros na rua de tras", "fiquei preso no elevador escuro", 
    "vi um vulto no corredor", "minha pressao caiu de susto",
    "fui seguido por um carro preto", "tremi de pavor com o barulho", 
    "panico total desse lugar", "medo de perder o emprego",
    "achei que ia morrer hoje", "coracao disparou do nada"
]

# NOJO (Comida, Insetos, Moral, Lugares)
nojo_templates = [
    "tinha uma barata na minha comida", "cheiro de podre insuportavel aqui", 
    "pisei no esgoto de chinelo", "vi um rato morto na calçada", 
    "banheiro imundo da rodoviaria", "me deu ansia de vomito esse video",
    "que nojo dessa atitude machista", "falsidade me enjoa demais",
    "encontrei cabelo no lanche", "agua com gosto de barro"
]

# RAIVA (Serviços, Tecnologia, Trânsito, Pessoal)
raiva_templates = [
    "internet caiu de novo nessa merda", "o onibus nao passa nunca", 
    "fui cobrado errado no cartao", "meu chefe gritou comigo a toa", 
    "quebrou a tela do meu celular", "furei o pneu no buraco",
    "falaram mal de mim pelas costas", "vontade de xingar tudo hoje",
    "o app do banco travou no pix", "vizinho com som alto as 3h"
]

# TRISTEZA (Luto, Fracasso, Solidão, Decepção)
tristeza_templates = [
    "perdi todo meu dinheiro", "briguei feio com minha mae", 
    "reprovei na prova que estudei", "me sinto sozinho nessa cidade", 
    "vontade de chorar o dia todo", "dia horrivel e cinza",
    "saudade de quem ja se foi", "coracao partido com o termino",
    "ninguem lembrou do meu aniversario", "desanimo total de viver"
]

# SURPRESA (Choque, Revelação, Sorte, Azar)
surpresa_templates = [
    "nao esperava por essa noticia", "ganhei um sorteio do nada", 
    "achava que era mentira mas é real", "fiquei de boca aberta com ele", 
    "nunca vi isso acontecer antes", "chocado com a audacia",
    "descobri um segredo bizarro", "o preço disso ta muito barato"
]

# ALEGRIA (Para misturas multilabel)
alegria_templates = [
    "finalmente consegui a vaga", "dia maravilhoso de sol", 
    "passei na prova dificil", "ganhei um presente lindo", 
    "to muito feliz com isso", "que alivio deu tudo certo"
]

templates_map = {
    "fear": medo_templates, "disgust": nojo_templates, "anger": raiva_templates,
    "sadness": tristeza_templates, "surprise": surpresa_templates, "joy": alegria_templates
}

# Pares compatíveis para Multilabel
pares_compativeis = {
    "fear": ["sadness", "surprise", "anger"],
    "disgust": ["anger", "sadness"],
    "anger": ["disgust", "sadness", "fear"],
    "sadness": ["anger", "fear", "disgust"],
    "surprise": ["joy", "fear", "anger"],
    "joy": ["surprise"]
}

# --- 4. Funções de Geração Avançada ---

def normalizar_unicode(txt):
    return unicodedata.normalize("NFKC", txt)

def sujar_texto(txt, intensidade):
    txt = txt.lower()
    
    # Typos (erros de digitação)
    if random.random() < 0.15 * intensidade:
        vizinhos = {'a': 's', 'e': 'r', 'o': 'p', 'm': 'n'}
        chars = list(txt)
        idx = random.randint(0, len(chars)-1)
        if chars[idx] in vizinhos: chars[idx] = vizinhos[chars[idx]]
        txt = "".join(chars)

    # Internetês
    if random.random() < 0.3 * intensidade:
        txt = txt.replace("que", "q").replace("voce", "vc").replace("muito", "mt").replace("porque", "pq")
    
    return normalizar_unicode(txt)

def misturar_registros(frase_base, emotion):
    """
    Cria o efeito híbrido: Começa formal e termina informal, ou vice-versa.
    """
    tipo = random.choice(["formal_start", "slang_bomb", "normal"])
    
    if tipo == "formal_start":
        # Ex: "É inadmissível que [frase informal]"
        prefixo = random.choice(inicio_formal)
        return f"{prefixo} {frase_base}"
    
    elif tipo == "slang_bomb":
        # Ex: "[Frase normal] tlgd?"
        sufixo = random.choice(giras_fim)
        return f"{frase_base} {sufixo}"
        
    return frase_base

def gerar_frase_multilabel(emo_primaria, emo_secundaria):
    # Pega templates base
    t1 = random.choice(templates_map.get(emo_primaria, ["erro"]))
    t2 = random.choice(templates_map.get(emo_secundaria, ["erro"]))
    
    # Conecta
    conectivo = random.choice(conectivos)
    frase = f"{t1} {conectivo} {t2}"
    
    # Aplica mistura de registros (Formal/Informal)
    if random.random() < 0.4:
        frase = misturar_registros(frase, emo_primaria)
    
    # Gíria de início (opcional)
    if random.random() < 0.25:
        frase = f"{random.choice(giras_inicio)} {frase}"
        
    return frase

# --- 5. Execução Principal ---

print(f"--- Balanceamento V13 HÍBRIDO (Alvo: {ALVO_MAXIMO}) ---")

if not os.path.exists(INPUT_FILE):
    print("❌ Arquivo original não encontrado!")
    exit()

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['text'])
cols_labels = list(TARGETS.keys())
df['total_labels'] = df[cols_labels].sum(axis=1)
df = df[df['total_labels'] > 0] 
df = df.drop(columns=['total_labels'])

# Carrega amostras reais
amostras_reais = {emo: df[df[emo] == 1]['text'].tolist() for emo in TARGETS}

current_counts = df[list(TARGETS.keys())].sum()
new_rows = []

for emotion_target, target_val in TARGETS.items():
    current = current_counts.get(emotion_target, 0)
    needed = max(0, int(target_val - current))

    print(f"[{emotion_target.upper()}] Gerando: {needed}")

    if needed > 0:
        checkpoint = max(1, needed // 10)
        
        for i in range(needed):
            if i % checkpoint == 0 and i > 0: print(".", end="", flush=True)

            intensidade = random.random()
            
            # 35% de chance de ser Multilabel (Mais complexidade)
            eh_multilabel = random.random() < 0.35
            
            row = {col: 0 for col in cols_labels}
            texto_final = ""
            
            if eh_multilabel:
                possiveis = pares_compativeis.get(emotion_target, [])
                if possiveis:
                    emo_sec = random.choice(possiveis)
                    texto_final = gerar_frase_multilabel(emotion_target, emo_sec)
                    row[emotion_target] = 1
                    row[emo_sec] = 1
                else:
                    texto_final = random.choice(templates_map[emotion_target])
                    row[emotion_target] = 1
            else:
                # Single Label: Pode usar mutação real ou template novo
                if len(amostras_reais[emotion_target]) > 0 and random.random() < 0.3:
                    base = random.choice(amostras_reais[emotion_target])
                    texto_final = misturar_registros(base, emotion_target) # Aplica híbrido na real tbm
                else:
                    base = random.choice(templates_map[emotion_target])
                    texto_final = misturar_registros(base, emotion_target)
                
                row[emotion_target] = 1

            # Sujeira final (textura humana)
            texto_final = sujar_texto(texto_final, intensidade)
            
            # Validação anti-duplicata simples (sufixo invisível se precisar)
            if i > 0 and i % 50 == 0: 
                texto_final += random.choice([" .", " !", "...", ""])

            row['text'] = texto_final
            new_rows.append(row)

        print(" OK!")

# --- 6. Salvamento ---
print("\nSalvando V13...")
synthetic_df = pd.DataFrame(new_rows)

if not synthetic_df.empty:
    synthetic_df['id'] = [f'sint_v13_{i}' for i in range(len(synthetic_df))]
    cols_finais = ['id', 'text'] + cols_labels
    synthetic_df = synthetic_df[cols_finais]

final_df = pd.concat([df, synthetic_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ DATASET V13 PRONTO: {OUTPUT_FILE}")
print(f"Total de linhas: {len(final_df)}")
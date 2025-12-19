import pandas as pd
import random
import numpy as np
import os
import unicodedata

# --- 1. Configurações ---
INPUT_FILE = 'Datasets/All Organic Data.csv'
OUTPUT_FILE = 'Data Generator/Balanced With Sintetic Data V5.csv'

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

# --- 2. O Vocabulário da Internet Brasileira (Caos Total) ---

# Reações Curtas (Twitter/Insta style)
reacoes_inicio = [
    "mano", "vei", "gente", "mds", "putz", "nossa", "na moral", "pqp", "caraca", 
    "vixe", "ave", "oxe", "uai", "eita", "jesus", "senhor", "amado?", "bicho",
    "olha", "cara", "entao", "tipo assim", "sinceramente", "desculpa mas"
]

reacoes_fim = [
    "slk", "tenso", "foda", "demais", "intankavel", "socorro", "tlgd", "cringe", 
    "paia", "de cria", "tankei foi nada", "deu ruim", "perdi tudo", "berro", 
    "fala serio", "me poupe", "se manca", "nao tanko", "foi de base", "ja era",
    "esquece", "receba", "gratiluz", "sem or", "ranço", "nojo", "bizarro"
]

# Intensificadores de Emoção (Com erros propositais)
intensificadores = [
    "muito", "mt", "mto", "demais", "pacas", "pra caramba", "pra krl", "super", 
    "mega", "hiper", "absurdamente", "horrores", "infinito", "real", "oficial"
]

# --- 3. Templates de "Rede Social" (Inspirados em Tweets e Comentários) ---

# MEDO (Ansiedade, Crime, Sobrenatural, Futuro)
# Ex: "o bostil n é para amadores", "medo de sair na rua"
templates_fear = [
    "o brasil nao é para amadores slk", "sensação que vai dar merda isso ai", 
    "quase fui de arrasta pra cima hoje", "medo real oficial disso acontecer",
    "minha ansiedade atacou forte vendo isso", "coração disparou aqui mds",
    "nao passo nessa rua nem que me paguem", "olha a cara desse sujeito socorro",
    "tem um vulto na minha foto ou eu to doido?", "ouvi barulho de tiro e me joguei no chao",
    "medo de abrir a fatura do cartao esse mes", "a policia parou o onibus geral gelou",
    "esse lugar tem uma energia pesada credo", "sonhei com cobra dizem que é traição",
    "tremendo mais que vara verde", "pavor de barata voadora quem mais?"
]

# NOJO (Repulsa, Comida, Política, Comportamento)
# Ex: "embuste", "lixo humano", "comida gore"
templates_disgust = [
    "que ser humano podre me da ansia", "olha o estado desse banheiro imundo", 
    "encontrei um cabelo na comida que nojo", "ranço eterno dessa pessoa", 
    "me da agonia so de olhar pra isso", "que atitude ridicula e nojenta",
    "vontade de vomitar lendo esse comentario", "a comida tava com gosto de azedo",
    "vi um rato no restaurante eca", "esse politico me da asco",
    "povo mal educado que joga lixo no chao", "cheiro de suor nesse onibus ta tenso",
    "que video satisfatorio sqn que nojo", "parece que saiu do esgoto credo"
]

# RAIVA (Hate, Reclamação, Serviços, Troll)
# Ex: "serviço lixo", "odio", "cancelado"
templates_anger = [
    "que odio desse app que nao funciona", "empresa lixo nunca mais compro", 
    "o onibus demorou uma vida que inferno", "vontade de mandar tudo pra aquele lugar", 
    "meu chefe é um sem noção to puto", "fui cobrado indevidamente palhaçada", 
    "internet da oi/vivo/claro ta uma porcaria", "cala a boca militante chato",
    "quem foi o genio que inventou isso?", "perdi meu tempo vendo esse video lixo",
    "furei o pneu no buraco da prefeitura inutil", "roubaram minha encomenda correios lixo",
    "o vizinho ta com som alto de novo que raiva", "deslike nesse video horrivel"
]

# TRISTEZA (Desabafo, Sadboy/Sadgirl, Luto, Fracasso)
# Ex: "gatilho", "depre", "chorando"
templates_sadness = [
    "só queria sumir um pouco ta dificil", "bata uma tristeza do nada hoje", 
    "saudade do meu ex mesmo ele nao valendo nada", "chorei largado vendo esse filme", 
    "me sinto sozinho nessa cidade grande", "reprovei de novo me sinto um inutil", 
    "o dinheiro acabou antes do mes que fase", "ninguem lembrou do meu niver #chateado",
    "que final triste nao tava preparado", "meu cachorro morreu to sem chao",
    "ver fotos antigas da um aperto no peito", "fiquei mal com essa noticia"
]

# SURPRESA (Plot Twist, Choque, Fofoca, Sorte)
# Ex: "chocada", "passada", "mentira"
templates_surprise = [
    "chocado com esse final nem imaginava", "mentira que vcs tao juntos???", 
    "ganhei um sorteio no insta nem acredito", "o preço disso ta muito barato corre", 
    "fiquei de cara com a audacia dele", "jamais pensaria nisso vindo de vc",
    "olha o tamanho desse bicho mds", "descobri que sou corno pelo facebook",
    "o plot twist carpado desse video", "gente olha quem apareceu do nada",
    "meu video viralizou to tremendo", "a nota da prova saiu e eu passei mds"
]

# ALEGRIA (Hype, Conquista, Elogio - para misturas multilabel)
templates_joy = [
    "finalmente uma noticia boa glória", "passei no vestibular chupa mundo", 
    "que dia lindo pra ir na praia", "amei esse video salvou meu dia", 
    "to muito feliz com essa conquista", "melhor show da minha vida",
    "comida boa e barata amo muito", "sextou com s de saudade sqn"
]

templates_map = {
    "fear": templates_fear, "disgust": templates_disgust, "anger": templates_anger,
    "sadness": templates_sadness, "surprise": templates_surprise, "joy": templates_joy
}

# Matriz de Mistura Multilabel (Compatibilidade Semântica)
pares_compativeis = {
    "fear": ["sadness", "surprise", "anger"],
    "disgust": ["anger", "sadness"],
    "anger": ["disgust", "sadness", "fear"],
    "sadness": ["anger", "fear", "disgust"],
    "surprise": ["joy", "fear", "anger"],
    "joy": ["surprise"]
}

# --- 4. Motor de "Naturalidade Caótica" ---

def normalizar_unicode(txt):
    return unicodedata.normalize("NFKC", txt)

def simular_digitacao_ruim(txt, intensidade):
    """
    Simula erros comuns de quem digita rápido no celular.
    """
    txt = txt.lower()
    
    # 1. Troca de letras vizinhas (Dedo gordo)
    teclado_vizinhos = {
        'a': 's', 's': 'a', 'e': 'r', 'r': 'e', 'o': 'p', 'p': 'o', 
        'n': 'm', 'm': 'n', 'b': 'v', 'v': 'b', 'c': 'x', 'x': 'c'
    }
    
    if random.random() < 0.15 * intensidade:
        chars = list(txt)
        idx = random.randint(0, len(chars)-1)
        if chars[idx] in teclado_vizinhos:
            chars[idx] = teclado_vizinhos[chars[idx]]
        txt = "".join(chars)

    # 2. Supressão de vogais/Internetês (Abreviações)
    if random.random() < 0.4 * intensidade:
        subs = {
            "voce": "vc", "que": "q", "porque": "pq", "muito": "mt", "tambem": "tb", 
            "beijo": "bj", "hoje": "hj", "quando": "qnd", "mesmo": "msm", 
            "comigo": "cmg", "nao": "n", "estou": "to", "para": "pra", "gente": "gnt"
        }
        words = txt.split()
        new_words = [subs.get(w, w) for w in words]
        txt = " ".join(new_words)

    # 3. Repetição de letras pra ênfase (ex: "muitoooo")
    if random.random() < 0.1 * intensidade:
        words = txt.split()
        if words:
            idx = random.randint(0, len(words)-1)
            if len(words[idx]) > 2:
                words[idx] += words[idx][-1] * random.randint(1, 3)
            txt = " ".join(words)
            
    # 4. Caps Lock Rage (Gritar na internet)
    if random.random() < 0.1 * intensidade:
        txt = txt.upper()
    
    # 5. Pontuação Caótica
    if random.random() < 0.5 * intensidade:
        # Remove pontuação correta
        txt = txt.replace(".", "").replace(",", "")
        # Adiciona pontuação excessiva de rede social
        if random.random() < 0.3:
            txt += random.choice(["...", "!!!", "???", " kkkk", " rs", " aff"])

    return normalizar_unicode(txt)

def gerar_frase_hibrida(emo_primaria, emo_secundaria=None):
    # Se for multilabel
    if emo_secundaria:
        t1 = random.choice(templates_map.get(emo_primaria, ["erro"]))
        t2 = random.choice(templates_map.get(emo_secundaria, ["erro"]))
        
        conectivos_web = ["e tipo", "mas ai", "so que", "e pra piorar", "dai", "e ainda", "afinal"]
        conectivo = random.choice(conectivos_web)
        
        frase = f"{t1} {conectivo} {t2}"
    else:
        # Single label
        frase = random.choice(templates_map.get(emo_primaria, ["erro"]))

    # Injeção de Gírias (Inicio/Fim)
    if random.random() < 0.3:
        frase = f"{random.choice(reacoes_inicio)} {frase}"
    if random.random() < 0.3:
        frase = f"{frase} {random.choice(reacoes_fim)}"

    return frase

# --- 5. Execução Principal ---

print(f"--- Balanceamento V14 SOCIAL MEDIA (Alvo: {ALVO_MAXIMO}) ---")

if not os.path.exists(INPUT_FILE):
    print("❌ Arquivo original não encontrado!")
    exit()

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['text'])
cols_labels = list(TARGETS.keys())
df['total_labels'] = df[cols_labels].sum(axis=1)
df = df[df['total_labels'] > 0] 
df = df.drop(columns=['total_labels'])

# Amostras reais para mutação (mantém um pouco da base orgânica)
amostras_reais = {emo: df[df[emo] == 1]['text'].tolist() for emo in TARGETS}

current_counts = df[list(TARGETS.keys())].sum()
new_rows = []

print("Gerando tweets, posts e comentários sintéticos...")

for emotion_target, target_val in TARGETS.items():
    current = current_counts.get(emotion_target, 0)
    needed = max(0, int(target_val - current))

    print(f"[{emotion_target.upper()}] Gerando: {needed}")

    if needed > 0:
        checkpoint = max(1, needed // 10)
        
        for i in range(needed):
            if i % checkpoint == 0 and i > 0: print(".", end="", flush=True)

            # Intensidade define o quão "internetês" o texto vai ficar
            # 0.0 = Texto formal | 1.0 = Twitter puro suco do caos
            intensidade = random.uniform(0.2, 1.0) 
            
            # 35% de chance de ser Multilabel (Comentário complexo)
            eh_multilabel = random.random() < 0.35
            
            row = {col: 0 for col in cols_labels}
            texto_final = ""
            
            if eh_multilabel:
                possiveis = pares_compativeis.get(emotion_target, [])
                if possiveis:
                    emo_sec = random.choice(possiveis)
                    texto_final = gerar_frase_hibrida(emotion_target, emo_sec)
                    row[emotion_target] = 1
                    row[emo_sec] = 1
                else:
                    texto_final = gerar_frase_hibrida(emotion_target)
                    row[emotion_target] = 1
            else:
                # Single Label: Pode usar mutação real ou template novo
                # Usa mais templates (70%) que amostra real pra garantir a "vibe" nova
                if len(amostras_reais[emotion_target]) > 0 and random.random() < 0.3:
                    base = random.choice(amostras_reais[emotion_target])
                    texto_final = base # Mutação acontece no `simular_digitacao_ruim`
                else:
                    texto_final = gerar_frase_hibrida(emotion_target)
                
                row[emotion_target] = 1

            # Aplica a sujeira de digitação (O segredo da naturalidade)
            texto_final = simular_digitacao_ruim(texto_final, intensidade)
            
            # Anti-duplicação burra (adiciona espaços invisíveis ou pontuação extra)
            if i % 100 == 0: 
                texto_final += random.choice(["", ".", "!", " :)", " :("])

            row['text'] = texto_final
            new_rows.append(row)

        print(" OK!")

# --- 6. Salvamento ---
print("\nSalvando V14...")
synthetic_df = pd.DataFrame(new_rows)

if not synthetic_df.empty:
    synthetic_df['id'] = [f'sint_social_v14_{i}' for i in range(len(synthetic_df))]
    cols_finais = ['id', 'text'] + cols_labels
    synthetic_df = synthetic_df[cols_finais]

final_df = pd.concat([df, synthetic_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ DATASET V14 (SOCIAL MEDIA REALISM) PRONTO: {OUTPUT_FILE}")
print(f"Total de linhas: {len(final_df)}")
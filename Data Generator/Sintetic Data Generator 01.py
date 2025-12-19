import pandas as pd
import random
import numpy as np
import os
import unicodedata

# --- 1. Configurações & Reprodutibilidade ---
INPUT_FILE = 'Datasets/All Organic Data.csv'
OUTPUT_FILE = 'Data Generator/Balanced With Sintetic Data.csv'

# SEED FIXA
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Alvo: Tudo nivelado pela classe majoritária (Joy)
ALVO_MAXIMO = 85549

TARGETS = {
    "anger": ALVO_MAXIMO,
    "disgust": ALVO_MAXIMO,
    "fear": ALVO_MAXIMO,
    "joy": ALVO_MAXIMO,
    "sadness": ALVO_MAXIMO,
    "surprise": ALVO_MAXIMO
}

# --- 2. Vocabulário Expandido & Templates Novos ---

giras_inicio = ["mano", "vei", "mds", "putz", "nossa", "aff", "caraca", "gente", "na moral", "pqp", "jesuss"]
giras_fim = ["slk", "tenso", "bizarro", "foda", "demais", "serio", "real", "intankavel", "me poupe", "inacreditavel", "socorro"]

mapa_intensificadores = {
    "muito": ["mt", "pra caramba", "demais", "super", "mega", "horrores"],
    "demais": ["mt", "pra krl", "muito", "exagerado", "infinim"],
    "pouco": ["meio", "quase nada", "tipo nada"]
}

# --- TEMPLATES MASSIVOS PARA FEAR E DISGUST ---

templates_medo = [
    # Clássicos
    "quase fui assaltado {lugar}", "tem um cara estranho me olhando", "ouvindo barulho de tiro {lugar}",
    "o {parente} nao atende o telefone", "sensação ruim de que algo vai acontecer", "meu coracao ta disparado",
    "travei totalmente na hora", "medo de morrer queimado", "panico de andar a noite", "vi um vulto {lugar}",
    # Novos (Físicos/Sintomas)
    "minha mao ta suando frio", "sinto que to sendo seguido", "gelou minha espinha agora",
    "nao consigo nem respirar de susto", "pernas tremendo horrores", "ta me dando falta de ar esse lugar",
    "meu estomago ta embrulhado de nervoso", "coracao saindo pela boca",
    # Novos (Psicológicos/Situacionais)
    "medo real de perder meu emprego", "pavor de ficar sozinho {lugar}", "sonhei que estava caindo",
    "sensacao de morte iminente", "ansiedade a milhao com isso", "medo do que vai acontecer amanha",
    "trauma disso pra sempre", "nao passo la nem pagando", "cagaço monstro disso",
    "tenho fobia disso mds", "fiquei branco de susto"
]

templates_nojo = [
    # Clássicos
    "encontrei um cabelo na comida", "cheiro de podre vindo {lugar}", "vi uma barata voando",
    "ansia de vomito com esse video", "o {parente} mastiga de boca aberta", "que estomago embrulhado",
    "gente que cospe no chao", "banheiro publico imundo", "hipocrisia desse povo", "verme na fruta q eu ia comer",
    # Novos (Físicos/Sensoriais)
    "me sinto sujo so de ler isso", "vontade de lavar os olhos com candida", "revirou meu estomago real",
    "cheiro de morte nesse lugar", "agua parada com larva eca", "grudento e nojento credo",
    "tem gosto de lixo isso", "parece vomito essa comida", "quase gorfei aqui",
    # Novos (Morais/Sociais)
    "que lixo de atitude", "nojo fisico dessa pessoa", "ranço eterno disso",
    "repulsa total por quem faz isso", "me da asco ver gente assim", "podridao humana",
    "falsidade me enjoa", "gente porca me irrita", "imundo demais slk"
]

templates_raiva = [
    "que odio desse {parente}", "internet caindo toda hora", "vontade de quebrar tudo",
    "o {parente} me tira do serio", "atendimento lixo desse lugar", "indignado com essa mentira",
    "paguei caro e veio estragado", "fura fila na minha frente", "nao tenho paciencia pra isso",
    "esse governo so faz merda", "transito infernal {lugar}", "meu sangue ferve com isso",
    "quero matar um hoje", "povo folgado do caramba"
]

templates_tristeza = [
    "saudade de quem ja se foi", "vontade de chorar do nada", "me sentindo um lixo hoje",
    "o {parente} me decepcionou", "noticia ruim logo cedo", "dia cinza e sem graça",
    "ninguem lembra de mim", "sensação de vazio no peito", "queria sumir um pouco",
    "coração partido com isso", "depre batendo forte", "lagrima descendo sozinha"
]

templates_surpresa = [
    "nao esperava por essa", "chocado com essa noticia", "mentira que isso aconteceu",
    "como assim o {parente} fez isso?", "fiquei de boca aberta agora", "surreal o que eu vi {lugar}",
    "nem nos meus sonhos imaginava isso", "caraca que plot twist", "jamais pensaria nisso",
    "to passado com isso", "bugou minha mente agora"
]

perguntas_retoricas = [
    "serio que vcs acham isso normal?", "como tem gente que aguenta isso?",
    "ate quando a gente vai aceitar isso?", "sera que so eu fico mal com isso?",
    "alguem me explica pq isso existe?", "onde esse mundo vai parar?"
]

respostas_curtas = [
    "simplesmente doentio.", "nao da pra defender.", "que coisa bizarra.",
    "sem condicoes.", "inacreditavel.", "nojo real.", "medo genuino.", "triste demais.", "que raiva.",
    "sem palavras."
]

lugares = ["na rua", "no onibus", "no metro", "em casa", "no trabalho", "na faculdade", "no uber", "na fila", "no shopping"]
parentes = ["minha mae", "meu pai", "meu irmao", "o vizinho", "o motorista", "esse povo", "meu chefe", "minha tia"]

emojis_map = {
    "fear": ["😰", "😱", "💀", "😨", "🥶", "🫨"],
    "disgust": ["🤢", "🤮", "💩", "🦗", "😖", "🗑️"],
    "sadness": ["😢", "😞", "💔", "🥀", "😩", "🥺"],
    "anger": ["😡", "🤬", "😤", "🖕", "🙄", "🔥"],
    "surprise": ["😲", "😮", "🤯", "😶", "wdym", "👀"]
}

# --- 3. Funções Otimizadas ---

def normalizar_unicode(txt):
    return unicodedata.normalize("NFKC", txt)

def sujar_texto(txt, intensidade):
    """Aplica ruído controlado."""
    txt = txt.lower()

    # Repetição enfática (Novo truque de diversidade)
    # Ex: "medo medo" ou "nojento nojento"
    if intensidade > 0.7 and random.random() < 0.15:
        palavras = txt.split()
        if len(palavras) > 2:
            idx = random.randint(0, len(palavras)-1)
            palavras.insert(idx, palavras[idx]) # Duplica uma palavra
            txt = " ".join(palavras)

    # Pontuação
    if random.random() < 0.4 * intensidade:
        txt = txt.replace(".", "").replace(",", "").replace("?", "").replace("!", "")

    # Internetês
    if random.random() < 0.5 * intensidade:
        txt = txt.replace("voce", "vc").replace("porque", "pq").replace("muito", "mt").replace("que", "q")

    # Typos
    if random.random() < 0.2 * intensidade:
        vizinhos = {'a': 's', 'e': 'r', 'o': 'p', 'm': 'n', 'b':'v', 'c':'x'}
        lista_chars = list(txt)
        if lista_chars:
            idx = random.randint(0, len(lista_chars)-1)
            char_alvo = lista_chars[idx]
            if char_alvo in vizinhos:
                lista_chars[idx] = vizinhos[char_alvo]
            txt = "".join(lista_chars)
    
    return normalizar_unicode(txt)

def mutar_frase_real(frase_original, intensidade):
    palavras = frase_original.split()
    nova_frase = palavras.copy()

    for i, word in enumerate(palavras):
        if word.lower() in mapa_intensificadores and random.random() < 0.5:
            nova_frase[i] = random.choice(mapa_intensificadores[word.lower()])

    if intensidade > 0.6:
        if random.random() < 0.3: nova_frase.insert(0, random.choice(giras_inicio))
        if random.random() < 0.3: nova_frase.append(random.choice(giras_fim))

    texto_final = " ".join(nova_frase)
    return sujar_texto(texto_final, intensidade)

def gerar_via_template(emotion, intensidade):
    base = "erro"
    if emotion == "fear": base = random.choice(templates_medo)
    elif emotion == "disgust": base = random.choice(templates_nojo)
    elif emotion == "anger": base = random.choice(templates_raiva)
    elif emotion == "sadness": base = random.choice(templates_tristeza)
    elif emotion == "surprise": base = random.choice(templates_surpresa)

    if base == "erro": return "frase generica"

    while "{lugar}" in base: base = base.replace("{lugar}", random.choice(lugares), 1)
    while "{parente}" in base: base = base.replace("{parente}", random.choice(parentes), 1)

    roll = random.random()
    if roll < 0.12:
        base = f"{random.choice(perguntas_retoricas)} {base}"
    elif roll < 0.25:
        base = f"{base} {random.choice(respostas_curtas)}"

    frase = sujar_texto(base, intensidade)

    if intensidade > 0.3 and random.random() < 0.35:
        emoji_list = emojis_map.get(emotion, [])
        if emoji_list:
            frase += f" {random.choice(emoji_list)}"

    return frase

# --- 4. Execução Principal ---

print(f"--- Balanceamento V8 FINAL (Alvo: {ALVO_MAXIMO}) ---")

if not os.path.exists(INPUT_FILE):
    print("❌ Arquivo original não encontrado!")
    exit()

df = pd.read_csv(INPUT_FILE)

# 1. LIMPEZA OBRIGATÓRIA (Correção do Sanity Check)
# Remove linhas onde texto é NaN
df = df.dropna(subset=['text'])
# Remove linhas que não têm NENHUMA label (soma = 0)
# Isso elimina o ruído "Dallas não jogou..." que não tinha label.
cols_labels = list(TARGETS.keys())
df['total_labels'] = df[cols_labels].sum(axis=1)
original_len = len(df)
df = df[df['total_labels'] > 0] # Mantém apenas se tiver pelo menos 1 emoção
filtered_len = len(df)
print(f"🧹 Limpeza realizada: {original_len - filtered_len} linhas inúteis (sem label) removidas.")
df = df.drop(columns=['total_labels']) # Limpa coluna auxiliar

# Prepara amostras reais para mutação
amostras_reais = {}
for emo in TARGETS.keys():
    textos = df[df[emo] == 1]['text'].tolist()
    amostras_reais[emo] = textos

frases_geradas = {emo: set() for emo in TARGETS}
current_counts = df[list(TARGETS.keys())].sum()
new_rows = []

for emotion, target in TARGETS.items():
    current = current_counts.get(emotion, 0)
    needed = max(0, int(target - current))

    print(f"\n[{emotion.upper()}] Atual: {current} | Meta: {target} | Gerando: {needed}")

    if needed > 0:
        checkpoint = max(1, needed // 10)
        qtd_reais = len(amostras_reais.get(emotion, []))
        
        # Ajuste: Se tivermos poucas amostras reais (Fear/Disgust), 
        # confiamos mais nos templates novos que são diversos.
        if qtd_reais > 5000: prob_mutacao = 0.8
        elif qtd_reais > 500: prob_mutacao = 0.6
        else: prob_mutacao = 0.4 # Reduzi para usar mais os templates novos

        count_gerados = 0
        for i in range(needed):
            if i % checkpoint == 0 and i > 0: print(".", end="", flush=True)

            intensidade = random.random()
            tentativas = 0
            texto_final = ""

            while True:
                tentativas += 1
                usar_mutacao = (qtd_reais > 0) and (random.random() < prob_mutacao)

                if usar_mutacao:
                    frase_base = random.choice(amostras_reais[emotion])
                    texto_final = mutar_frase_real(frase_base, intensidade)
                else:
                    texto_final = gerar_via_template(emotion, intensidade)

                if texto_final not in frases_geradas[emotion]:
                    frases_geradas[emotion].add(texto_final)
                    break
                
                if tentativas > 5:
                    texto_final = gerar_via_template(emotion, intensidade)
                    break 
            
            if len(texto_final.split()) < 3:
                 texto_final = gerar_via_template(emotion, intensidade)

            row = {
                "text": texto_final,
                "anger": 0, "disgust": 0, "fear": 0,
                "joy": 0, "sadness": 0, "surprise": 0
            }
            row[emotion] = 1
            new_rows.append(row)
            count_gerados += 1

        print(f" Concluído! ({count_gerados})")
        diversidade = len(frases_geradas[emotion]) / max(1, count_gerados)
        print(f"Diversidade: {diversidade:.2f}")

# --- 5. Salvamento ---
print("\nCriando DataFrame final...")
synthetic_df = pd.DataFrame(new_rows)

if not synthetic_df.empty:
    synthetic_df['id'] = [f'sintetico_v8_{i}' for i in range(len(synthetic_df))]

cols_order = ['id', 'text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
cols_finais = [c for c in cols_order if c in df.columns or c in synthetic_df.columns]

if not synthetic_df.empty:
    synthetic_df = synthetic_df[cols_finais]

final_df = pd.concat([df, synthetic_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print("="*50)
print(f"✅ DATASET FINAL V8 PRONTO: {OUTPUT_FILE}")
print(f"Total de linhas: {len(final_df)}")
print("\n--- Distribuição Final ---")
print(final_df[["anger", "disgust", "fear", "joy", "sadness", "surprise"]].sum())
print("="*50)
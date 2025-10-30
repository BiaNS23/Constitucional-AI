import pandas as pd
from collections import Counter, defaultdict
import re
import json
import os

# =======================
# Configuração
# =======================
HATEBR_PATH = "../data/HateBR/dataset/HateBR.csv"
HATEBRX_PATH = "../data/HateBR/dataset/HateBRXplain/HateBRXplain.csv"
OUTPUT_FILE = "../data/hatebr_keywords.json"

# =======================
# Funções auxiliares
# =======================
def limpar_texto(txt):
    """Limpa e normaliza texto."""
    txt = str(txt)
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", "", txt)
    return txt.lower().strip()

def extrair_racional(rationale_str):
    """Divide e normaliza os rationales anotados."""
    if pd.isna(rationale_str):
        return []
    partes = re.split(r"[;,.]", str(rationale_str))
    return [limpar_texto(p) for p in partes if len(p.strip()) > 2]

# =======================
# Carregar datasets
# =======================
if not os.path.exists(HATEBR_PATH) or not os.path.exists(HATEBRX_PATH):
    raise FileNotFoundError("❌ Verifique os caminhos dos arquivos HateBR e HateBRXplain.")

print("📥 Lendo HateBR e HateBRXplain...")
df_main = pd.read_csv(HATEBR_PATH)
df_xplain = pd.read_csv(HATEBRX_PATH)

print(f"✅ HateBR: {len(df_main)} linhas | HateBRXplain: {len(df_xplain)} linhas")

# =======================
# Combinar dados
# =======================
# hateBRXplain tem colunas: id, comment, offensive_label, rationales_annotator1, rationales_annotator2
# vamos extrair os rationales diretamente
rationales_por_id = defaultdict(list)

for _, row in df_xplain.iterrows():
    if int(row.get("offensive_label", 0)) == 1:  # só ofensivos
        r1 = extrair_racional(row.get("rationales_annotator1", ""))
        r2 = extrair_racional(row.get("rationales_annotator2", ""))
        rationales_por_id[row["id"]].extend(r1 + r2)

# =======================
# Gerar lista de palavras e expressões ofensivas
# =======================
todas_palavras = []
for termos in rationales_por_id.values():
    todas_palavras.extend(termos)

contagem = Counter(todas_palavras)

# Remove ruído e palavras muito genéricas
stopwords_basicas = {"essa", "esse", "mais", "toda", "todo", "aquela", "aquele", "coisa"}
filtradas = [p for p, c in contagem.items() if c >= 2 and p not in stopwords_basicas]

# =======================
# Agrupar por tipo simples (exemplo: insultos de gênero, ideologia, etc.)
# =======================
categorias = {
    "ideologia": ["comunista", "petista", "esquerdista", "bolsominion"],
    "sexo_genero": ["puta", "vagabunda", "vadia", "burra", "mulherzinha"],
    "xenofobia": ["nordestino", "estrangeiro", "argentino"],
    "geral": filtradas
}

# =======================
# Salvar resultado final
# =======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(categorias, f, ensure_ascii=False, indent=2)

print(f"✅ Arquivo gerado com sucesso: {OUTPUT_FILE}")
print(f"Total de termos únicos: {len(filtradas)}")

import json
import csv
from datetime import datetime

# =======================
# Configuração
# =======================
CRITIC_FILE = "../data/criticized_prompts_fs_pp_V4.json"
EVAL_FILE = "../data/evaluation_set.json"
LOG_FILE = "../data/results_history.csv"

# =======================
# Carregar dados
# =======================
with open(CRITIC_FILE, "r", encoding="utf-8") as f:
    model_data = json.load(f)

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Criar dicionário de classificações do modelo
# {prompt: classificacao}
model_dict = {d["prompt_nocivo"]: d.get("classificacao", "neutro") for d in model_data}

# =======================
# Avaliar desempenho
# =======================
total = len(eval_data)
acertos = 0
falsos_positivos = 0
falsos_negativos = 0

for item in eval_data:
    prompt = item["prompt"]
    esperado = item["esperado"]
    previsto = model_dict.get(prompt, "neutro")

    if previsto == esperado:
        acertos += 1
    elif previsto == "nocivo" and esperado == "neutro":
        falsos_positivos += 1
    elif previsto == "neutro" and esperado == "nocivo":
        falsos_negativos += 1

acuracia = acertos / total if total > 0 else 0

# =======================
# Mostrar resultados
# =======================
print(f"\n📊 Resultados da avaliação:")
print(f"Total de exemplos: {total}")
print(f"Acurácia: {acuracia*100:.2f}%")
print(f"Falsos positivos: {falsos_positivos}")
print(f"Falsos negativos: {falsos_negativos}")

# =======================
# Salvar no log CSV
# =======================
try:
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Cabeçalho (apenas se arquivo vazio)
        if csvfile.tell() == 0:
            writer.writerow(["Data", "Acurácia", "Falsos Positivos", "Falsos Negativos", "Total"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{acuracia:.3f}",
            falsos_positivos,
            falsos_negativos,
            total
        ])
    print(f"\n📝 Resultados registrados em {LOG_FILE}")
except Exception as e:
    print(f"Erro ao salvar log: {e}")

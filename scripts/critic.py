import json

# Caminhos dos arquivos
GENERATED_FILE = "../data/generated_responses.json"
PRINCIPLES_FILE = "../data/principlesV1.json"
OUTPUT_FILE = "../data/criticized_responses.json"

# 1️⃣ Carregar respostas geradas
with open(GENERATED_FILE, "r", encoding="utf-8") as f:
    responses = json.load(f)

# 2️⃣ Carregar princípios
with open(PRINCIPLES_FILE, "r", encoding="utf-8") as f:
    principles = json.load(f)["principles"]

# 3️⃣ Para cada resposta, verificar “superficialmente” se viola algo
criticized = []

for item in responses:
    prompt = item["prompt"]
    response = item["response"]

    # Exemplo: marcar "violou" se conter certas palavras de risco (simplificação!)
    # Na vida real você chamaria o modelo de novo para fazer a crítica.
    issues = []

    for principle in principles:
        if "violência" in principle["principle"]:
            if "bomba" in response or "matar" in response:
                issues.append("Inofensividade violada")

        if "precisas" in principle["principle"]:
            if "eu inventei" in response:
                issues.append("Honestidade violada")

    criticized.append({
        "prompt": prompt,
        "response": response,
        "criticisms": issues if issues else ["OK"]
    })

# 4️⃣ Salvar resultado
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(criticized, f, ensure_ascii=False, indent=2)

print(f"Críticas salvas em {OUTPUT_FILE}")

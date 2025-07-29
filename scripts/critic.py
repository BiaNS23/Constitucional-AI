import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ===== Configurações =====
MODEL_ID = "TucanoBR/Tucano-160m"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256

GENERATED_FILE = "../data/generated_responses.json"
OUTPUT_FILE = "../data/criticized_responses.json"

print(f"Usando dispositivo: {DEVICE}")

# ===== Carregar modelo e tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Modelo Tucano 160m carregado")

# Pipeline de geração
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE.startswith("cuda") else -1
)

# ===== Carregar dados =====
with open(GENERATED_FILE, "r", encoding="utf-8") as f:
    responses = json.load(f)

# ===== Criar prompts de crítica =====
prompts_to_critique = []
original_items = []

for item in responses:
    prompt = item["prompt"]
    response = item["response"]

    critique_prompt = f"""
Resposta do assistente:
{response}

Identifique maneiras específicas pelas quais a resposta acima é nociva, antiética, racista, sexista, tóxica, perigosa ou ilegal. Caso não haja problemas, diga: "A resposta está adequada aos princípios constitucionais."
""".strip()

    prompts_to_critique.append(critique_prompt)
    original_items.append(item)

# ===== Processar em batches =====
criticized = []

for i in range(0, len(prompts_to_critique), BATCH_SIZE):
    batch_prompts = prompts_to_critique[i:i + BATCH_SIZE]
    batch_items = original_items[i:i + BATCH_SIZE]

    results = generator(
		batch_prompts, 
		max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=True,
		temperature=0.7)

    for original, generated in zip(batch_items, results):
        criticized.append({
            "prompt": original["prompt"],
            "response": original["response"],
            "critique": generated[0]["generated_text"].strip()
        })

    print(f"Processado batch {i//BATCH_SIZE + 1}/{(len(prompts_to_critique) + BATCH_SIZE - 1) // BATCH_SIZE}")

# ===== Salvar resultados =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(criticized, f, ensure_ascii=False, indent=2)

print(f"\n✅ Críticas salvas em {OUTPUT_FILE}")


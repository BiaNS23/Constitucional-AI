import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# ===== Configurações =====
MODEL_ID = "TucanoBR/Tucano-160m"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8
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
generated_critiques = []

print("Gerando críticas...")
for i in tqdm(range(0, len(prompts_to_critique), batch_size)):
    batch_prompts = prompts_to_critique[i:i + batch_size]
    outputs = generator(
        batch_prompts,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    # Extrai apenas o texto gerado (removendo o prompt original)
    for prompt, output in zip(batch_prompts, outputs):
        full_text = output[0]["generated_text"]
        generated_part = full_text[len(prompt):].strip()
        generated_critiques.append(generated_part)

# Salva as críticas junto com os prompts/respostas originais
criticized = []
for item, critique in zip(original_items, generated_critiques):
    criticized.append({
        "prompt": item["prompt"],
        "response": item["response"],
        "critique": critique
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(criticized, f, indent=2, ensure_ascii=False)

print("Críticas geradas e salvas com sucesso em", OUTPUT_FILE)

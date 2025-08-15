from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

# =======================
# Configuração
# =======================
MODEL_ID = "TucanoBR/Tucano-2b4-Instruct"
INPUT_FILE = "../data/responses.json"  # entrada
OUTPUT_FILE = "../data/criticized_responses.json"  # saída
BATCH_SIZE = 8  # ajuste conforme sua GPU/CPU

PREFIX = (
    "Identifique maneiras específicas pelas quais a resposta a seguir é "
    "prejudicial, antiética, racista, sexista, tóxica, perigosa ou ilegal: "
)

# =======================
# Checa o device
# =======================
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use: {'cuda:0' if device==0 else 'cpu'}")

# =======================
# Carregar modelo e tokenizer
# =======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# =======================
# Carregar prompts
# =======================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

prompts_list = data["prompts"]

# =======================
# Preparar prompts
# =======================
prefixed_prompts = [f"{PREFIX}{p}" for p in prompts_list]

# =======================
# Gerar respostas em batch
# =======================
results = []
for i in range(0, len(prefixed_prompts), BATCH_SIZE):
    batch = prefixed_prompts[i:i+BATCH_SIZE]
    batch_outputs = generator(
        batch,
        max_new_tokens=200,
        do_sample=True,
        temperature=1,
        top_k=50,
        top_p=0.9
    )
    
    for prompt, output in zip(batch, batch_outputs):
        results.append({
            "prompt": prompt,
            "response": output[0]["generated_text"]
        })

# =======================
# Salvar em JSON
# =======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Respostas salvas em {OUTPUT_FILE}")

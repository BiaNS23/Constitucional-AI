from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

# =======================
# Configuração
# =======================
MODEL_ID = "TucanoBR/Tucano-2b4-Instruct"
INPUT_FILE = "../data/generated_responses_2b4_v2.json" # arquivo de entrada (respostas longas)
OUTPUT_FILE = "../data/short_responses.json"  # saída com resumos
BATCH_SIZE = 8
MAX_NEW_TOKENS = 80  # resumos curtos

PREFIX = (
    "Resuma a resposta a seguir de forma breve, em no máximo 3 frases, "
    "mantendo apenas a essência.\n\nRESPOSTA:\n\"\"\"\n"
)
SUFFIX = "\n\"\"\"\n\nRESUMO:"

# =======================
# Checa o device
# =======================
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use: {'cuda:0' if device==0 else 'cpu'}")

# =======================
# Carregar modelo/tokenizer
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
# Carregar respostas originais
# =======================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "prompts" in data:
    responses_list = data["prompts"]  # caso simples
elif isinstance(data, list):
    responses_list = [d.get("response", "") for d in data if isinstance(d, dict)]
else:
    raise ValueError("Formato de INPUT_FILE não reconhecido.")

# =======================
# Preparar prompts
# =======================
summary_prompts = [f"{PREFIX}{resp}{SUFFIX}" for resp in responses_list if resp]

# =======================
# Gerar resumos em batch
# =======================
short_responses = []
for i in range(0, len(summary_prompts), BATCH_SIZE):
    batch = summary_prompts[i:i+BATCH_SIZE]
    batch_outputs = generator(
        batch,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        return_full_text=False
    )

    for output in batch_outputs:
        short_responses.append({"response": output[0]["generated_text"].strip()})

# =======================
# Salvar em JSON
# =======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(short_responses, f, ensure_ascii=False, indent=2)

print(f"✅ Resumos salvos em {OUTPUT_FILE} (total: {len(short_responses)})")

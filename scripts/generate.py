from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

# =======================
# Configuração
# =======================
MODEL_ID = "TucanoBR/Tucano-2b4-Instruct"
INPUT_FILE = "../data/harmful_prompts.json"
OUTPUT_FILE = "../data/generated_responses_2b4.json"

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

# para debug
print("Eaw data keys:", data.keys())

prompts_list = data["prompts"]
print("Primeiro prompt:", prompts_list[0])


# =======================
# Gerar respostas
# =======================
results = generator(
	prompts_list,
        max_new_tokens=200,
        do_sample=True,
        temperature=1,
	top_k=50,
	top_p=0.9
    )

# =======================
# Montagem de saída
# =======================
outputs = []
for prompt_text, result in zip(prompts_list, results):
	outputs.append({
	   "prompt": prompt_text,
	   "response": result[0]["generated_text"]
	})

# =======================
# Salvar resultado
# =======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=2, ensure_ascii=False)

print(f"Respostas geradas e salvas em {OUTPUT_FILE}")

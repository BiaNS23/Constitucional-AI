from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

# =======================
# Configuração
# =======================
MODEL_ID = "TucanoBR/Tucano-160m-Instruct"
INPUT_FILE = "../data/constitucional.json"
OUTPUT_FILE = "../data/generated_responses.json"

# =======================
# Carregar modelo e tokenizer
# =======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Usa GPU se estiver tudo certo com CUDA
)

# =======================
# Carregar prompts
# =======================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    prompts = json.load(f)

outputs = []

# =======================
# Gerar respostas
# =======================
for item in prompts:
    prompt_text = f"<instruction>{item['principle']}</instruction>"
    result = generator(
        prompt_text,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
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

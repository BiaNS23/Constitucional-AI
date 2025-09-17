from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

# =======================
# Configuração
# =======================
MODEL_ID = "TucanoBR/Tucano-2b4-Instruct"
INPUT_FILE = "../data/harmful_prompts.json"
FEW_SHOT_FILE = "../data/few_shot_examples.json"
OUTPUT_FILE = "../data/criticized_prompts_fewshot_post.json"
BATCH_SIZE = 8

# =======================
# Device
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
# Carregar exemplos few-shot
# =======================
with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
    few_shot_data = json.load(f)

FEW_SHOT_EXAMPLES = ""
for ex in few_shot_data["examples"]:
    FEW_SHOT_EXAMPLES += f"Exemplo:\nPergunta: {ex['pergunta']}\nCrítica: {ex['critica']}\n\n"

FEW_SHOT_EXAMPLES += "Agora, analise a pergunta a seguir e explique de que maneira ela pode ser prejudicial:\n"

# =======================
# Carregar prompts nocivos
# =======================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

prompts_list = data["prompts"]

# =======================
# Função de pós-processamento
# =======================
def post_process(critica: str) -> str:
    """Aplica regras simples para reforçar classificação nociva."""
    keywords = ["risco", "perigo", "prejudicial", "ilegal", "antiético", "tóxico"]
    for k in keywords:
        if k in critica.lower():
            return critica + " (⚠️ Classificado como nocivo por regra)"
    return critica

# =======================
# Gerar críticas em batch
# =======================
results = []
for i in range(0, len(prompts_list), BATCH_SIZE):
    batch = prompts_list[i:i+BATCH_SIZE]

    few_shot_prompts = [f"{FEW_SHOT_EXAMPLES}\nPergunta: {p}\nCrítica:" for p in batch]

    batch_outputs = generator(
        few_shot_prompts,
        max_new_tokens=120,
        do_sample=False,
        temperature=1,
        top_k=50,
        top_p=0.9
    )

    for original_prompt, input_with_examples, output in zip(batch, few_shot_prompts, batch_outputs):
        raw_response = output[0]["generated_text"].replace(input_with_examples, "").strip()
        processed_response = post_process(raw_response)
        results.append({
            "prompt_nocivo": original_prompt,
            "critica": processed_response
        })

# =======================
# Salvar resultados
# =======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Críticas com few-shot + pós-processamento salvas em {OUTPUT_FILE}")

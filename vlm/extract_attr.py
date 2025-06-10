import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ---------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
PROMPT = "a photo of a person, single person, single face, ultra detailed, raw photo, realistic face"
OUTPUT_FILE = "attributes_schema.json"

# --------------- OPENBIAS PROMPT TEMPLATE ---------------- #
def build_openbias_prompt(user_prompt: str):
    return f"""
Your task is to identify **the top 5 most relevant semantic biases** that may arise when generating an image using the following prompt in a generative model like Stable Diffusion.

For each bias:
- Give a short name (`name`)
- List 2–5 possible values (`classes`)
- Provide one relevant question that can help identify it in the generated image (`question`)
- Indicate whether the bias is already revealed in the prompt (`present_in_prompt`)

Respond ONLY as a JSON list with 5 entries. No explanation.

### EXAMPLE

Prompt: "A picture of a doctor"

[
  {{
    "name": "person gender",
    "classes": ["male", "female"],
    "question": "What is the gender of the doctor?",
    "present_in_prompt": false
  }},
  {{
    "name": "person ethnicity",
    "classes": ["white", "black", "indian", "asian"],
    "question": "What is the race of the doctor?",
    "present_in_prompt": false
  }}
]

### YOUR TASK

Prompt: "{user_prompt}"

[
""".strip()

# --------------- STEP: ATTRIBUTE EXTRACTION ---------------- #
def get_attributes_from_llm(prompt, model, tokenizer):
    full_prompt = build_openbias_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("🔁 RAW LLM OUTPUT (truncated):\n", decoded[:1000])

    try:
        after_task = decoded.split("### YOUR TASK", 1)[-1]
        blocks = re.findall(r"\{.*?\}", after_task, re.DOTALL)

        parsed = []
        for block in blocks:
            try:
                entry = json.loads(block)
                if all(k in entry for k in ["name", "classes", "question"]) and not entry.get("present_in_prompt", False):
                    parsed.append(entry)
            except Exception:
                continue

        attributes = {entry["name"]: entry["classes"] for entry in parsed}
        print(f"✅ Extracted {len(attributes)} attributes.")
        return attributes
    except Exception as e:
        print("⚠️ Parsing failed:", e)
        return {}

# --------------- MODEL LOADING ---------------- #
def load_llm():
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# --------------- MAIN ---------------- #
if __name__ == "__main__":
    model, tokenizer = load_llm()
    attributes_dict = get_attributes_from_llm(PROMPT, model, tokenizer)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(attributes_dict, f, indent=2)
    print(f"📄 Top 5 attribute schema saved to {OUTPUT_FILE}")

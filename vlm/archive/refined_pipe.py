import os
import re
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Blip2Processor,
    Blip2ForConditionalGeneration
)

# ---------------- Load Mistral ---------------- #
def load_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        quantization_config=config
    )
    return model, tokenizer

def run_llm(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_list(response):
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            parsed = eval(match.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, str)]
        except:
            pass
    return None

def extract_top3_numbered(response):
    lines = response.strip().splitlines()
    return [
        line.split(".", 1)[1].strip().lower()
        for line in lines
        if re.match(r"^\d+\.\s*\w+", line)
    ][:5]

def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    return processor, model

# def ask_choice_question(image, attr, choices, processor, model, device):
#     q = f"Which of the following best describes the {attr} in this image: {', '.join(choices)}? Respond with one word only."
#     inputs = processor(images=image, text=f"Question: {q} Answer:", return_tensors="pt").to(device)
#     output = model.generate(**inputs, max_new_tokens=10, pad_token_id=processor.tokenizer.eos_token_id)
#     response = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
#     for c in choices:
#         if c in response:
#             return c
#     return response.split()[0]

def ask_choice_question(image, attr, choices, processor, model, device):
    # Construct a clearer VQA-style prompt
    q = f"What is the most appropriate {attr} for this image? Choose from: {', '.join(choices)}. Respond with one word only."
    
    inputs = processor(images=image, text=f"Question: {q} Answer:", return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=10, pad_token_id=processor.tokenizer.eos_token_id)
    response = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()

    # Normalize and correct known fuzzy matches
    for c in choices:
        if c in response:
            return c

    # Heuristic recovery for gender specifically
    if attr == "gender":
        if "fem" in response:
            return "female"
        elif "mal" in response:
            return "male"
        elif "trans" in response:
            return "transgender"
        elif "non" in response:
            return "non-binary"
        elif "neutral" in response:
            return "neutral"

    return response.split()[0]  # fallback


# ---------------- Main Pipeline ---------------- #
def main(prompt, image_folder, output_json="image_attributes.json", hint=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔧 Loading Mistral...")
    llm, tokenizer = load_llm()

    # --- Step 1: Get Attributes ---
    print("🤖 Getting attribute list...")
    if hint is None:
        hint = "Focus on demographic traits (like age, gender, ethnicity) for people. Use appearance/breed traits for animals."
    attr_query = (
        "You're a vision-language expert. Given an image generation prompt, return a Python list of 20 coarse, high-level, "
        "visually grounded attributes that would help meaningfully cluster the resulting images into diverse semantic groups. "
        f"Hint: {hint}\n\nPrompt: {prompt}\n\nOutput ONLY a Python list (no explanation):"
    )
    attr_response = run_llm(attr_query, llm, tokenizer, device)
    print("\n🧾 Raw LLM Attribute Response:\n", attr_response)
    attrs = extract_list(attr_response)
    print("📌 Inferred Attributes:", attrs)

    # --- Step 2: Top-5 Selection ---
    print("🎯 Selecting top 5...")
    top_query = (
        f"You are a semantic reasoning expert.\nGiven the image generation prompt: \"{prompt}\"\n"
        f"and the following list of high-level attributes:\n[{', '.join(attrs)}]\n\n"
        "Choose the 5 most informative and diverse attributes for categorizing images.\n"
        "Return your answer in the numbered list format as given here:\n1. <attribute>\n2. <attribute>\n3. <attribute>\n4. <attribute>\n5. <attribute>\n(No explanation)"
    )
    top_response = run_llm(top_query, llm, tokenizer, device)
    print("\n🧾 Raw LLM Top-5 Response:\n", top_response)
    top3 = extract_top3_numbered(top_response)
    print("🏆 Top attributes:", top3)

    # --- Step 3: Choices for Attributes ---
    attr_to_values = {}
    for attr in top3:
        choice_query = (
            f"You are building a generic image understanding system.\n"
            f"For the attribute '{attr}', list 5 diverse and coarse-grained values that can appear in realistic images.\n"
            "Respond only with a Python list of 5 lowercase strings (no explanation):"
        )
        choice_response = run_llm(choice_query, llm, tokenizer, device)
        print(f"\n🧾 Raw Choices for {attr}:\n", choice_response)
        values = extract_list(choice_response)
        attr_to_values[attr] = values or []
        print(f"📊 {attr}: {attr_to_values[attr]}")

    # --- Step 4: BLIP Inference ---
    print("🖼️ Loading BLIP-2...")
    processor, blip_model = load_blip2()

    results = {}
    image_files = sorted(f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png")))[:100]
    print("🔍 Annotating images...")
    for fname in tqdm(image_files):
        image = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        results[fname] = {}
        for attr in top3:
            val = ask_choice_question(image, attr, attr_to_values[attr], processor, blip_model, device)
            results[fname][attr] = val

    # --- Step 5: Summary of Unique Attribute Values ---
    attribute_sets = {attr: set() for attr in top3}
    for image_attrs in results.values():
        for attr, val in image_attrs.items():
            attribute_sets[attr].add(val)

    print("\n📊 Unique Values per Attribute:")
    for attr, vals in attribute_sets.items():
        print(f"# {attr}: {sorted(vals)}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved final output to {output_json}")

# Example call
main(
    prompt="a photo of a person, single person, single face, ultra detailed, raw photo, realistic face",
    image_folder="/home/teja/three/vardhan/new_faces/data/images"
)

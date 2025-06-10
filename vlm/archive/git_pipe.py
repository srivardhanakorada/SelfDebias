import os
import re
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoProcessor,
    GitForCausalLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ---------------- Load GIT Captioning ---------------- #
def load_git(device):
    model_id = "microsoft/git-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = GitForCausalLM.from_pretrained(model_id).to(device)
    return processor, model

def generate_caption(image, processor, model, device):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    return caption

# ---------------- Load Mistral LLM ---------------- #
def load_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map={"": 0},
        trust_remote_code=True, quantization_config=config
    )
    return tokenizer, model

def run_llm(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_list(response):
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            parsed = eval(match.group(0))
            if isinstance(parsed, list):
                return [x.strip().lower() for x in parsed if isinstance(x, str)]
        except:
            return []
    return []

# ---------------- Main Pipeline ---------------- #
def main(image_folder="path/to/images", output_json="final_results.json", captions_json="captions_debug.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


    # --- Fixed Attributes ---
    attributes = ["gender", "ethnicity", "age"]

    print("🔧 Loading models...")
    git_processor, git_model = load_git(device)
    llm_tokenizer, llm_model = load_llm()

    # --- Step 1: Caption Images ---
    print("🖼️ Generating captions...")
    captions = {}
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))])
    for fname in tqdm(image_files):
        image = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        caption = generate_caption(image, git_processor, git_model, device)
        captions[fname] = caption
    with open(captions_json, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"✅ Captions saved to {captions_json}")

    # --- Step 2: Get Possible Values ---
    print("🎯 Getting possible values...")
    attr_to_values = {}
    for attr in attributes:
        prompt = (
            f"For the attribute '{attr}', give only a valid Python list of 5 lowercase values "
            f"that describe realistic diversity in the attribute. Do not include any explanation. "
            f"Respond ONLY with the list."
        )
        response = run_llm(prompt, llm_model, llm_tokenizer, device)
        print(f"\n🧾 Raw LLM Output for '{attr}':\n{response}")
        values = extract_list(response)
        attr_to_values[attr] = values
    print("✅ Possible Values:")
    for attr, values in attr_to_values.items():
        print(f"# {attr}: {values}")

    # --- Step 3: Infer Values from Captions ---
    print("🧠 Inferring attributes from captions...")
    results = {}
    for fname in tqdm(image_files):
        caption = captions[fname]
        results[fname] = {}
        for attr in attributes:
            values = attr_to_values.get(attr, [])
            if not values:
                results[fname][attr] = "unknown"
                continue
            prompt = (
                f"You are an expert in image understanding.\n"
                f"Caption: \"{caption}\"\n"
                f"Select the most appropriate value for '{attr}' from the following list: {values}.\n"
                f"Respond with only one word from the list (no explanation)."
            )
            response = run_llm(prompt, llm_model, llm_tokenizer, device)
            choice = [v for v in values if v in response.lower()]
            results[fname][attr] = choice[0] if choice else "unknown"

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved final output to {output_json}")

if __name__ == "__main__":
    main(
        image_folder="/home/teja/three/vardhan/new_faces/data/images",
        output_json="final_results.json",
        captions_json="captions_debug.json"
    )

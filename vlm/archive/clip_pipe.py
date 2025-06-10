import os
import re
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)

# ---------------- Load LLM (Mistral) ---------------- #
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
                return [x.strip().lower() for x in parsed if isinstance(x, str)]
        except:
            pass
    return []

# ---------------- Load CLIP ---------------- #
def load_clip():
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

# ---------------- Main Pipeline ---------------- #
def main(image_folder, prompt, attributes=["gender", "ethnicity", "age"], output_json="clip_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🔧 Loading models...")
    clip_model, clip_processor = load_clip()
    clip_model.to(device)
    
    llm, tokenizer = load_llm()

    # Step 1: Get 5 values per attribute
    print("🎯 Getting possible values...")
    attr_to_values = {}
    for attr in attributes:
        query = (
            f"You are building an image analysis system.\n"
            f"For the attribute '{attr}', give a Python list of 5 coarse-level values relevant to realistic photos.\n"
            "List must be lowercase and diverse. No explanation. Format: ['a', 'b', 'c', 'd', 'e']"
        )
        resp = run_llm(query, llm, tokenizer, device)
        values = extract_list(resp)
        attr_to_values[attr] = values
    print("✅ Possible Values:")
    for k, v in attr_to_values.items():
        print(f"# {k}: {v}")

    # Step 2: CLIP-based assignment
    image_files = sorted(f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png")))
    results = {}
    attr_sets = {k: set() for k in attributes}

    print("🔍 Annotating images...")
    for fname in tqdm(image_files):
        img_path = os.path.join(image_folder, fname)
        image = Image.open(img_path).convert("RGB")
        results[fname] = {}

        for attr in attributes:
            choices = attr_to_values[attr]
            texts = [f"{attr}: {v}" for v in choices]

            inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            best_idx = probs.argmax().item()
            best_val = choices[best_idx]
            results[fname][attr] = best_val
            attr_sets[attr].add(best_val)

    # Step 3: Output results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved results to {output_json}")

    print("\n📊 Unique Values per Attribute:")
    for attr in attributes:
        print(f"# {attr}: {sorted(attr_sets[attr])}")

# Example run
main(
    image_folder="/home/teja/three/vardhan/new_faces/data/images",
    prompt="a photo of a person, single person, single face, ultra detailed, raw photo, realistic face"
)

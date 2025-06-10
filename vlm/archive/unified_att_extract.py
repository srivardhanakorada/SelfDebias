import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# -------- Load Local LLM on Single GPU -------- #
def load_local_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    quant_config = BitsAndBytesConfig(
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
        quantization_config=quant_config
    )
    return model, tokenizer

# -------- Infer Key Attributes from Prompt -------- #
def get_local_llm_attributes(prompt, model, tokenizer, device):
    system_prompt = (
        "You are an expert in analyzing realistic images. "
        "Given the generation prompt, infer the most dominant high-level attributes "
        "along which the generated images will vary. Output ONLY a Python list of 5 lowercase strings."
    )
    user_prompt = f"{system_prompt}\n\nPrompt: {prompt}\n\nAttributes:"
    inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    extracted = decoded.split("Attributes:")[-1].strip()
    try:
        if extracted.startswith("[") and extracted.endswith("]"):
            return eval(extracted)[:5]
        return [line.strip("-• ").lower() for line in extracted.splitlines() if line.strip()][:5]
    except Exception:
        raise ValueError(f"Failed to parse attribute list from LLM output:\n{decoded}")

# -------- BLIP-2 Image Attribute Query -------- #
def query_image_attr(image, question, processor, model, device):
    prompt = f"Question: {question}\nAnswer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20, pad_token_id=processor.tokenizer.eos_token_id)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

# -------- Main Attribute Extraction Pipeline -------- #
def main():
    image_folder = "pet_data/images"
    generation_prompt = "Photo of a pet"
    output_file = "raw_attributes.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    llm_model, llm_tokenizer = load_local_llm()
    attribute_keys = get_local_llm_attributes(generation_prompt, llm_model, llm_tokenizer, device)
    print("🔍 Extracted attributes:", attribute_keys)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16,
        device_map={"": 0}
    )

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:100]

    results = {}

    for fname in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, fname)
        image = Image.open(image_path).convert("RGB")
        attr_dict = {}

        for attr in attribute_keys:
            question = f"Under {attr} categorization, which category does this image fall into? Only give category name."
            answer = query_image_attr(image, question, processor, blip_model, device)
            attr_dict[attr] = answer.strip().lower()

        results[fname] = attr_dict

    # Save results
    with open(output_file, "w") as f:
        for fname, attrs in results.items():
            f.write(f"{fname}:\n")
            for k, v in attrs.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"✅ Attributes saved to {output_file}")

if __name__ == "__main__":
    main()

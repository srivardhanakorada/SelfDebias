import os
import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm


# ----------- Step 1: Use Local LLM to Infer Attributes ----------- #
def get_local_llm_attributes(prompt, model, tokenizer, device):
    system_prompt = (
        "You are an expert in analyzing large sets of realistic vehicle photos. "
        "Given the generation prompt, infer the most dominant high-level attributes along which the generated images "
        "are likely to vary. These should be coarse, semantic-level traits that help group vehicles meaningfully."
        "Do not mention details from the prompt. "
        "Output ONLY a Python list of 5 lowercase strings."
    )
    user_prompt = f"{system_prompt}\n\nPrompt: {prompt}\n\nAttributes:"
    inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    extracted = decoded.split("Attributes:")[-1].strip()
    try:
        if extracted.startswith("[") and extracted.endswith("]"):
            attributes = eval(extracted)
        else:
            attributes = [line.strip("-• ").lower() for line in extracted.splitlines() if line.strip()]
    except Exception:
        raise ValueError(f"Failed to parse attribute list from LLM output:\n{decoded}")

    return attributes[:5]


# ----------- Step 2: Load Local LLM ----------- #
def load_local_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    return model, tokenizer


# ----------- Step 3: Load BLIP-2 ----------- #
def query_image_attr(image, question, processor, model, device):
    prompt = f"Question: {question}\nAnswer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


# ----------- Main Pipeline ----------- #
def main():
    generation_prompt = "a photo of a vehicle"
    image_folder = "vhl_data/images"
    output_file = "raw_attributes.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Local LLM
    llm_model, llm_tokenizer = load_local_llm()
    attribute_keys = get_local_llm_attributes(generation_prompt, llm_model, llm_tokenizer, device)
    print("🔑 Top 5 Attributes for Segregation:", attribute_keys)

    # Load BLIP-2
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Process Images
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:100]

    results = {}
    for fname in tqdm(image_files):
        image_path = os.path.join(image_folder, fname)
        image = Image.open(image_path).convert("RGB")
        attr_dict = {}
        for attr in attribute_keys:
            question = f"Under {attr} categorization, which category does this image fall into? Only give category name"
            answer = query_image_attr(image, question, processor, blip_model, device)
            attr_dict[attr] = answer
        results[fname] = attr_dict

    # Save Output
    with open(output_file, "w") as f:
        for fname, attr_dict in results.items():
            f.write(f"{fname}:\n")
            for k, v in attr_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"✅ Done. Raw BLIP-2 attributes saved to: {output_file}")


if __name__ == "__main__":
    main()

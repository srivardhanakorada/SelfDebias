import os
import re
import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm

# ----------- Load Local LLM ----------- #
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

# ----------- Use Local LLM to Infer Attribute Axes ----------- #
def get_local_llm_attributes(prompt, model, tokenizer, device):
    system_prompt = (
        "You are an expert in analyzing large sets of realistic human portraits. "
        "Given the generation prompt, infer the most dominant high-level attributes along which the generated images "
        "are likely to vary. These should be coarse, semantic-level traits that help group images meaningfully — "
        "such as demographics, appearance, or expression style. Do not mention details from the prompt. "
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

# ----------- Query Image Attributes ----------- #
def query_image_attr(image, question, processor, model, device):
    prompt = f"Question: {question}\nAnswer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

# ----------- Coarsen Raw Answers ----------- #
def coarsen_value(attribute_name, fine_answer, model, tokenizer, device):
    prompt = (
        f"Given the following attribute and value, return only a single coarse-grained category label as one lowercase word.\n"
        f"Attribute: {attribute_name}\n"
        f"Value: {fine_answer}\n"
        f"Category:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(attribute_name,fine_answer,response)  ## Response = Given the following attribute and value, return only a single coarse-grained category label as one lowercase word.
# Attribute: 1. age
# Value: young woman
# Category: 1. age
    return response.strip().split()[0].lower()

# ----------- Main Pipeline ----------- #
def main():
    generation_prompt = "a photo of a person"
    image_folder = "/home/teja/three/vardhan/data/images"
    raw_output_file = "raw_attributes.txt"
    coarse_output_file = "coarse_attributes.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Local LLM and BLIP-2
    llm_model, llm_tokenizer = load_local_llm()
    attribute_keys = get_local_llm_attributes(generation_prompt, llm_model, llm_tokenizer, device)
    print("🔑 Top 5 Attributes for Segregation:", attribute_keys)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Query Attributes
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:10]

    raw_results, coarse_results = {}, {}
    for fname in tqdm(image_files):
        image_path = os.path.join(image_folder, fname)
        image = Image.open(image_path).convert("RGB")
        attr_dict_raw, attr_dict_coarse = {}, {}
        for attr in attribute_keys:
            question = f"Under {attr} categorization, which category does this image fall into? Only give category name"
            answer = query_image_attr(image, question, processor, blip_model, device)
            attr_dict_raw[attr] = answer
            coarse = coarsen_value(attr, answer, llm_model, llm_tokenizer, device)
            attr_dict_coarse[attr] = coarse
        raw_results[fname] = attr_dict_raw
        coarse_results[fname] = attr_dict_coarse

    # Save Raw Attributes
    with open(raw_output_file, "w") as f:
        for fname, attr_dict in raw_results.items():
            f.write(f"{fname}:\n")
            for k, v in attr_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    # Save Coarse Attributes
    with open(coarse_output_file, "w") as f:
        for fname, attr_dict in coarse_results.items():
            f.write(f"{fname}:\n")
            for k, v in attr_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"✅ Saved raw attributes to: {raw_output_file}")
    print(f"✅ Saved coarse attributes to: {coarse_output_file}")

if __name__ == "__main__":
    main()

# import os
# import re
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import (
#     Blip2Processor,
#     Blip2ForConditionalGeneration,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )

# # ---------- Load LLM ----------
# def load_llm():
#     model_id = "mistralai/Mistral-7B-Instruct-v0.1"
#     config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
#                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
#                                                  device_map={"": 0}, trust_remote_code=True,
#                                                  quantization_config=config)
#     return model, tokenizer

# # ---------- Run LLM ----------
# def run_llm(prompt, model, tokenizer, device):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     output = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # ---------- Extract first valid Python list from LLM output ----------
# def extract_list_from_response(response):
#     print("🔍 Raw LLM response:\n", response)
#     try:
#         matches = re.findall(r"\[[^\[\]]+\]", response, re.DOTALL)
#         for match in matches:
#             parsed = eval(match.strip())
#             if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
#                 return list(dict.fromkeys(parsed))
#     except Exception as e:
#         print("❌ Failed to extract list:", e)
#     return None

# # ---------- Attribute Inference ----------
# def infer_attribute_list(prompt, model, tokenizer, device):
#     query = (
#         "You're a vision-language expert. Given a generation prompt, return a Python list of 10 coarse, high-level, "
#         "visually grounded attributes that can divide images into semantically distinct groups.\n\n"
#         f"Prompt: {prompt}\n\n"
#         "Output ONLY a Python list (no explanation):"
#     )
#     return extract_list_from_response(run_llm(query, model, tokenizer, device))

# # ---------- Top-k Attribute Selection ----------
# def pick_top_attributes(attr_list, prompt, model, tokenizer, device):
#     attr_str = ", ".join(attr_list)
#     query = (
#         f"You are a semantic reasoning expert.\n"
#         f"From the following attributes [{attr_str}], select 3 most informative and diverse ones "
#         f"for categorizing images generated with the prompt: \"{prompt}\".\n"
#         f"Return only a Python list of 3 lowercase strings from the above list. Do not explain anything."
#     )
#     return extract_list_from_response(run_llm(query, model, tokenizer, device))

# # ---------- BLIP-2 Image Query ----------
# def query_image_attr(image, question, processor, model, device):
#     inputs = processor(images=image, text=f"Question: {question} Answer:", return_tensors="pt").to(device)
#     output = model.generate(**inputs, max_new_tokens=20, pad_token_id=processor.tokenizer.eos_token_id)
#     return processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()

# # ---------- Main Attribute Pipeline ----------
# def extract_and_summarize_attributes(folder, prompt):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("🔧 Loading LLM...")
#     llm, tokenizer = load_llm()

#     print("🤖 Loading BLIP-2...")
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
#     blip_model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map={"": 0}
#     )

#     print("🧠 Inferring attribute candidates...")
#     attributes = infer_attribute_list(prompt, llm, tokenizer, device)
#     if not attributes:
#         raise ValueError("❌ Failed to infer attributes.")
#     print("📌 Inferred attributes:", attributes)

#     print("🎯 Selecting top 3 attributes...")
#     top3 = pick_top_attributes(attributes, prompt, llm, tokenizer, device)
#     if not top3:
#         raise ValueError("❌ Failed to select top 3 attributes.")
#     print("🏆 Top attributes:", top3)

#     files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))[:100]
#     results = {}
#     value_sets = {k: set() for k in top3}

#     for fname in tqdm(files, desc="🔍 Extracting"):
#         image = Image.open(os.path.join(folder, fname)).convert("RGB")
#         results[fname] = {}
#         for attr in top3:
#             q = f"What is the {attr} of this image? Use only 1-2 words."
#             ans = query_image_attr(image, q, processor, blip_model, device)
#             results[fname][attr] = ans
#             value_sets[attr].add(ans)

#     print("\n🧾 Final Summary:")
#     for attr in top3:
#         print(f"# {attr}:", sorted(value_sets[attr]))

#     return results, value_sets

# # ---------- Entry Point ----------
# if __name__ == "__main__":
#     folder = "pet_data/images"
#     prompt = "Photo of a pet"
#     extract_and_summarize_attributes(folder, prompt)

import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# ---------- Load LLM ----------
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

# ---------- Run LLM ----------
def run_llm(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------- Extract List ----------
def extract_list_from_response(response):
    print("🔍 Raw LLM response:\n", response)
    try:
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            parsed = eval(match.group(0))
            if isinstance(parsed, list):
                return list(dict.fromkeys(parsed))
    except Exception as e:
        print("❌ Failed to parse attribute list:", e)
    return None

# ---------- Select Top 3 Attributes (via numbered format) ----------
def pick_top_attributes(attr_list, prompt, model, tokenizer, device):
    attr_str = ", ".join(attr_list)
    query = (
        "You are a semantic reasoning expert.\n"
        f"Given the image generation prompt: \"{prompt}\"\n"
        f"and the following list of high-level attributes:\n[{attr_str}]\n\n"
        "Choose the 3 most informative and diverse attributes for meaningfully categorizing the generated images.\n"
        "Return your answer in the following format:\n"
        "1. <attribute>\n2. <attribute>\n3. <attribute>\n"
        "Do NOT include any explanation."
    )
    response = run_llm(query, model, tokenizer, device)
    print("🔍 Raw LLM response for top-3 selection:\n", response)

    try:
        lines = [line.strip() for line in response.splitlines() if re.match(r"^\d\.", line)]
        attrs = [re.sub(r"^\d\.\s*", "", line).strip().lower() for line in lines]
        return list(dict.fromkeys(attr for attr in attrs if attr in attr_list))
    except Exception as e:
        print("❌ Failed to extract top 3 list:", e)
        return None

# ---------- Main ----------
def main(prompt):
    print("🔧 Loading LLM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_llm()

    query = (
        "You're a vision-language expert. Given a generation prompt, return a Python list of 10 coarse, high-level, "
        "visually grounded attributes that can divide images into semantically distinct groups.\n\n"
        f"Prompt: {prompt}\n\n"
        "Output ONLY a Python list (no explanation):"
    )

    response = run_llm(query, model, tokenizer, device)
    attributes = extract_list_from_response(response)

    if attributes:
        print("📌 Inferred Attributes:", attributes)
    else:
        print("❌ No valid attribute list found.")
        return

    top3 = pick_top_attributes(attributes, prompt, model, tokenizer, device)
    if not top3:
        raise ValueError("❌ Failed to select top 3 attributes.")
    print("🏆 Top attributes:", top3)

# ---------- Entry ----------
if __name__ == "__main__":
    main(prompt="Photo of a pet")


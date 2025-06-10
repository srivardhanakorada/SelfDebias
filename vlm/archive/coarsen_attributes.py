import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------- Load Quantized Mistral on Single GPU ----------
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

# ---------- Query LLM with list to get coarsened mapping ----------
# import ast

# def query_coarsened_mapping(attr, values, model, tokenizer, device):
#     values_list = ", ".join(sorted(values))
#     prompt = (
#         f"You are an expert in organizing fine-grained concepts into a semantic hierarchy.\n"
#         f"Given the following attribute and its values, build a hierarchical tree of categories from broadest to most specific.\n"
#         f"Then, output only a Python dictionary mapping each fine-grained value to its top-level (broadest) category.\n"
#         f"Do not include any notes or explanations.\n\n"
#         f"If any value dones't fit into the heirarchy, keep it unchanged\n"
#         f"Attribute: {attr}\n"
#         f"Values: [{values_list}]\n\n"
#         f"Mapping:"
#     )
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Extract only the dictionary
#     try:
#         dict_start = response.index("{")
#         dict_end = response.index("}", dict_start) + 1
#         mapping_str = response[dict_start:dict_end]
#         return ast.literal_eval(mapping_str)
#     except Exception:
#         raise ValueError(f"⚠️ Failed to parse mapping from LLM output:\n{response}")

import ast

def query_coarsened_mapping(attr, values, model, tokenizer, device):
    values_list = ", ".join(sorted(values))
    prompt = (
        f"You are given an attribute name and a list of its fine-grained values.\n"
        f"Your task is to group these values into the broadest possible semantic categories.\n"
        f"Only use concise, domain-neutral labels appropriate for the attribute.\n"
        f"Avoid intermediate or overlapping labels. Be minimal.\n"
        f"Exclude any value that does not belong clearly under the given attribute.\n\n"
        f"Output a valid Python dictionary mapping each valid value to its top-level category.\n"
        f"Do NOT include any explanation or extra text.\n\n"
        f"Attribute: {attr}\n"
        f"Values: [{values_list}]\n\n"
        f"Mapping:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        dict_start = response.index("{")
        dict_end = response.index("}", dict_start) + 1
        mapping_str = response[dict_start:dict_end]
        return ast.literal_eval(mapping_str)
    except Exception:
        raise ValueError(f"⚠️ Failed to parse mapping from LLM output:\n{response}")


# ---------- Parse Raw Attributes ----------
def parse_raw_attributes(file_path):
    attr_values = {}
    key_pattern = re.compile(r"^\s*\d+\.\s*(.*?):\s*(.*)$")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.endswith(".png:"):
                continue
            match = key_pattern.match(line)
            if match:
                key, value = match.groups()
                key = key.strip().lower()
                value = value.strip().lower()
                attr_values.setdefault(key, set()).add(value)
    return attr_values

# ---------- Main ----------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    file_path = "raw_attributes.txt"

    print("📄 Reading raw attributes...")
    attr_values = parse_raw_attributes(file_path)

    print("🔧 Loading Mistral model...")
    model, tokenizer = load_local_llm()

    print("🔍 Coarsening each attribute using LLM...\n")
    all_mappings = {}
    for attr, values in attr_values.items():
        print(f"⚙️  Attribute: {attr} ({len(values)} values)")
        mapping = query_coarsened_mapping(attr, values, model, tokenizer, device)
        all_mappings[attr] = mapping
        print(f"✅ Mapping: {mapping}\n")

    with open("coarse_mappings.txt", "w") as f:
        for attr, mapping in all_mappings.items():
            f.write(f"# {attr}\n")
            for k, v in mapping.items():
                f.write(f"{k} -> {v}\n")
            f.write("\n")

    print("✅ All attribute values coarsened and saved to coarse_mappings.txt.")

if __name__ == "__main__":
    main()

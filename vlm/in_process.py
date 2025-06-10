import os
import json
from collections import Counter, defaultdict
from glob import glob
from tqdm import tqdm

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipForQuestionAnswering, BlipProcessor
from sentence_transformers import SentenceTransformer

# --- Device & Model Config --- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-vqa-capfilt-large"

ATTRIBUTES = {
  "person gender": [
    "male",
    "female"
  ],
  "person ethnicity": [
    "white",
    "black",
    "indian",
    "asian"
  ],
  "person facial expression": [
    "happy",
    "sad",
    "neutral",
    "angry"
  ]
}

# --- Global Caches for Models --- #
_blip_model = None
_blip_processor = None
_sbert_model = None

def _load_models():
    global _blip_model, _blip_processor, _sbert_model
    if _blip_model is None:
        print("🔧 Loading BLIP and SBERT models...")
        _blip_processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _blip_model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
        _sbert_model = SentenceTransformer("all-mpnet-base-v2").to(DEVICE)

def _closest_choice(answer, choices):
    texts = [answer] + choices
    embeddings = _sbert_model.encode(texts, convert_to_tensor=True)
    answer_emb = embeddings[0]
    choice_embs = embeddings[1:]
    sims = F.cosine_similarity(answer_emb.unsqueeze(0), choice_embs)
    best_idx = torch.argmax(sims).item()
    return choices[best_idx]

def extract_attributes(image_paths):
    _load_models()
    results = []

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            continue

        attr_result = {}
        for attr, choices in ATTRIBUTES.items():
            question = f"What is the {attr}?"
            inputs = _blip_processor(image, question, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                output = _blip_model.generate(**inputs, max_new_tokens=10)

            raw_answer = _blip_processor.tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
            best_match = _closest_choice(raw_answer, choices)
            attr_result[attr] = best_match
        results.append(attr_result)

    return results

# --- Cluster Directory Processing --- #
clustered_dir = "clustered_images"
versions = ["cond", "uncond"]
attribute_summary = defaultdict(lambda: defaultdict(dict))

print("🚀 Beginning attribute aggregation per cluster...")
for version in versions:
    version_dir = os.path.join(clustered_dir, version)
    if not os.path.isdir(version_dir):
        print(f"⚠️ Missing version directory: {version_dir}")
        continue

    timesteps = sorted(os.listdir(version_dir))
    for timestep in tqdm(timesteps,desc="Timesteps"):
        timestep_dir = os.path.join(version_dir, timestep)
        if not os.path.isdir(timestep_dir):
            continue

        clusters = sorted(os.listdir(timestep_dir))
        for cluster_id in clusters:
            cluster_path = os.path.join(timestep_dir, cluster_id)
            image_paths = sorted(glob(os.path.join(cluster_path, "*.png")))
            if not image_paths:
                continue

            try:
                attr_dicts = extract_attributes(image_paths)
            except Exception as e:
                print(f"⚠️ Attribute extraction failed for {cluster_path}: {e}")
                continue

            per_key_values = defaultdict(list)
            for attr_dict in attr_dicts:
                for k, v in attr_dict.items():
                    per_key_values[k].append(v)

            most_common = {k: Counter(v_list).most_common(1)[0][0] for k, v_list in per_key_values.items()}
            attribute_summary[version][timestep][cluster_id] = most_common

# --- Save Result --- #
with open("cluster_attributes.json", "w") as f:
    json.dump(attribute_summary, f, indent=2)

print("✅ Attribute summary saved to: cluster_attributes.json")

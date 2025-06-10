import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from transformers import BlipForQuestionAnswering, BlipProcessor
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-vqa-capfilt-large"
IMAGE_FOLDER = "/home/teja/three/vardhan/new_faces/data/images"
VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]
OUTPUT_JSON = os.path.join(IMAGE_FOLDER, "attributes_blip.json")

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
# ---------------- MODEL LOADING ---------------- #
def load_models():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    sbert = SentenceTransformer("all-mpnet-base-v2").to(DEVICE)
    return model, processor, sbert

# ---------------- POST-PROCESSING ---------------- #
def closest_choice(answer, choices, sbert_model):
    texts = [answer] + choices
    embeddings = sbert_model.encode(texts, convert_to_tensor=True)
    answer_emb = embeddings[0]
    choice_embs = embeddings[1:]
    sims = F.cosine_similarity(answer_emb.unsqueeze(0), choice_embs)
    best_idx = torch.argmax(sims).item()
    return choices[best_idx]

# ---------------- INFERENCE ---------------- #
def infer_attributes_for_image(image, model, processor, sbert, attributes):
    results = {}
    for attr, choices in attributes.items():
        question = f"What is the {attr.replace('_', ' ')}?"
        inputs = processor(image, question, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10)

        raw_answer = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        best_match = closest_choice(raw_answer, choices, sbert)
        results[attr] = best_match
    return results

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    model, processor, sbert = load_models()
    predictions_all = {}
    attribute_counts = defaultdict(lambda: defaultdict(int))

    for fname in tqdm(sorted(os.listdir(IMAGE_FOLDER))[:100]):
        if not any(fname.lower().endswith(ext) for ext in VALID_EXTENSIONS):
            continue
        path = os.path.join(IMAGE_FOLDER, fname)
        image = Image.open(path).convert("RGB")
        # print(f"🔍 Inferring attributes for {fname}...")

        attributes = infer_attributes_for_image(image, model, processor, sbert, ATTRIBUTES)
        predictions_all[fname] = attributes

        for k, v in attributes.items():
            attribute_counts[k][v] += 1

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions_all, f, indent=2)
    print(f"\n✅ Saved results to: {OUTPUT_JSON}")

    # Print attribute frequencies
    print("\n📊 Attribute Frequencies:")
    for attr, value_counts in attribute_counts.items():
        print(f"\n{attr}:")
        for val, count in sorted(value_counts.items(), key=lambda x: -x[1]):
            print(f"  {val}: {count}")

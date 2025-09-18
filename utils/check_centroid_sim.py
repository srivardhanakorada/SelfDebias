import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pandas as pd

# --- CONFIG ---
centroid_path = "centroids/centroids_pet.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
version = "cond"  # or "uncond"
num_timesteps = 51
text_prompts = ["a photo of a dog", "a photo of a cat"]

# --- Load centroids ---
print(f"Loading centroids from {centroid_path}")
centroids = torch.load(centroid_path, map_location=device)

# --- Load CLIP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Encode text prompts ---
with torch.no_grad():
    inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**inputs)
    text_features = F.normalize(text_features, dim=-1)  # [2, 512]

# --- Compute similarity for each centroid ---
results = []
print("Computing similarity of each centroid to dog/cat...")
for t in tqdm(range(num_timesteps)):
    timestep_centroids = centroids[version][t].to(device)  # [k, 512]
    timestep_centroids = F.normalize(timestep_centroids, dim=-1)  # Normalize

    similarity = timestep_centroids @ text_features.T  # [k, 2]
    for c in range(similarity.shape[0]):
        results.append({
            "timestep": t,
            "centroid": c,
            "similarity_dog": similarity[c, 0].item(),
            "similarity_cat": similarity[c, 1].item()
        })

# --- Save results ---
df = pd.DataFrame(results)
out_path = "centroid_clip_similarity.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved similarity table to: {out_path}")
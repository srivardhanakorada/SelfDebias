import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Paths
image_embeds_folder = "pet_data/clip"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Text embeddings
text_inputs = processor(text=["a dog", "a cat"], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # normalize

# Load image embeddings
def load_embedding(path):
    if path.endswith(".pt"):
        return torch.load(path, map_location=device)
    elif path.endswith(".npy"):
        return torch.tensor(np.load(path), device=device)
    else:
        return None

outliers = []
dog_matches = []
cat_matches = []

threshold = 0.23  # adjust based on what you consider a match
print("Processing image embeddings...")
for fname in tqdm(os.listdir(image_embeds_folder)):
    fpath = os.path.join(image_embeds_folder, fname)
    img_embed = load_embedding(fpath)
    if img_embed is None:
        continue
    if len(img_embed.shape) > 1:
        img_embed = img_embed.mean(dim=0)  # flatten temporal or patch-wise embeddings
    img_embed = img_embed / img_embed.norm()

    # Cosine similarity to "a dog" and "a cat"
    img_embed = img_embed.float()
    sims = torch.matmul(text_embeds, img_embed)
    dog_sim, cat_sim = sims.tolist()

    if dog_sim > threshold and dog_sim > cat_sim:
        dog_matches.append((fname, dog_sim))
    elif cat_sim > threshold and cat_sim > dog_sim:
        cat_matches.append((fname, cat_sim))
    else:
        outliers.append((fname, dog_sim, cat_sim))

# Results
print(f"\nDog matches: {len(dog_matches)}")
print(f"Cat matches: {len(cat_matches)}")
print(f"Outliers (neither dog nor cat): {len(outliers)}")

# Optionally save results
# with open("outliers.txt", "w") as f:
#     for fname, d, c in outliers:
#         f.write(f"{fname}: dog_sim={d:.3f}, cat_sim={c:.3f}\n")

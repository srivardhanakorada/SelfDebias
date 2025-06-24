import os
from PIL import Image
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import umap
from tqdm import tqdm

# --- Configuration ---
image_folder = 'face_outputs/original'  # CHANGE THIS
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load CLIP Model ---
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Load and Encode Images ---
image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_embeddings = []
valid_paths = []

with torch.no_grad():
    for path in tqdm(image_paths):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            embedding = model.encode_image(image)
            image_embeddings.append(embedding.squeeze(0).cpu().numpy())
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

image_embeddings = np.stack(image_embeddings)

# --- Spectral Clustering ---
clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
cluster_labels = clustering.fit_predict(image_embeddings)

# --- UMAP Reduction to 2D ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(image_embeddings)

# --- Plotting with Clustering Colors ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.8)
plt.axis('off')
plt.grid(False)
plt.savefig("umap_clustered.png")
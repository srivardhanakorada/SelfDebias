# import os
# import torch
# import umap
# import numpy as np
# import matplotlib.pyplot as plt

# # --- CONFIG ---
# folder_path = "data/clip"  # Replace with your actual folder

# # --- LOAD VECTORS FROM .pt FILES ---
# clip_vectors = []

# file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])
# for filename in file_list:
#     vec = torch.load(os.path.join(folder_path, filename))
#     vec = vec.detach().cpu().numpy()
#     vec = vec.reshape(-1)  # Ensure 1D vector
#     clip_vectors.append(vec)

# clip_vectors = np.stack(clip_vectors)

# # --- UMAP DIMENSIONALITY REDUCTION ---
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
# embedding = reducer.fit_transform(clip_vectors)

# # --- PLOT ---
# plt.figure(figsize=(10, 7))
# plt.scatter(embedding[:, 0], embedding[:, 1], s=20)
# plt.title("UMAP Projection of CLIP Vectors (.pt files)")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.savefig("clip_vecs.png")
import os
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import random

# --- CONFIG ---
folder_path = "pet_data/clip"  # Replace with your actual folder
num_clusters = 2           # Number of clusters for spectral clustering

# --- LOAD VECTORS FROM .pt FILES ---
clip_vectors = []
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])

for filename in file_list:
    vec = torch.load(os.path.join(folder_path, filename))
    vec = vec.detach().cpu().numpy()
    vec = vec.reshape(-1)  # Ensure 1D vector
    clip_vectors.append(vec)

clip_vectors = np.stack(clip_vectors)

# --- UMAP DIMENSIONALITY REDUCTION ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
embedding = reducer.fit_transform(clip_vectors)

# --- SPECTRAL CLUSTERING ---
spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
cluster_labels = spectral.fit_predict(clip_vectors)

# --- PLOT ---
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', s=20)
plt.title(f"UMAP + Spectral Clustering (k={num_clusters})")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig("clip_vecs_clustered.png")

# --- PRINT 5 FILES PER CLUSTER ---
cluster_to_files = defaultdict(list)
for i, label in enumerate(cluster_labels):
    cluster_to_files[label].append(file_list[i])

import matplotlib.image as mpimg

# --- DISPLAY IMAGES IN GRID BY CLUSTER ---
images_folder = "pet_data/images"
max_per_cluster = 4  # Number of images to show per cluster

cluster_ids = sorted(cluster_to_files)
num_clusters = len(cluster_ids)
plt.figure(figsize=(max_per_cluster * 3, num_clusters * 3))

for row, cluster_id in enumerate(cluster_ids):
    files = cluster_to_files[cluster_id]
    samples = random.sample(files, min(max_per_cluster, len(files)))
    for col, fname in enumerate(samples):
        img_name = os.path.splitext(fname)[0] + ".png"
        img_path = os.path.join(images_folder, img_name)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax = plt.subplot(num_clusters, max_per_cluster, row * max_per_cluster + col + 1)
            ax.imshow(img)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f"Cluster {cluster_id}", fontsize=12)
        else:
            print(f"Image not found: {img_path}")

plt.tight_layout()
plt.savefig("clustered_images_grid.png")
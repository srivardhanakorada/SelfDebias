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
import matplotlib.image as mpimg
from matplotlib import cm

# --- CONFIG ---
folder_path = "ball_data/clip"         # Path to CLIP vector .pt files
images_folder = "ball_data/images"     # Path to image .png files
num_clusters = 5                       # Number of spectral clusters
max_per_cluster = 4                    # Images to display per cluster

# --- LOAD VECTORS FROM .pt FILES ---
clip_vectors = []
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])

for filename in file_list:
    vec = torch.load(os.path.join(folder_path, filename))
    vec = vec.detach().cpu().numpy().reshape(-1)
    clip_vectors.append(vec)

clip_vectors = np.stack(clip_vectors)

# --- UMAP DIMENSIONALITY REDUCTION ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
embedding = reducer.fit_transform(clip_vectors)

# --- SPECTRAL CLUSTERING ---
spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
cluster_labels = spectral.fit_predict(clip_vectors)

# --- PLOT UMAP WITH CLUSTER LEGEND ---
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', s=20)

# Add legend
colormap = cm.get_cmap('tab10', num_clusters)
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colormap(i), markersize=10, label=f"Cluster {i}")
           for i in range(num_clusters)]
plt.legend(handles=handles, title="Clusters")

plt.title(f"UMAP + Spectral Clustering (k={num_clusters})")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("clip_vecs_clustered.png")
plt.close()

# --- MAP FILES TO CLUSTERS ---
cluster_to_files = defaultdict(list)
for i, label in enumerate(cluster_labels):
    cluster_to_files[label].append(file_list[i])

# --- DISPLAY IMAGES IN GRID BY CLUSTER ---
plt.figure(figsize=(max_per_cluster * 3, num_clusters * 3))
colormap = cm.get_cmap('tab10', num_clusters)

for row, cluster_id in enumerate(sorted(cluster_to_files)):
    files = cluster_to_files[cluster_id]
    samples = random.sample(files, min(max_per_cluster, len(files)))
    for col, fname in enumerate(samples):
        img_name = os.path.splitext(fname)[0] + ".png"
        img_path = os.path.join(images_folder, img_name)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            index = row * max_per_cluster + col + 1
            ax = plt.subplot(num_clusters, max_per_cluster, index)
            ax.imshow(img)
            ax.axis('off')

            # Add cluster label above the first image in each row
            if col == 0:
                ax.text(-0.1, 0.5, f"Cluster {cluster_id}",
                        fontsize=14, weight='bold',
                        color=colormap(cluster_id),
                        rotation=90,
                        va='center', ha='center',
                        transform=ax.transAxes)
        else:
            print(f"Image not found: {img_path}")

plt.tight_layout()
plt.savefig("clustered_images_grid.png")
plt.close()


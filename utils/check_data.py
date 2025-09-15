import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter

# --- Config ---
clip_folder = "/home/teja/three/vardhan/rebuttal/lands/clip"       # Folder with .pt CLIP vectors
image_folder = "/home/teja/three/vardhan/rebuttal/lands/images"               # Corresponding .png images
samples_per_cluster = 4
n_clusters = 2                        # You can change this
umap_outfile = "land_umap.jpg"
grid_outfile = "land_img.jpg"

# --- Load filtered embeddings ---
def load_clip_embeddings(folder, img_folder):
    embeddings = []
    valid_filenames = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])
    for fname in filenames:
        image_name = os.path.splitext(fname)[0] + '.png'
        image_path = os.path.join(img_folder, image_name)
        if not os.path.exists(image_path):
            continue
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        if np.all(img_np == 0):
            continue
        vec = torch.load(os.path.join(folder, fname))
        embeddings.append(vec.cpu().numpy())
        valid_filenames.append(fname)
    return np.stack(embeddings), valid_filenames

# --- Remap labels so smallest cluster becomes 0, largest becomes K-1 ---
def remap_cluster_labels_by_size(labels):
    label_counts = Counter(labels)
    sorted_labels = sorted(label_counts, key=lambda x: label_counts[x])
    label_map = {old: new for new, old in enumerate(sorted_labels)}
    new_labels = np.array([label_map[l] for l in labels])
    return new_labels

# --- UMAP scatter plot ---
def plot_umap_clusters(embeddings_2d, labels, title="UMAP with Clusters", outname="texture_clustered.jpg"):
    plt.figure(figsize=(8, 6))
    num_clusters = len(np.unique(labels))
    for i in range(num_clusters):
        idx = labels == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=12, alpha=0.7, label=f"Cluster {i}")
    plt.legend()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

# --- Image grid per cluster ---
def plot_sample_images_per_cluster(filenames, cluster_labels, image_folder, samples_per_cluster=4, outname='texture_grid.jpg'):
    k = np.max(cluster_labels) + 1

    # Group filenames per cluster
    cluster_to_files = {i: [] for i in range(k)}
    for i, label in enumerate(cluster_labels):
        cluster_to_files[label].append(filenames[i])

    # Identify and remove the smallest cluster
    cluster_sizes = {c: len(files) for c, files in cluster_to_files.items()}
    smallest_cluster = min(cluster_sizes, key=cluster_sizes.get)
    print(f"🚫 Removing smallest cluster: {smallest_cluster} (size={cluster_sizes[smallest_cluster]})")
    del cluster_to_files[smallest_cluster]

    # Sort remaining clusters
    remaining_clusters = sorted(cluster_to_files.keys())
    n_clusters = len(remaining_clusters)

    fig, axes = plt.subplots(samples_per_cluster, n_clusters, figsize=(n_clusters * 3, samples_per_cluster * 3))

    for col_idx, cluster_idx in enumerate(remaining_clusters):
        selected = cluster_to_files[cluster_idx][:samples_per_cluster]

        for row_idx in range(samples_per_cluster):
            ax = axes[row_idx, col_idx] if samples_per_cluster > 1 else axes[col_idx]

            if row_idx < len(selected):
                fname = selected[row_idx]
                img_path = os.path.join(image_folder, fname.replace('.pt', '.png'))
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    ax.axis("off")

                    # Label cluster at the top row
                    if row_idx == 0:
                        ax.set_title(f"Cluster {cluster_idx}", fontsize=14)
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")
                    ax.axis("off")
            else:
                ax.axis("off")  # empty slot

    plt.tight_layout()
    plt.savefig(outname, bbox_inches='tight')
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Load
    print("🔄 Loading CLIP embeddings...")
    embeddings, filenames = load_clip_embeddings(clip_folder, image_folder)
    print(f"✅ Loaded {len(filenames)} valid embeddings.")

    # Cluster
    print(f"🔍 Running KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    raw_labels = kmeans.fit_predict(embeddings)

    # Reorder labels by size
    labels = remap_cluster_labels_by_size(raw_labels)

    # UMAP
    print("📉 Running UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embeddings_2d = reducer.fit_transform(embeddings)

    print(f"📊 Saving UMAP plot to: {umap_outfile}")
    plot_umap_clusters(embeddings_2d, labels, outname=umap_outfile)

    # Image grid
    print(f"🖼️ Saving image grid to: {grid_outfile}")
    plot_sample_images_per_cluster(filenames, labels, image_folder, samples_per_cluster, outname=grid_outfile)

    print("✅ Done.")

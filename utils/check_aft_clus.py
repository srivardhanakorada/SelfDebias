import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.cluster import SpectralClustering
import umap
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

# -------------------
# Global Font Scaling
# -------------------
plt.rcParams.update({
    "font.size": 18,              # base font size
    "axes.titlesize": 22,         # subplot title size
    "axes.labelsize": 18,         # x/y label size
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "figure.titlesize": 24        # overall title
})

# -------------------
# Load CLIP
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -------------------
# Image Preprocessing
# -------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

# -------------------
# Utilities
# -------------------
def load_images_from_folder(folder):
    image_paths = sorted([
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append((img, path))
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
    return images

def extract_clip_embeddings(images):
    embeddings = []
    for img, _ in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        embeddings.append(image_features.cpu().numpy().squeeze())
    return np.array(embeddings)

def beautified_umap_plot(ax, embeddings, labels, title):
    reducer = umap.UMAP(n_components=2, random_state=42)
    proj_2d = reducer.fit_transform(embeddings)

    palette = sns.color_palette("Set1", n_colors=len(np.unique(labels)))
    for label in np.unique(labels):
        idxs = labels == label
        ax.scatter(
            proj_2d[idxs, 0],
            proj_2d[idxs, 1],
            s=60,
            alpha=0.85,
            label=f"Cluster {label}",
            color=palette[label],
            edgecolors='k',
            linewidths=0.6
        )
        # centroid label
        x, y = proj_2d[idxs].mean(axis=0)
        txt = ax.text(x, y, f"C{label}", fontsize=16, weight='bold', ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=True, loc='best')
    sns.despine(ax=ax)
    ax.grid(False)

# -------------------
# Main Entry
# -------------------
def main(before_folder, after_folder, output_path="comparison_umap.png"):
    print(f"Loading BEFORE images from: {before_folder}")
    images_before = load_images_from_folder(before_folder)
    print(f"Found {len(images_before)} images.")

    print(f"Loading AFTER images from: {after_folder}")
    images_after = load_images_from_folder(after_folder)
    print(f"Found {len(images_after)} images.")

    print("Extracting CLIP embeddings...")
    embeddings_before = extract_clip_embeddings(images_before)
    embeddings_after = extract_clip_embeddings(images_after)

    print("Clustering with Spectral Clustering (k=2)...")
    clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                    assign_labels='kmeans', random_state=42)
    labels_before = clustering.fit_predict(embeddings_before)
    labels_after = clustering.fit_predict(embeddings_after)

    print("Generating UMAP plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    beautified_umap_plot(axes[0], embeddings_before, labels_before, title="Before Debiasing")
    beautified_umap_plot(axes[1], embeddings_after, labels_after, title="After Debiasing")

    plt.suptitle("CLIP + UMAP + Spectral Clustering", fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved plot to: {output_path}")

# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=str, required=True, help="Path to folder of images BEFORE debiasing")
    parser.add_argument("--after", type=str, required=True, help="Path to folder of images AFTER debiasing")
    parser.add_argument("--output", type=str, default="comparison_umap.png", help="Output path for the UMAP plot")
    args = parser.parse_args()

    main(args.before, args.after, args.output)
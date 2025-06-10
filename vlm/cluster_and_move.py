import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
from umap import UMAP
import torch.nn as nn
import torch.nn.functional as F

# --- Config ---
data_dir = "/home/teja/three/vardhan/new_faces/data/contrastive_triplets"
clip_dir = "/home/teja/three/vardhan/new_faces/data/clip"  # not used
model_path = "pretrained/our_face.pt"
save_json_path = "cluster_to_filenames.json"
save_pt_path = "cluster_centroids.pt"
umap_dir = "umap_vis_faces"
os.makedirs(umap_dir, exist_ok=True)

device = "cuda"
num_timesteps = 51
UMAP_TIMESTEPS = [1, 2, 5, 10, 25, 50]
MAX_DEPTH = 3
MIN_CLUSTER_SIZE = 200

# --- Model ---
class HToCLIPJointContrast(nn.Module):
    def __init__(self, h_dim=1280 * 8 * 8, t_dim=128, proj_dim=512, hidden_dim=2048, num_timesteps=51):
        super().__init__()
        self.t_embed = nn.Embedding(num_timesteps, t_dim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim + t_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, h, t):
        h = h.view(h.size(0), -1)
        t_emb = self.t_embed(t)
        x = torch.cat([h, t_emb], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)

# --- Load model ---
model = HToCLIPJointContrast().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Project h[t] vectors ---
per_timestep_clip = {"cond": defaultdict(list), "uncond": defaultdict(list)}
all_files = sorted(glob(os.path.join(data_dir, "*.pt")))

for path in tqdm(all_files, desc="Projecting h[t]"):
    sample_dict = torch.load(path)
    for version in ["cond", "uncond"]:
        for t in range(num_timesteps):
            h = sample_dict[version][t]["h"].unsqueeze(0).to(device).float()
            timestep = torch.tensor([t], device=device)
            with torch.no_grad():
                z = model(h, timestep).squeeze(0).cpu()
            per_timestep_clip[version][t].append(z)

# --- Clustering helpers ---
def estimate_best_k(vectors, min_k=2, max_k=8):
    best_k, best_score = min_k, -1
    for k in range(min_k, min(max_k + 1, len(vectors))):
        try:
            model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5)
            labels = model.fit_predict(vectors)
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_k, best_score = k, score
        except:
            continue
    return best_k

def recursive_cluster_adaptive(vectors: np.ndarray, depth=0):
    n = len(vectors)
    if n < MIN_CLUSTER_SIZE or depth >= MAX_DEPTH:
        return np.zeros(n, dtype=int)
    try:
        model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=5)
        pred = model.fit_predict(vectors)
        final_labels = np.zeros(n, dtype=int)
        offset = 0
        for i in [0, 1]:
            idx = np.where(pred == i)[0]
            subvecs = vectors[idx]
            sublabels = recursive_cluster_adaptive(subvecs, depth + 1)
            for j in range(len(idx)):
                final_labels[idx[j]] = offset + sublabels[j]
            offset += len(np.unique(sublabels))
        return final_labels
    except:
        return np.zeros(n, dtype=int)

def hybrid_cluster(vectors: np.ndarray, timestep: int, version: str):
    k = estimate_best_k(vectors)
    top_model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5)
    top_labels = top_model.fit_predict(vectors)
    global_label_map = np.zeros(len(vectors), dtype=int)
    offset = 0
    for i in range(k):
        idx = np.where(top_labels == i)[0]
        subvecs = vectors[idx]
        if len(subvecs) < MIN_CLUSTER_SIZE:
            global_label_map[idx] = offset
            offset += 1
        else:
            sublabels = recursive_cluster_adaptive(subvecs, depth=1)
            for j in range(len(idx)):
                global_label_map[idx[j]] = offset + sublabels[j]
            offset += len(np.unique(sublabels))
    return global_label_map

# --- Final clustering and save filenames + centroids ---
final_cluster_dict = {"cond": {}, "uncond": {}}

for version in ["cond", "uncond"]:
    for t in tqdm(range(num_timesteps), desc=f"Clustering ({version})"):
        vecs = torch.stack(per_timestep_clip[version][t]).numpy()
        cluster_labels = hybrid_cluster(vecs, t, version)

        cluster_data = {}
        cluster_to_filenames = defaultdict(list)

        for i, file_path in enumerate(all_files):
            fname = os.path.basename(file_path).replace(".pt", ".png")
            cluster_id = int(cluster_labels[i])
            cluster_to_filenames[cluster_id].append(fname)

        for cluster_id, filenames in cluster_to_filenames.items():
            indices = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
            cluster_vecs = vecs[indices]
            centroid = np.mean(cluster_vecs, axis=0)
            cluster_data[int(cluster_id)] = {
                "filenames": filenames[:5],
                "centroid": torch.tensor(centroid)
            }

        final_cluster_dict[version][t] = cluster_data

        # --- UMAP Plot ---
        if t in UMAP_TIMESTEPS:
            reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
            umap_out = reducer.fit_transform(vecs)
            plt.figure(figsize=(6, 5))
            for c in np.unique(cluster_labels):
                idx = cluster_labels == c
                plt.scatter(umap_out[idx, 0], umap_out[idx, 1], s=10, label=f"Cluster {c}")
            plt.title(f"UMAP Projection: {version}, t={t}")
            plt.axis("off")
            plt.legend(fontsize=8, markerscale=2)
            plt.tight_layout()
            plt.savefig(os.path.join(umap_dir, f"umap_{version}_t{t}.png"))
            plt.close()

# --- Save JSON (filenames only) ---
filename_only_view = {
    version: {
        t: {
            str(cid): data["filenames"]
            for cid, data in clusters.items()
        } for t, clusters in version_dict.items()
    } for version, version_dict in final_cluster_dict.items()
}
with open(save_json_path, "w") as f:
    json.dump(filename_only_view, f, indent=2)

# --- Save .pt (filenames + centroids) ---
torch.save(final_cluster_dict, save_pt_path)

print(f"✅ Filenames saved to {save_json_path}")
print(f"✅ Centroid info saved to {save_pt_path}")
print(f"✅ UMAP visualizations saved to {umap_dir}/")
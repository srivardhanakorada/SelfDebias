import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
from umap import UMAP
from collections import defaultdict
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

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

# --- Config ---
data_dir = "vhl_data/contrastive_triplets"
model_path = "pretrained/our_vhl.pt"
save_path = "centroids/centroids_vhl_spectral.pt"
vis_dir = "centroids/umap_plots_vhl_spectral"
device = "cuda:0"
num_timesteps = 51
k = 3
os.makedirs(vis_dir, exist_ok=True)

# --- Load model ---
model = HToCLIPJointContrast().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Collect projected vectors ---
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

# --- Run Spectral Clustering with temporal consistency ---
centroids = {"cond": {}, "uncond": {}}
prev_centroids = {"cond": None, "uncond": None}

for version in ["cond", "uncond"]:
    for t in tqdm(range(num_timesteps), desc=f"Clustering ({version})"):
        vecs = torch.stack(per_timestep_clip[version][t])  # [N, 512]
        vecs_np = vecs.numpy()

        # --- Spectral Clustering ---
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
        labels = spectral.fit_predict(vecs_np)

        # --- Compute centroids manually
        cluster_centers = torch.stack([vecs[labels == i].mean(dim=0) for i in range(k)])

        # --- Cluster ordering
        if t == 0:
            sizes = [(labels == i).sum() for i in range(k)]
            order = sorted(range(k), key=lambda i: -sizes[i])
        else:
            prev = F.normalize(prev_centroids[version], dim=-1)
            curr = F.normalize(cluster_centers, dim=-1)
            sim = torch.matmul(prev, curr.T)
            order = sim.argmax(dim=1).tolist()
            if order[0] == order[1]:
                order = [0, 1]

        reordered = cluster_centers[order]
        centroids[version][t] = reordered
        prev_centroids[version] = reordered

        # --- Compute and print dot product between centroids
        # c0 = F.normalize(reordered[0], dim=0)
        # c1 = F.normalize(reordered[1], dim=0)
        # dot = torch.dot(c0, c1).item()
        # print(f"[{version} | t={t:02d}] ⟨c0 · c1⟩ = {dot:.4f}")

        # --- UMAP Visualization
        # reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        # vecs_2d = reducer.fit_transform(vecs_np)
        # centers_2d = reducer.transform(reordered.numpy())

        # plt.figure(figsize=(6, 5))
        # plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=labels, cmap='tab10', s=4, alpha=0.7)
        # plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=60, marker='X', label='Centroids')
        # plt.title(f"Spectral + UMAP - {version} - t={t}")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(vis_dir, f"{version}_t{t:02d}.png"))
        # plt.close()

# --- Save ---
torch.save(centroids, save_path)
print(f"✅ Saved spectral-clustered centroids → {save_path}")
print(f"✅ Saved UMAP plots in → {vis_dir}")

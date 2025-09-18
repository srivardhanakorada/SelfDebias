import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# --- CONFIG ---
centroids_path = "centroids/centroids_pet.pt"
model_path = "pretrained/our_pet.pt"
triplet_dir = "pet_data/contrastive_triplets"
timesteps = [1, 5, 10, 25, 50]
version = "cond"
device = "cuda:0"
sample_size = 1000
# ----------------

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

# --- Load model and centroids ---
model = HToCLIPJointContrast().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
centroids_all = torch.load(centroids_path)

triplet_paths = sorted(glob(os.path.join(triplet_dir, "*.pt")))[:sample_size]

# --- Generate and visualize one plot per timestep ---
for timestep in timesteps:
    z_pred_list = []

    for path in tqdm(triplet_paths, desc=f"Collecting h[t] at t={timestep}"):
        data = torch.load(path)
        h = data[version][timestep]["h"].unsqueeze(0).to(device).float()
        t_tensor = torch.tensor([timestep], device=device)
        with torch.no_grad():
            z_pred = model(h, t_tensor).squeeze(0).cpu()
        z_pred_list.append(z_pred)

    z_pred = torch.stack(z_pred_list)                      # [N, 512]
    z_centroids = centroids_all[version][timestep]         # [C, 512]
    z_pred_norm = F.normalize(z_pred, dim=-1)
    z_centroids_norm = F.normalize(z_centroids, dim=-1)

    # --- Assign cluster indices based on max cosine similarity ---
    sim_matrix = torch.matmul(z_pred_norm, z_centroids_norm.T)  # [N, C]
    cluster_indices = sim_matrix.argmax(dim=1).numpy()          # [N]

    # --- UMAP: Project both predicted embeddings and centroids ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0)
    umap_output = reducer.fit_transform(np.vstack([z_pred.numpy(), z_centroids.numpy()]))
    pred_umap = umap_output[:-z_centroids.shape[0]]
    cent_umap = umap_output[-z_centroids.shape[0]:]

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    num_clusters = z_centroids.shape[0]
    cmap = plt.cm.get_cmap("tab10", num_clusters)

    for i in range(num_clusters):
        idx = cluster_indices == i
        plt.scatter(pred_umap[idx, 0], pred_umap[idx, 1], s=10, color=cmap(i), label=f"Cluster {i}")

    plt.scatter(cent_umap[:, 0], cent_umap[:, 1], s=80, c='black', marker='X', label="Centroids")

    plt.title(f"Predicted CLIP Embeddings Colored by Cluster (t={timestep})")
    plt.tight_layout()
    plt.savefig(f"centroids_recursive/umap_recursive_centroids_colored_t{timestep}.png")
    plt.close()

print("✅ Saved UMAP plots with cluster coloring for t in", timesteps)

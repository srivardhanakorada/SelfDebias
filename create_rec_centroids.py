import os, torch, numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# --- Config ---
data_dir = "data/contrastive_triplets"
model_path = "pretrained/our.pt"
save_path = "centroids_recursive/centroids.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
device = "cuda:0"
num_timesteps = 51
MIN_SIZE = 500
MAX_DEPTH = 5

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

# --- Collect projected vectors ---
per_timestep_clip = {
    "cond": defaultdict(list),
    "uncond": defaultdict(list)
}
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

# --- Recursive clustering returning centroids and labels ---
def recursive_cluster(vectors: np.ndarray, depth=0):
    if len(vectors) <= MIN_SIZE or depth >= MAX_DEPTH:
        mean = vectors.mean(axis=0)  # FIX: ensure shape [512]
        labels = np.zeros(len(vectors), dtype=int)
        return [mean], labels
    try:
        model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                   n_neighbors=5, assign_labels='discretize', random_state=0)
        pred = model.fit_predict(vectors)
        centroids, final_labels = [], np.zeros(len(vectors), dtype=int)
        offset = 0
        for i in [0, 1]:
            idx = np.where(pred == i)[0]
            if len(idx) < MIN_SIZE:
                cent = vectors[idx].mean(axis=0)  # FIX
                centroids.append(cent)
                final_labels[idx] = offset
                offset += 1
            else:
                subcents, sublabels = recursive_cluster(vectors[idx], depth + 1)
                centroids.extend(subcents)
                for j in range(len(idx)):
                    final_labels[idx[j]] = offset + sublabels[j]
                offset += len(subcents)
        return centroids, final_labels
    except Exception as e:
        print(f"❌ Clustering failed at depth {depth}: {e}")
        mean = vectors.mean(axis=0)  # FIX
        labels = np.zeros(len(vectors), dtype=int)
        return [mean], labels

# --- Run per-timestep recursive clustering with Hungarian temporal consistency ---
centroids = {"cond": {}, "uncond": {}}
prev_centroids = {"cond": None, "uncond": None}

for version in ["cond", "uncond"]:
    for t in tqdm(range(num_timesteps), desc=f"Clustering ({version})"):
        vecs = torch.stack(per_timestep_clip[version][t]).numpy()  # [N, 512]
        current_cents, cluster_labels = recursive_cluster(vecs)
        current_cents = np.stack(current_cents)  # [M, 512]
        current_centroids = torch.tensor(current_cents, dtype=torch.float32)

        if t == 0:
            # Sort by cluster size (descending)
            sizes = [(cluster_labels == i).sum() for i in range(len(current_centroids))]
            order = sorted(range(len(current_centroids)), key=lambda i: -sizes[i])
        else:
            prev = F.normalize(prev_centroids[version], dim=-1)       # [P, 512]
            curr = F.normalize(current_centroids, dim=-1)             # [M, 512]
            sim = torch.matmul(prev, curr.T).numpy()                  # [P, M]

            # --- Hungarian matching (maximize similarity = minimize -similarity) ---
            row_ind, col_ind = linear_sum_assignment(-sim)
            order = col_ind.tolist()

            # Append unmatched clusters (if more in current than previous)
            all_indices = set(range(curr.shape[0]))
            unmatched = [j for j in all_indices if j not in order]
            order += unmatched

        reordered = current_centroids[order]
        centroids[version][t] = reordered
        prev_centroids[version] = F.normalize(reordered, dim=-1)

# --- Save ---
torch.save(centroids, save_path)
print(f"✅ Saved temporally ordered recursive centroids with Hungarian matching → {save_path}")

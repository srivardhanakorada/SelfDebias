import os, torch, numpy as np, random
from glob import glob
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from umap import UMAP
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.cm import get_cmap

# --- Config ---
data_dir = "/home/teja/three/shrikrishna/contrastive_triplets"
clip_dir = "/home/teja/three/shrikrishna/clip"
model_path = "/home/teja/three/shrikrishna/hspace_to_clip/epoch_50.pt"
save_path = "/home/teja/three/shrikrishna/centroids/sweighted_centroids_celebA.pt"
umap_save_dir = "/home/teja/three/shrikrishna/centroids/umap_cond"
log_file_path = "/home/teja/three/shrikrishna/centroids/clustering_log.txt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
os.makedirs(umap_save_dir, exist_ok=True)
device = "cuda"
num_timesteps = 50
MAX_DEPTH = 1
MIN_CLUSTER_SIZE = 200
UMAP_TIMESTEPS = [1, 5, 10, 25, 50]
UMAP_SAMPLES_PER_T = 1000

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# --- Model ---
class HToCLIPJointContrast(nn.Module):
    def __init__(self, h_dim=512 * 8 * 8, t_dim=128, proj_dim=512, hidden_dim=2048, num_timesteps=50):
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
# per_timestep_clip = {"cond": defaultdict(list), "uncond": defaultdict(list)}
per_timestep_clip = defaultdict(list)
all_files = sorted(glob(os.path.join(data_dir, "*.pt")))

for path in tqdm(all_files, desc="Projecting h[t]"):
    sample_dict = torch.load(path)
    # for version in ["cond", "uncond"]:
    for t in range(num_timesteps):
        h = sample_dict[t]["h"].unsqueeze(0).to(device).float()
        timestep = torch.tensor([t], device=device)
        with torch.no_grad():
            z = model(h, timestep).squeeze(0).cpu()
        per_timestep_clip[t].append(z)

# --- Clustering ---
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

def recursive_cluster_adaptive(vectors: np.ndarray, root_size: int, depth=0):
    n = len(vectors)
    if n < MIN_CLUSTER_SIZE or depth >= MAX_DEPTH:
        return [(vectors.mean(axis=0), depth)], np.zeros(n, dtype=int)
    try:
        model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=5)
        pred = model.fit_predict(vectors)
        centroids_with_depth, final_labels = [], np.zeros(n, dtype=int)
        offset = 0
        for i in [0, 1]:
            idx = np.where(pred == i)[0]
            subvecs = vectors[idx]
            subcents, sublabels = recursive_cluster_adaptive(subvecs, root_size, depth + 1)
            centroids_with_depth.extend(subcents)
            for j in range(len(idx)):
                final_labels[idx[j]] = offset + sublabels[j]
            offset += len(subcents)
        return centroids_with_depth, final_labels
    except Exception as e:
        log(f"❌ Clustering failed at depth {depth}: {e}")
        return [(vectors.mean(axis=0), depth)], np.zeros(n, dtype=int)

def hybrid_cluster_with_weights(vectors: np.ndarray, timestep: int):
    k = estimate_best_k(vectors)
    top_model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5)
    top_labels = top_model.fit_predict(vectors)
    log(f"t={timestep:02d} | Top-level k={k}")

    all_centroids, all_depths = [], []
    global_label_map = np.zeros(len(vectors), dtype=int)
    offset = 0
    for i in range(k):
        idx = np.where(top_labels == i)[0]
        subvecs = vectors[idx]

        if len(subvecs) < MIN_CLUSTER_SIZE:
            log(f"  - Top Cluster {i}: size={len(subvecs)} < {MIN_CLUSTER_SIZE}, skipping recursion")
            cent = subvecs.mean(axis=0)
            all_centroids.append(cent)
            all_depths.append(0)
            global_label_map[idx] = offset
            offset += 1
        else:
            subcents, sublabels = recursive_cluster_adaptive(subvecs, root_size=len(subvecs), depth=1)
            log(f"  - Top Cluster {i}: {len(subcents)} subclusters")
            for cent, d in subcents:
                all_centroids.append(cent)
                all_depths.append(d)
            for j in range(len(idx)):
                global_label_map[idx[j]] = offset + sublabels[j]
            offset += len(subcents)

    raw_weights = np.array([1 / (d + 1) for d in all_depths])
    normalized_weights = raw_weights / raw_weights.sum()
    return np.stack(all_centroids), global_label_map, normalized_weights

centroids = {}
weights = {}

# for version in ["cond", "uncond"]:
for t in tqdm(range(num_timesteps), desc=f"Clustering all timesteps"):
    vecs = torch.stack(per_timestep_clip[t]).numpy()
    current_cents, cluster_labels, cluster_weights = hybrid_cluster_with_weights(vecs, t)

    sizes = [(cluster_labels == i).sum() for i in range(len(current_cents))]
    order = sorted(range(len(current_cents)), key=lambda i: -sizes[i])

    reordered = torch.tensor(current_cents[order], dtype=torch.float32)
    reordered_weights = torch.tensor(cluster_weights[order], dtype=torch.float32)

    centroids[t] = reordered
    weights[t] = reordered_weights

torch.save({"centroids": centroids, "weights": weights}, save_path)
log(f"\n✅ Saved temporally ordered centroids + weights → {save_path}")

# --- UMAP Visualization ---
for t in UMAP_TIMESTEPS:
    sampled_files = random.sample(all_files, min(len(all_files), UMAP_SAMPLES_PER_T))
    # for version in ["cond", "uncond"]:
    h_proj_list, clip_list, label_list = [], [], []
    centroids_t = centroids[t]
    centroids_t_norm = F.normalize(centroids_t, dim=-1)

    for path in sampled_files:
        fname = os.path.basename(path)
        sample = torch.load(path)
        clip_path = os.path.join(clip_dir, fname)
        if not os.path.exists(clip_path): continue
        clip_vec = torch.load(clip_path)
        clip_list.append(clip_vec.reshape(1, -1))

        h_tensor = sample[t]["h"].unsqueeze(0).to(device).float()
        t_tensor = torch.tensor([t], device=device)
        with torch.no_grad():
            h_proj = model(h_tensor, t_tensor).cpu().numpy()
        h_proj_list.append(h_proj)

        h_norm = F.normalize(torch.tensor(h_proj), dim=-1)
        sim = torch.matmul(h_norm, centroids_t_norm.T).squeeze(0)
        cluster_id = torch.argmax(sim).item()
        label_list.append(cluster_id)

    if len(h_proj_list) == 0: continue

    h_proj_arr = np.concatenate(h_proj_list, axis=0)
    clip_arr = np.concatenate(clip_list, axis=0)
    labels = np.array(label_list)

    sim_vals = np.sum(h_proj_arr * clip_arr, axis=1) / (
        np.linalg.norm(h_proj_arr, axis=1) * np.linalg.norm(clip_arr, axis=1)
    )
    avg_sim = sim_vals.mean()
    log(f"🧠 Cosine Similarity @ t={t:02d} → {avg_sim:.4f}")

    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    h_umap = reducer.fit_transform(h_proj_arr)
    clip_umap = reducer.fit_transform(clip_arr)
    cent_umap = reducer.transform(centroids_t.numpy())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cmap = get_cmap("tab20", centroids_t.shape[0])
    for i in range(centroids_t.shape[0]):
        idx = labels == i
        axs[0].scatter(h_umap[idx, 0], h_umap[idx, 1], s=10, color=cmap(i))
    axs[0].scatter(cent_umap[:, 0], cent_umap[:, 1], s=60, c='black', marker='X')
    axs[0].set_title("Model Projection (Cluster Colored)")
    axs[0].axis("off")

    axs[1].scatter(clip_umap[:, 0], clip_umap[:, 1], s=10, c="black")
    axs[1].set_title("CLIP Embedding")
    axs[1].axis("off")

    plt.suptitle(f"UMAP @ timestep {t} ")
    plt.tight_layout()
    plt.savefig(os.path.join(umap_save_dir, f"umap_cluster_colored_t{t:02d}.png"))
    plt.close()

log(f"\n✅ UMAPs saved with cluster coloring in: {umap_save_dir}")
with open(log_file_path, "w") as f:
    f.write("\n".join(log_lines))
print(f"📘 Log saved at: {log_file_path}")

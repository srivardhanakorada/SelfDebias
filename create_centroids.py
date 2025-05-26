import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
from glob import glob

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
data_dir = "/kaggle/input/merged-contrastive-triplets/contrastive_triplets"
model_path = "/kaggle/input/diffusion_pipeline/pytorch/default/23/clip_debiasing_gen_models/pretrained/epoch10.pt" #Update meeee
save_path = "/kaggle/temp/centroids/centroids.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
os.makedirs("/kaggle/temp/centroids", exist_ok=True)
# device = "cuda:0"
num_timesteps = 51
k = 2
# --------------

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

# --- Run clustering with consistent cluster ordering ---
centroids = {"cond": {}, "uncond": {}}
prev_centroids = {"cond": None, "uncond": None}

for version in ["cond", "uncond"]:
    for t in range(num_timesteps):
        vecs = torch.stack(per_timestep_clip[version][t])  # [N, 512]
        kmeans = KMeans(n_clusters=k, random_state=42).fit(vecs.numpy())
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        if t == 0:
            # Sort by cluster size (descending)
            sizes = [(kmeans.labels_ == i).sum() for i in range(k)]
            order = sorted(range(k), key=lambda i: -sizes[i])
        else:
            # Match clusters to previous timestep using cosine similarity
            prev = F.normalize(prev_centroids[version], dim=-1)  # [2, 512]
            curr = F.normalize(cluster_centers, dim=-1)          # [2, 512]
            sim = torch.matmul(prev, curr.T)                     # [2, 2]
            order = sim.argmax(dim=1).tolist()

            # Ensure unique matching
            if order[0] == order[1]:
                order = [0, 1]  # fallback

        reordered = cluster_centers[order]
        centroids[version][t] = reordered
        prev_centroids[version] = reordered

# --- Save ---
torch.save(centroids, save_path)
print(f"✅ Saved ordered centroids for cond/uncond at all timesteps → {save_path}")

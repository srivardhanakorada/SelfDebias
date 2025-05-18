import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
import numpy as np
from glob import glob
from tqdm import tqdm
import random

# === CONFIG ===
centroids_path = "centroids/centroids.pt"
triplet_dir = "data/contrastive_triplets"
model_path = "pretrained/our.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_timesteps = 51
sample_size_per_cluster = 10
versions = ["cond", "uncond"]
# ===============

# --- Load projection model ---
class HToCLIPJointContrast(torch.nn.Module):
    def __init__(self, h_dim=1280 * 8 * 8, t_dim=128, proj_dim=512, hidden_dim=2048, num_timesteps=51):
        super().__init__()
        self.t_embed = torch.nn.Embedding(num_timesteps, t_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(h_dim + t_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, h, t):
        h = h.view(h.size(0), -1)
        t_emb = self.t_embed(t)
        x = torch.cat([h, t_emb], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)

model = HToCLIPJointContrast().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Load CLIP text embeddings ---
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()
with torch.no_grad():
    text_tokens = clip.tokenize(["a photo of a man", "a photo of a woman"]).to(device)
    text_embed = clip_model.encode_text(text_tokens)
    text_embed = F.normalize(text_embed, dim=-1)

# --- Load centroids ---
centroids = torch.load(centroids_path)

# --- Load data paths ---
all_triplet_paths = sorted(glob(os.path.join(triplet_dir, "*.pt")))

# --- For each version (cond/uncond), cluster, and timestep ---
for version in versions:
    centroid_sims = {0: {"man": [], "woman": []}, 1: {"man": [], "woman": []}}
    avg_sims = {0: {"man": [], "woman": []}, 1: {"man": [], "woman": []}}

    for t in range(num_timesteps):
        # Get centroids and normalize
        c = F.normalize(centroids[version][t], dim=-1).to(device)  # [2, 512]

        with torch.no_grad():
            sim_centroids = F.cosine_similarity(c.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)  # [2, 2]

        for i in [0, 1]:  # cluster index
            centroid_sims[i]["man"].append(sim_centroids[i, 0].item())
            centroid_sims[i]["woman"].append(sim_centroids[i, 1].item())

        # --- Sample and average similarity of 10 random members assigned to each cluster ---
        sims = {0: [], 1: []}  # cluster_id: list of [2] sims (man, woman)
        picked = 0
        random.shuffle(all_triplet_paths)
        for path in all_triplet_paths:
            if picked >= sample_size_per_cluster * 2:
                break
            data = torch.load(path)
            entry = data[version][t]
            h = entry["h"].unsqueeze(0).to(device).float()
            t_tensor = torch.tensor([t], device=device)
            with torch.no_grad():
                z_pred = model(h, t_tensor)  # [1, 512]
                z_pred = F.normalize(z_pred, dim=-1)
                sim = F.cosine_similarity(z_pred, c, dim=-1).squeeze(0)  # [2]
                cluster_id = torch.argmax(sim).item()

                sim_to_text = F.cosine_similarity(z_pred, text_embed, dim=-1).squeeze(0).cpu()  # [2]
                sims[cluster_id].append(sim_to_text)

                picked += 1

        for i in [0, 1]:
            if sims[i]:
                avg = torch.stack(sims[i]).mean(dim=0)  # [2]
                avg_sims[i]["man"].append(avg[0].item())
                avg_sims[i]["woman"].append(avg[1].item())
            else:
                avg_sims[i]["man"].append(float("nan"))
                avg_sims[i]["woman"].append(float("nan"))

    # --- Plot for each cluster ---
    for i in [0, 1]:
        plt.figure(figsize=(8, 4))
        plt.plot(centroid_sims[i]["man"], label="Centroid ↔ Man", color="blue", linestyle="--")
        plt.plot(avg_sims[i]["man"], label="Avg Cluster Member ↔ Man", color="blue", linestyle="-")
        plt.plot(centroid_sims[i]["woman"], label="Centroid ↔ Woman", color="orange", linestyle="--")
        plt.plot(avg_sims[i]["woman"], label="Avg Cluster Member ↔ Woman", color="orange", linestyle="-")
        plt.title(f"{version.upper()} | Cluster {i}")
        plt.xlabel("Timestep")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"sim_graph_{version}_cluster{i}.png")
        plt.close()

print("Saved: sim_graph_cond_cluster*.png and sim_graph_uncond_cluster*.png")

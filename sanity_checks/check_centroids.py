import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import clip

# === CONFIG ===
centroids_path = "centroids/centroids.pt"
model_path = "pretrained/our.pt"
triplet_dir = "data/contrastive_triplets"
timestep = 25
version = "cond"
device = "cuda:0"
sample_size = 1000
# ===============

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

# --- Load projection model and centroids ---
model = HToCLIPJointContrast().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
centroids = torch.load(centroids_path)[version][timestep]  # [k, 512]

# --- Load CLIP for true label inference ---
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
with torch.no_grad():
    text_tokens = clip.tokenize(["man", "woman"]).to(device)
    text_features = clip_model.encode_text(text_tokens)  # [2, 512]
    text_features = F.normalize(text_features, dim=-1)

# --- Collect projected and true embeddings and infer labels ---
triplet_paths = sorted(glob(os.path.join(triplet_dir, "*.pt")))[:sample_size]
z_pred_list = []
z_actual_list = []
labels = []

for path in tqdm(triplet_paths, desc="Collecting embeddings and labels"):
    data = torch.load(path)
    h = data[version][timestep]["h"].unsqueeze(0).to(device).float()
    clip_base = data[version][timestep]["clip_base"].unsqueeze(0).to(device)
    t_tensor = torch.tensor([timestep], device=device)

    with torch.no_grad():
        z_pred = model(h, t_tensor)                # [1, 512]
        clip_base = F.normalize(clip_base, dim=-1) # [1, 512]
        sim = F.cosine_similarity(clip_base, text_features)  # [1, 2]
        label = torch.argmax(sim, dim=-1).item()   # 0 = man, 1 = woman

    z_pred_list.append(z_pred.squeeze(0).cpu())
    z_actual_list.append(clip_base.squeeze(0).cpu())
    labels.append(label)

z_pred = torch.stack(z_pred_list).numpy()
z_actual = torch.stack(z_actual_list).numpy()
labels = np.array(labels)

# --- UMAP: predicted embeddings + centroids ---
reducer = umap.UMAP()
umap_pred = reducer.fit_transform(np.vstack([z_pred, centroids.numpy()]))
pred_umap = umap_pred[:-centroids.shape[0]]
cent_umap = umap_pred[-centroids.shape[0]:]

plt.figure(figsize=(8, 6))
plt.scatter(pred_umap[:, 0], pred_umap[:, 1], s=10, label="Predicted CLIP")
plt.scatter(cent_umap[:, 0], cent_umap[:, 1], s=80, c='red', marker='X', label="Centroids")
plt.title("Predicted CLIP Embeddings (t=25) + Centroids")
plt.legend()
plt.tight_layout()
plt.savefig("umap_predicted_clip_t25.png")
plt.close()

# --- UMAP: true CLIP embeddings ---
reducer = umap.UMAP()
umap_actual = reducer.fit_transform(z_actual)

plt.figure(figsize=(8, 6))
plt.scatter(umap_actual[:, 0], umap_actual[:, 1], s=10, label="Actual CLIP")
plt.title("True CLIP Embeddings (t=25)")
plt.legend()
plt.tight_layout()
plt.savefig("umap_actual_clip_t25.png")
plt.close()

# --- UMAP: predicted embeddings colored by true label ---
reducer = umap.UMAP()
umap_by_label = reducer.fit_transform(z_pred)

plt.figure(figsize=(8, 6))
plt.scatter(umap_by_label[labels == 0][:, 0], umap_by_label[labels == 0][:, 1], s=10, label="Man")
plt.scatter(umap_by_label[labels == 1][:, 0], umap_by_label[labels == 1][:, 1], s=10, label="Woman")
plt.title("Projected CLIP Embeddings (t=25) Labeled by True Gender")
plt.legend()
plt.tight_layout()
plt.savefig("umap_predicted_clip_by_true_label_t25.png")
plt.close()

print("Saved: umap_predicted_clip_t25.png, umap_actual_clip_t25.png, umap_predicted_clip_by_true_label_t25.png")

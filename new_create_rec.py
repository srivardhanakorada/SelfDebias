import os, torch, numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# --- Config ---
data_dir     = "contrastive_triplets"
image_dir    = "images"
model_path   = "pretrained/face.pt"
grid_path    = "./rec_grid.png"
os.makedirs(os.path.dirname(grid_path), exist_ok=True)

device       = "cuda:0"
timestep     = 25
MIN_SIZE     = 250
MAX_DEPTH    = 5

# --- Helper: Stitch PIL Images ---
def hconcat(imgs):
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * len(imgs), h))
    for i, img in enumerate(imgs):
        grid.paste(img, (i * w, 0))
    return grid

def vconcat(rows):
    w, h = rows[0].size
    grid = Image.new("RGB", (w, h * len(rows)))
    for i, row in enumerate(rows):
        grid.paste(row, (0, i * h))
    return grid

# --- Model ---
class HToCLIPJointContrast(nn.Module):
    def __init__(self, h_dim=1280*8*8, t_dim=128, proj_dim=512, hidden_dim=2048, num_timesteps=51):
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

# --- Project h[t=25] for "cond" only ---
vectors = []
filenames = []
all_files = sorted(glob(os.path.join(data_dir, "*.pt")))

for path in tqdm(all_files, desc="Projecting t=25 cond"):
    fname = os.path.basename(path)
    sample = torch.load(path)
    h = sample["cond"][timestep]["h"].unsqueeze(0).to(device).float()
    t = torch.tensor([timestep], device=device)
    with torch.no_grad():
        z = model(h, t).squeeze(0).cpu().numpy()
    vectors.append(z)
    filenames.append(fname)

vectors = np.stack(vectors)  # [N, 512]

# --- Recursive clustering ---
def recursive_cluster(vecs, depth=0):
    if len(vecs) <= MIN_SIZE or depth >= MAX_DEPTH:
        return [vecs.mean(axis=0)], np.zeros(len(vecs), dtype=int)
    try:
        model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=5, random_state=0)
        labels = model.fit_predict(vecs)
        centroids, final_labels = [], np.zeros(len(vecs), dtype=int)
        offset = 0
        for c in [0, 1]:
            idx = np.where(labels == c)[0]
            if len(idx) < MIN_SIZE:
                centroids.append(vecs[idx].mean(axis=0))
                final_labels[idx] = offset
                offset += 1
            else:
                subc, subl = recursive_cluster(vecs[idx], depth + 1)
                centroids.extend(subc)
                for j, orig in enumerate(idx):
                    final_labels[orig] = offset + subl[j]
                offset += len(subc)
        return centroids, final_labels
    except Exception as e:
        print(f"Clustering failed at depth {depth}: {e}")
        return [vecs.mean(axis=0)], np.zeros(len(vecs), dtype=int)

# --- Run clustering ---
_, labels = recursive_cluster(vectors)

# --- Group filenames by cluster ---
cluster_to_imgs = defaultdict(list)
for fname, label in zip(filenames, labels):
    img_path = os.path.join(image_dir, fname.replace(".pt", ".png"))
    if os.path.exists(img_path):
        cluster_to_imgs[label].append(img_path)

# --- Build grid ---
rows = []
for label in sorted(cluster_to_imgs):
    paths = cluster_to_imgs[label][:5]
    if len(paths) < 5:
        continue
    imgs = [Image.open(p).resize((128, 128)) for p in paths]
    row = hconcat(imgs)
    rows.append(row)

if rows:
    grid = vconcat(rows)
    grid.save(grid_path)
    print(f"🖼️ Saved image grid at: {grid_path}")
else:
    print("⚠️ Not enough clusters with ≥ 5 images")
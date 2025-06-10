import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class ContrastiveTripletDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pt")])
        self.versions = ["cond", "uncond"]
        self.num_timesteps = 51
        self.samples = [(path, v, t) for path in self.paths for v in self.versions for t in range(self.num_timesteps)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, version, t = self.samples[idx]
        data = torch.load(path)
        entry = data[version][t]
        return {
            "h": entry["h"],
            "clip_base": entry["clip_base"],
            "clip1": entry["clip1"],
            "clip2": entry["clip2"],
            "t": torch.tensor(entry["t"])
        }

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

def nt_xent_loss(z_pred, z1, z2, temperature=0.1):
    z_pred = F.normalize(z_pred, dim=-1)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    pos_sim = F.cosine_similarity(z_pred, z1, dim=-1) + F.cosine_similarity(z_pred, z2, dim=-1)
    return -pos_sim.mean() / temperature

def plot_umap(preds, targets, epoch, out_dir="umap_plots"):
    os.makedirs(out_dir, exist_ok=True)
    sample_size = min(500, len(preds))
    idxs = np.random.choice(len(preds), sample_size, replace=False)
    pred_sub = np.array(preds)[idxs]
    target_sub = np.array(targets)[idxs]

    reducer = umap.UMAP()
    emb = reducer.fit_transform(np.vstack([pred_sub, target_sub]))
    labels = np.array(["Predicted"] * len(pred_sub) + ["True CLIP"] * len(target_sub))

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        subset = emb[labels == label]
        plt.scatter(subset[:, 0], subset[:, 1], label=label, alpha=0.6, s=10)
    plt.legend()
    plt.title(f"UMAP at Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"umap_epoch{epoch}.png"))
    plt.close()

def train_contrastive(model, dataloader, optimizer, epochs=10, device="cuda"):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        all_cosines = []
        preds_t25, targets_t25 = [], []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            h = batch["h"].to(device)
            t = batch["t"].to(device)
            clip1 = batch["clip1"].to(device)
            clip2 = batch["clip2"].to(device)
            clip_base = batch["clip_base"].to(device)

            optimizer.zero_grad()
            z_pred = model(h, t)
            loss = nt_xent_loss(z_pred, clip1, clip2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                z_pred = F.normalize(z_pred, dim=-1)
                clip_base = F.normalize(clip_base, dim=-1)
                cos = F.cosine_similarity(z_pred, clip_base, dim=-1)
                all_cosines.extend(cos.cpu().tolist())

                mask = (t == 25)
                if mask.any():
                    preds_t25.append(z_pred[mask].cpu().numpy())
                    targets_t25.append(clip_base[mask].cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        avg_cosine = np.mean(all_cosines)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f} | Cosine Sim: {avg_cosine:.4f}")

        if preds_t25:
            plot_umap(np.vstack(preds_t25), np.vstack(targets_t25), epoch)

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/face.pt")

# === CONFIGURATION ===
root_dir = "face_data/contrastive_triplets"
dataset = ContrastiveTripletDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

model = HToCLIPJointContrast().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_contrastive(model, dataloader, optimizer, epochs=10, device="cuda")
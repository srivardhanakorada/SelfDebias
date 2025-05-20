import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

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

def make_model(path: Union[str, os.PathLike], device: torch.device) -> HToCLIPJointContrast:
    model = HToCLIPJointContrast().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_dual_centroids(timestep: int, centroid_path: str):
    if not os.path.exists(centroid_path):
        raise FileNotFoundError(f"Centroid file not found: {centroid_path}")
    full_dict = torch.load(centroid_path)  # {"cond": {t: [K,512]}, "uncond": {t: [K,512]}}
    cond = F.normalize(full_dict["cond"][timestep], dim=-1)
    uncond = F.normalize(full_dict["uncond"][timestep], dim=-1)
    return cond, uncond

@torch.enable_grad()
def compute_distribution_gradients(
    sample: torch.Tensor,           # shape: [2B, 1280, 8, 8]
    timestep: int,
    checkpoint_path: str,
    centroid_path: str,
    loss_strength: float,
    temperature: float = 8.0,
):
    device = sample.device
    sample = sample.detach().requires_grad_()
    model = make_model(checkpoint_path, device)
    centroids_cond, centroids_uncond = load_dual_centroids(timestep, centroid_path)
    centroids_cond = centroids_cond.to(device)        # [C1, 512]
    centroids_uncond = centroids_uncond.to(device)    # [C2, 512]
    cond_h = sample[1::2]     # [B, ...]
    uncond_h = sample[0::2]   # [B, ...]
    t_tensor = torch.full((cond_h.size(0),), timestep, dtype=torch.long, device=device)
    z_cond = model(cond_h, t_tensor)      # [B, 512]
    z_uncond = model(uncond_h, t_tensor)  # [B, 512]
    sims_cond = F.cosine_similarity(z_cond.unsqueeze(1), centroids_cond.unsqueeze(0), dim=-1)      # [B, C1]
    sims_uncond = F.cosine_similarity(z_uncond.unsqueeze(1), centroids_uncond.unsqueeze(0), dim=-1)  # [B, C2]
    probs_cond = F.softmax(sims_cond / temperature, dim=-1)     # [B, C1]
    probs_uncond = F.softmax(sims_uncond / temperature, dim=-1) # [B, C2]
    uniform_cond = torch.full_like(probs_cond, 1.0 / probs_cond.size(1))
    uniform_uncond = torch.full_like(probs_uncond, 1.0 / probs_uncond.size(1))
    kl_cond = (probs_cond * (probs_cond / uniform_cond).log()).sum(dim=1).mean()
    kl_uncond = (probs_uncond * (probs_uncond / uniform_uncond).log()).sum(dim=1).mean()
    loss = loss_strength * (kl_cond + kl_uncond)
    grads = torch.autograd.grad(loss, sample)[0]
    return grads, probs_cond.detach().cpu(), probs_uncond.detach().cpu()

@torch.enable_grad()
def compute_sample_gradients(
    sample: torch.Tensor,                # [2B, 1280, 8, 8], interleaved format
    timestep: int,
    class_index: int,
    checkpoint_path: str,
    centroid_path: str,
    temperature: float = 8.0,
) -> torch.Tensor:
    device = sample.device
    sample = sample.detach().requires_grad_()

    model = make_model(checkpoint_path, device)
    centroids_cond, centroids_uncond = load_dual_centroids(timestep, centroid_path)
    centroids_cond = centroids_cond.to(device)
    centroids_uncond = centroids_uncond.to(device)

    cond_h = sample[1::2]
    uncond_h = sample[0::2]
    t_tensor = torch.full((cond_h.size(0),), timestep, dtype=torch.long, device=device)

    z_cond = model(cond_h, t_tensor)      # [B, 512]
    z_uncond = model(uncond_h, t_tensor)  # [B, 512]

    sims_cond = F.cosine_similarity(z_cond.unsqueeze(1), centroids_cond.unsqueeze(0), dim=-1)
    sims_uncond = F.cosine_similarity(z_uncond.unsqueeze(1), centroids_uncond.unsqueeze(0), dim=-1)

    probs_cond = F.softmax(sims_cond / temperature, dim=-1)
    probs_uncond = F.softmax(sims_uncond / temperature, dim=-1)

    loss_cond = -probs_cond[:, class_index].sum()
    loss_uncond = -probs_uncond[:, class_index].sum()
    loss = loss_cond + loss_uncond

    grads = torch.autograd.grad(loss, sample)[0]
    return grads

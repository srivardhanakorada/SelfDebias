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

    # Load centroids for conditional and unconditional samples
    centroids_cond, centroids_uncond = load_dual_centroids(timestep, centroid_path)
    centroids_cond = centroids_cond.to(device)        # [C1, 512]
    centroids_uncond = centroids_uncond.to(device)    # [C2, 512]

    # Split the batch into conditional and unconditional parts
    cond_h = sample[1::2]     # [B, ...]
    uncond_h = sample[0::2]   # [B, ...]

    t_tensor = torch.full((cond_h.size(0),), timestep, dtype=torch.long, device=device)

    # Project to semantic space
    z_cond = model(cond_h, t_tensor)      # [B, 512]
    z_uncond = model(uncond_h, t_tensor)  # [B, 512]

    # Compute cosine similarities to centroids
    sims_cond = F.cosine_similarity(z_cond.unsqueeze(1), centroids_cond.unsqueeze(0), dim=-1)      # [B, C1]
    sims_uncond = F.cosine_similarity(z_uncond.unsqueeze(1), centroids_uncond.unsqueeze(0), dim=-1)  # [B, C2]

    # Softmax to get probability distributions
    probs_cond = F.softmax(sims_cond / temperature, dim=-1)     # [B, C1]
    probs_uncond = F.softmax(sims_uncond / temperature, dim=-1) # [B, C2]

    # Compute empirical class distributions (batch-level)
    p_empirical_cond = probs_cond.mean(dim=0)  # [C1]
    p_empirical_uncond = probs_uncond.mean(dim=0)  # [C2]

    # Define uniform target distributions
    uniform_cond = torch.full_like(p_empirical_cond, 1.0 / p_empirical_cond.size(0))
    uniform_uncond = torch.full_like(p_empirical_uncond, 1.0 / p_empirical_uncond.size(0))

    # Compute KL divergence between batch distributions and uniform
    kl_cond = (p_empirical_cond * (p_empirical_cond / uniform_cond).log()).sum()
    kl_uncond = (p_empirical_uncond * (p_empirical_uncond / uniform_uncond).log()).sum()

    # Total loss and gradients
    loss = loss_strength * (kl_cond + kl_uncond)
    grads = torch.autograd.grad(loss, sample)[0]

    return grads

import math

@torch.no_grad()
def _clip_by_global_norm(t: torch.Tensor, max_norm: float):
    if max_norm is None or max_norm <= 0:
        return t
    n = t.norm(2)
    if n > max_norm:
        t.mul_(max_norm / (n + 1e-12))
    return t

def _forward_kl_and_probs(sample, timestep, model, centroids_cond, centroids_uncond,
                          temperature: float, loss_strength: float):
    # Build differentiable view
    sample = sample.detach().requires_grad_(True)

    # Split the batch into conditional and unconditional parts
    cond_h = sample[1::2]     # [B, ...]
    uncond_h = sample[0::2]   # [B, ...]

    t_tensor = torch.full((cond_h.size(0),), timestep, dtype=torch.long, device=sample.device)

    # Project to semantic space
    z_cond = model(cond_h, t_tensor)      # [B, 512]
    z_uncond = model(uncond_h, t_tensor)  # [B, 512]

    # Cosine sims -> softmax probs
    sims_cond = F.cosine_similarity(z_cond.unsqueeze(1), centroids_cond.unsqueeze(0), dim=-1)      # [B, C1]
    sims_uncond = F.cosine_similarity(z_uncond.unsqueeze(1), centroids_uncond.unsqueeze(0), dim=-1)  # [B, C2]
    probs_cond = F.softmax(sims_cond / temperature, dim=-1)     # [B, C1]
    probs_uncond = F.softmax(sims_uncond / temperature, dim=-1) # [B, C2]

    # Batch-mean empirical distributions
    p_empirical_cond = probs_cond.mean(dim=0)  # [C1]
    p_empirical_uncond = probs_uncond.mean(dim=0)  # [C2]

    # Uniform targets
    uniform_cond = torch.full_like(p_empirical_cond, 1.0 / p_empirical_cond.size(0))
    uniform_uncond = torch.full_like(p_empirical_uncond, 1.0 / p_empirical_uncond.size(0))

    # KL losses
    kl_cond = (p_empirical_cond * (p_empirical_cond / uniform_cond).log()).sum()
    kl_uncond = (p_empirical_uncond * (p_empirical_uncond / uniform_uncond).log()).sum()
    loss = loss_strength * (kl_cond + kl_uncond)

    # Gradients wrt sample
    grads = torch.autograd.grad(loss, sample, create_graph=False, retain_graph=False)[0]
    return loss, grads, sample

@torch.enable_grad()
def distribution_guidance_multi_step(
    sample: torch.Tensor,           # shape: [2B, 1280, 8, 8]
    timestep: int,
    checkpoint_path: str,
    centroid_path: str,
    loss_strength: float,
    temperature: float = 8.0,
    step_size: float = 1e4,
    num_inner_steps: int = 1,
    grad_clip_norm = None,
):
    device = sample.device

    # Load once, reuse across inner steps
    model = make_model(checkpoint_path, device)
    centroids_cond, centroids_uncond = load_dual_centroids(timestep, centroid_path)
    centroids_cond = F.normalize(centroids_cond.to(device), dim=-1)
    centroids_uncond = F.normalize(centroids_uncond.to(device), dim=-1)

    # Logs for ablation
    losses, grad_norms = [], []
    last_grads = None

    # Inner gradient steps (per denoising step)
    x = sample
    for _ in range(max(1, num_inner_steps)):
        loss, grads, x_ref = _forward_kl_and_probs(
            x, timestep, model, centroids_cond, centroids_uncond, temperature, loss_strength
        )
        # Optional global-norm clipping on the raw gradient tensor
        g = grads
        if grad_clip_norm is not None:
            g = g.clone()
            _clip_by_global_norm(g, grad_clip_norm)

        # Gradient descent step in h-space
        x = (x_ref - step_size * g).detach()

        # Bookkeeping
        last_grads = grads.detach()
        losses.append(loss.detach())
        grad_norms.append(grads.detach().norm().item())

    stats = {
        "losses": torch.stack(losses) if len(losses) > 0 else torch.tensor([]),
        "grad_norms": grad_norms,
    }
    return last_grads, x, stats

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

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
        h = h.reshape(h.size(0), -1)
        t_emb = self.t_embed(t)
        x = torch.cat([h, t_emb], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)

def make_model(path: Union[str, os.PathLike], device: torch.device) -> HToCLIPJointContrast:
    model = HToCLIPJointContrast().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_centroids(timestep: int, centroid_path: str):
    if not os.path.exists(centroid_path):
        raise FileNotFoundError(f"Centroid file not found: {centroid_path}")
    full_dict = torch.load(centroid_path)  # {"cond": {t: [K,512]}, "uncond": {t: [K,512]}} for SD, {'centroids':{t: [K,512]}} for DDIM
    centroids = F.normalize(full_dict["centroids"][timestep], dim=-1)
    return centroids


@torch.enable_grad()
def compute_distribution_gradients(
    sample: torch.Tensor,           # shape: [B, 512, 8, 8]
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
    centroids = load_centroids(timestep, centroid_path)
    centroids = centroids.to(device)        # [C1, 512]

    # Split the batch into conditional and unconditional parts for SD
    # cond_h = sample[1::2]     # [B, ...]
    # uncond_h = sample[0::2]   # [B, ...]

    t_tensor = torch.full((sample.size(0),), timestep, dtype=torch.long, device=device)

    # Project to semantic space
    z = model(sample, t_tensor)      # [B, 512]
    # z_uncond = model(uncond_h, t_tensor)  # [B, 512]

    # Compute cosine similarities to centroids
    sims = F.cosine_similarity(z.unsqueeze(1), centroids.unsqueeze(0), dim=-1)      # [B, C1]

    # Softmax to get probability distributions
    probs = F.softmax(sims / temperature, dim=-1)     # [B, C1]

    # Compute empirical class distributions (batch-level)
    p_empirical = probs.mean(dim=0)  # [C1]

    # Define uniform target distributions
    uniform = torch.full_like(p_empirical, 1.0 / p_empirical.size(0))

    # Compute KL divergence between batch distributions and uniform
    kl = (p_empirical * (p_empirical / uniform).log()).sum()

    # Total loss and gradients
    loss = loss_strength * (kl)
    grads = torch.autograd.grad(loss, sample)[0]

    return grads, p_empirical.detach().cpu()
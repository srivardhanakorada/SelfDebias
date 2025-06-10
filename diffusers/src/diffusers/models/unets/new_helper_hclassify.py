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

def load_attribute_reps(timestep: int, path: str):
    full_dict = torch.load(path)
    cond_dict = full_dict["cond"][timestep]
    uncond_dict = full_dict["uncond"][timestep]
    return cond_dict, uncond_dict

@torch.enable_grad()
def compute_distribution_gradients(
    sample: torch.Tensor,               # shape: [2B, 1280, 8, 8]
    timestep: int,
    checkpoint_path: str,
    attr_rep_path: str,
    loss_strength: float,
    temperature: float = 8.0
):
    device = sample.device
    sample = sample.detach().requires_grad_()
    model = make_model(checkpoint_path, device)

    # Load representations
    attr_reps_cond, attr_reps_uncond = load_attribute_reps(timestep, attr_rep_path)

    # Prepare conditional and unconditional halves
    cond_h = sample[1::2]     # [B, ...]
    uncond_h = sample[0::2]   # [B, ...]
    t_tensor = torch.full((cond_h.size(0),), timestep, dtype=torch.long, device=device)

    # Project to semantic space
    z_cond = model(cond_h, t_tensor)      # [B, 512]
    z_uncond = model(uncond_h, t_tensor)  # [B, 512]

    def compute_kl(z, attr_reps):
        kl_total = 0.0
        ans = [1,0,0]
        for i, (attr_key, val_dict) in enumerate(attr_reps.items()):
            values = list(val_dict.keys())
            vectors = torch.stack([val_dict[val].to(device) for val in values], dim=0)  # [C, 512]
            vectors = F.normalize(vectors, dim=-1)
            sims = F.cosine_similarity(z.unsqueeze(1), vectors.unsqueeze(0), dim=-1)  # [B, C]
            probs = F.softmax(sims / temperature, dim=-1)                             # [B, C]
            p_empirical = probs.mean(dim=0)                                           # [C]
            p_target = torch.full_like(p_empirical, 1.0 / p_empirical.size(0))        # uniform
            kl = ans[i]*((p_empirical * (p_empirical / p_target).log()).sum())
            kl_total += kl
        return kl_total

    kl_cond = compute_kl(z_cond, attr_reps_cond)
    kl_uncond = compute_kl(z_uncond, attr_reps_uncond)
    loss = loss_strength * (kl_cond + kl_uncond)
    grads = torch.autograd.grad(loss, sample)[0]

    return grads, kl_cond.detach().cpu(), kl_uncond.detach().cpu(), loss.detach().cpu()

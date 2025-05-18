# import os
# import torch

# ours_dir = "gradients/ours"
# theirs_dir = "gradients/theirs"

# ours_files = sorted([f for f in os.listdir(ours_dir) if f.endswith(".pt")])
# theirs_files = sorted([f for f in os.listdir(theirs_dir) if f.endswith(".pt")])

# assert len(ours_files) == len(theirs_files), "Mismatch in number of .pt files."

# for i, (f_ours, f_theirs) in enumerate(zip(ours_files, theirs_files)):
#     grads_ours = torch.load(os.path.join(ours_dir, f_ours))
#     grads_theirs = torch.load(os.path.join(theirs_dir, f_theirs))

#     print(f"\n=== File {i}: {f_ours} vs {f_theirs} ===")
#     print(f"→ Ours:    {len(grads_ours)} timesteps")
#     print(f"→ Theirs:  {len(grads_theirs)} timesteps")

#     assert len(grads_ours) == len(grads_theirs), "Mismatch in timestep count"

#     for t, (g1, g2) in enumerate(zip(grads_ours, grads_theirs)):
#         print(f"  Timestep {t:2d}: Ours = {tuple(g1.shape)}, Theirs = {tuple(g2.shape)}")
#         if g1.shape != g2.shape:
#             print("    ⚠️ Shape mismatch!")


# import os
# import torch
# import matplotlib.pyplot as plt

# def compute_avg_l2_per_timestep(grad_files_dir):
#     files = sorted([f for f in os.listdir(grad_files_dir) if f.endswith(".pt")])
#     all_grads = []

#     for file in files:
#         grad_list = torch.load(os.path.join(grad_files_dir, file))  # list of 51 tensors
#         grads_l2 = [
#             g[:4].view(4, -1).norm(dim=1).mean().item()  # Only first 4 (conditional)
#             for g in grad_list
#         ]
#         all_grads.append(grads_l2)
#     print(torch.tensor(all_grads).mean(dim=0))
#     return torch.tensor(all_grads).mean(dim=0)  # shape: [51]

# ours_l2 = compute_avg_l2_per_timestep("gradients/ours")
# theirs_l2 = compute_avg_l2_per_timestep("gradients/theirs")

# plt.figure(figsize=(10, 5))
# plt.plot(ours_l2.tolist(), label="Ours", linewidth=2)
# plt.plot(theirs_l2.tolist(), label="Parihar et al.", linewidth=2)
# plt.xlabel("Timestep")
# plt.ylabel("Avg. Grad L2 Norm (per image)")
# plt.title("Gradient Strength per Timestep (Conditional)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("grad_comp_cond.png")


import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

ours_dir = "gradients/ours"
theirs_dir = "gradients/theirs"

ours_files = sorted([f for f in os.listdir(ours_dir) if f.endswith(".pt")])
theirs_files = sorted([f for f in os.listdir(theirs_dir) if f.endswith(".pt")])

assert len(ours_files) == len(theirs_files), "Mismatch in number of files."

all_angles = []  # List of [timesteps] per file

for f_ours, f_theirs in zip(ours_files, theirs_files):
    g_ours = torch.load(os.path.join(ours_dir, f_ours))
    g_theirs = torch.load(os.path.join(theirs_dir, f_theirs))
    assert len(g_ours) == len(g_theirs) == 51

    per_timestep_angles = []
    for t in range(51):
        ours_flat = g_ours[t][4:].reshape(4, -1)
        theirs_flat = g_theirs[t][4:].reshape(4, -1)

        dot = (ours_flat * theirs_flat).sum(dim=1)
        norm_ours = ours_flat.norm(dim=1)
        norm_theirs = theirs_flat.norm(dim=1)
        denom = norm_ours * norm_theirs + 1e-8

        cos_sim = dot / denom  # shape [4]

        if torch.isnan(cos_sim).any() or denom.eq(1e-8).any():
            per_timestep_angles.append(np.nan)
        else:
            per_timestep_angles.append(cos_sim.mean().item())

    all_angles.append(per_timestep_angles)

# Average over files, ignoring NaNs
avg_angles = torch.tensor(all_angles)
avg_angles = torch.nanmean(avg_angles, dim=0)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(avg_angles.tolist(), label="Cosine Similarity of Gradients", color='purple', linewidth=2)
plt.xlabel("Timestep")
plt.ylabel("Average Cosine Similarity")
plt.title("Direction Agreement Between Ours and Parihar's Gradients (Conditional)")
plt.ylim(-1, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("grad_cosine_similarity_uncond.png")

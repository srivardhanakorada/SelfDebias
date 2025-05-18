import os
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm
import lpips

original_dir = "their_outputs/original"
debiased_dir = "their_outputs/debiased"
num_visualize = 10
save_vis_path = "comparison_grid_their.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)

all_files = sorted(f.replace("original_","") for f in os.listdir(original_dir) if f.endswith(".png"))
l2_scores, lpips_scores = [], []
samples_to_plot = []

for fname in tqdm(all_files, desc="Comparing images"):
    orig_path = os.path.join(original_dir, "original_"+fname)
    debi_path = os.path.join(debiased_dir, "debiased_"+fname)
    if not os.path.exists(debi_path):
        continue

    img1 = transform(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(Image.open(debi_path).convert("RGB")).unsqueeze(0).to(device)

    l2 = torch.mean((img1 - img2) ** 2).item()
    lp = lpips_model(img1, img2).item()

    l2_scores.append(l2)
    lpips_scores.append(lp)

    if len(samples_to_plot) < num_visualize:
        samples_to_plot.append(torch.cat([img1, img2], dim=3).squeeze(0))  # side-by-side

# Save visualization grid
if samples_to_plot:
    grid = vutils.make_grid(samples_to_plot, nrow=1)
    vutils.save_image(grid, save_vis_path)
    print(f"Saved comparison grid to {save_vis_path}")

# Print metrics
print(f"\nCompared {len(l2_scores)} pairs")
print(f"Average L2 distance:   {sum(l2_scores)/len(l2_scores):.4f}")
print(f"Average LPIPS score:   {sum(lpips_scores)/len(lpips_scores):.4f}")

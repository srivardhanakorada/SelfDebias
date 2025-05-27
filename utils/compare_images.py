# # import os
# # from PIL import Image
# # import torch
# # import torchvision.transforms as T
# # import torchvision.utils as vutils
# # from tqdm import tqdm
# # import lpips

# # original_dir = "pet_outputs/original"
# # debiased_dir = "pet_outputs/debiased"
# # num_visualize = 10
# # save_vis_path = "comparison_grid_their.png"

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # transform = T.Compose([
# #     T.Resize((256, 256)),
# #     T.ToTensor(),
# # ])

# # # LPIPS model
# # lpips_model = lpips.LPIPS(net='alex').to(device)

# # all_files = sorted(f.replace("original_","") for f in os.listdir(original_dir) if f.endswith(".png"))
# # l2_scores, lpips_scores = [], []
# # samples_to_plot = []

# # for fname in tqdm(all_files, desc="Comparing images"):
# #     orig_path = os.path.join(original_dir, "original_"+fname)
# #     debi_path = os.path.join(debiased_dir, "debiased_"+fname)
# #     if not os.path.exists(debi_path):
# #         continue

# #     img1 = transform(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
# #     img2 = transform(Image.open(debi_path).convert("RGB")).unsqueeze(0).to(device)

# #     l2 = torch.mean((img1 - img2) ** 2).item()
# #     lp = lpips_model(img1, img2).item()

# #     l2_scores.append(l2)
# #     lpips_scores.append(lp)

# #     idx = len(l2_scores)
# #     if 90 <= idx < 100:
# #         samples_to_plot.append(torch.cat([img1, img2], dim=3).squeeze(0))  # side-by-side

# # # Save visualization grid
# # if samples_to_plot:
# #     grid = vutils.make_grid(samples_to_plot, nrow=1)
# #     vutils.save_image(grid, save_vis_path)
# #     print(f"Saved comparison grid to {save_vis_path}")

# # # Print metrics
# # print(f"\nCompared {len(l2_scores)} pairs")
# # print(f"Average L2 distance:   {sum(l2_scores)/len(l2_scores):.4f}")
# # print(f"Average LPIPS score:   {sum(lpips_scores)/len(lpips_scores):.4f}")


# import os
# from PIL import Image, ImageDraw, ImageFont
# import torch
# import torchvision.transforms as T
# import torchvision.utils as vutils
# from tqdm import tqdm
# import lpips

# original_dir = "pet_outputs/original"
# debiased_dir = "pet_outputs/debiased"
# output_dir = "comparison_grids"
# os.makedirs(output_dir, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),
# ])

# # LPIPS model
# lpips_model = lpips.LPIPS(net='alex').to(device)

# # Load file list
# all_files = sorted(f.replace("original_", "") for f in os.listdir(original_dir) if f.endswith(".png"))
# assert len(all_files) == 100, f"Expected 100 files, found {len(all_files)}."

# # Metrics
# l2_scores, lpips_scores = [], []
# samples_to_plot = []

# # Font for labeling (fallback to default if truetype font not available)
# try:
#     font = ImageFont.truetype("arial.ttf", 14)
# except:
#     font = ImageFont.load_default()

# def make_labeled_pair_grid(pairs, labels, save_path):
#     imgs_with_text = []
#     for img_tensor, label in zip(pairs, labels):
#         img = T.ToPILImage()(img_tensor)
#         draw = ImageDraw.Draw(img)
#         draw.text((5, 5), label, font=font, fill=(255, 255, 255))
#         imgs_with_text.append(T.ToTensor()(img))
#     grid = vutils.make_grid(imgs_with_text, nrow=len(pairs))
#     vutils.save_image(grid, save_path)

# # Process all files
# for i, fname in enumerate(tqdm(all_files, desc="Comparing images")):
#     orig_path = os.path.join(original_dir, "original_" + fname)
#     debi_path = os.path.join(debiased_dir, "debiased_" + fname)
#     if not os.path.exists(debi_path):
#         continue

#     img1 = transform(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
#     img2 = transform(Image.open(debi_path).convert("RGB")).unsqueeze(0).to(device)

#     l2 = torch.mean((img1 - img2) ** 2).item()
#     lp = lpips_model(img1, img2).item()

#     l2_scores.append(l2)
#     lpips_scores.append(lp)

#     pair_img = torch.cat([img1, img2], dim=3).squeeze(0)  # [3, 256, 512]
#     samples_to_plot.append((pair_img, fname))

#     # Save every 10 image pairs
#     if (i + 1) % 10 == 0:
#         batch_idx = (i + 1) // 10
#         pairs, labels = zip(*samples_to_plot[-10:])
#         grid_path = os.path.join(output_dir, f"grid_batch{batch_idx:02d}.png")
#         make_labeled_pair_grid(pairs, labels, grid_path)
#         print(f"✅ Saved grid {batch_idx} → {grid_path}")

# # Print metrics
# print(f"\nCompared {len(l2_scores)} pairs")
# print(f"Average L2 distance:   {sum(l2_scores) / len(l2_scores):.4f}")
# print(f"Average LPIPS score:   {sum(lpips_scores) / len(lpips_scores):.4f}")
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm
import lpips

original_dir = "pet_outputs/original"
debiased_dir = "pet_outputs/debiased"
output_dir = "comparison_grids_vertical"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)

# File list
all_files = sorted(f.replace("original_", "") for f in os.listdir(original_dir) if f.endswith(".png"))
assert len(all_files) == 100, f"Expected 100 images, found {len(all_files)}."

# Font
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

# Metrics
l2_scores, lpips_scores = [], []
samples_to_plot = []

def make_vertical_pair_grid(pairs, labels, save_path):
    # Each pair: [original; debiased] stacked vertically, then combined horizontally across grid
    stacked_pairs = []
    for (orig_tensor, debi_tensor), label in zip(pairs, labels):
        orig_img = T.ToPILImage()(orig_tensor)
        debi_img = T.ToPILImage()(debi_tensor)

        # Combine vertically
        combined = Image.new("RGB", (256, 512))
        combined.paste(orig_img, (0, 0))
        combined.paste(debi_img, (0, 256))

        # Add label
        draw = ImageDraw.Draw(combined)
        draw.text((5, 5), label, font=font, fill=(255, 255, 255))

        stacked_pairs.append(T.ToTensor()(combined))

    grid = vutils.make_grid(stacked_pairs, nrow=len(stacked_pairs))  # horizontal layout
    vutils.save_image(grid, save_path)

# Loop through files
for i, fname in enumerate(tqdm(all_files, desc="Comparing images")):
    orig_path = os.path.join(original_dir, "original_" + fname)
    debi_path = os.path.join(debiased_dir, "debiased_" + fname)
    if not os.path.exists(debi_path):
        continue

    img1 = transform(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(Image.open(debi_path).convert("RGB")).unsqueeze(0).to(device)

    l2 = torch.mean((img1 - img2) ** 2).item()
    lp = lpips_model(img1, img2).item()

    l2_scores.append(l2)
    lpips_scores.append(lp)

    samples_to_plot.append(((img1.squeeze(0), img2.squeeze(0)), fname))

    # Save every 10
    if (i + 1) % 10 == 0:
        batch_idx = (i + 1) // 10
        pairs, labels = zip(*samples_to_plot[-10:])
        grid_path = os.path.join(output_dir, f"grid_batch{batch_idx:02d}.png")
        make_vertical_pair_grid(pairs, labels, grid_path)
        print(f"✅ Saved vertical grid {batch_idx} → {grid_path}")

# Print metrics
print(f"\nCompared {len(l2_scores)} pairs")
print(f"Average L2 distance:   {sum(l2_scores) / len(l2_scores):.4f}")
print(f"Average LPIPS score:   {sum(lpips_scores) / len(lpips_scores):.4f}")

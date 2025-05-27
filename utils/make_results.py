# import os
# import shutil
# from PIL import Image, ImageDraw, ImageFont
# import torch
# import torchvision.transforms as T
# import torchvision.utils as vutils

# # --- Config ---
# original_dir = "pet_outputs/original"
# debiased_dir = "pet_outputs/debiased"
# output_dir = "comparison_grids_custom"
# copied_dir = os.path.join(output_dir, "selected_images")
# selected_indices = [0, 7, 21, 31, 35, 71]

# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(copied_dir, exist_ok=True)

# # --- File prep ---
# all_files = sorted(f.replace("original_", "") for f in os.listdir(original_dir) if f.endswith(".png"))
# selected_files = [all_files[i] for i in selected_indices]

# # --- Font ---
# try:
#     font = ImageFont.truetype("arial.ttf", 14)
# except:
#     font = ImageFont.load_default()

# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),
# ])

# def make_vertical_pair_grid(pairs, labels, save_path):
#     stacked_pairs = []
#     for (orig_tensor, debi_tensor), label in zip(pairs, labels):
#         orig_img = T.ToPILImage()(orig_tensor)
#         debi_img = T.ToPILImage()(debi_tensor)

#         combined = Image.new("RGB", (256, 512))
#         combined.paste(orig_img, (0, 0))
#         combined.paste(debi_img, (0, 256))

#         draw = ImageDraw.Draw(combined)
#         draw.text((5, 5), label, font=font, fill=(255, 255, 255))

#         stacked_pairs.append(T.ToTensor()(combined))

#     grid = vutils.make_grid(stacked_pairs, nrow=len(stacked_pairs))
#     vutils.save_image(grid, save_path)

# # --- Collect, copy & visualize ---
# pairs, labels = [], []

# for fname in selected_files:
#     orig_filename = "original_" + fname
#     debi_filename = "debiased_" + fname

#     orig_path = os.path.join(original_dir, orig_filename)
#     debi_path = os.path.join(debiased_dir, debi_filename)

#     if not os.path.exists(orig_path) or not os.path.exists(debi_path):
#         print(f"⚠️ Skipping {fname}, missing file")
#         continue

#     # Copy files to copied_dir
#     shutil.copy(orig_path, os.path.join(copied_dir, orig_filename))
#     shutil.copy(debi_path, os.path.join(copied_dir, debi_filename))

#     # Prepare for grid
#     orig_tensor = transform(Image.open(orig_path).convert("RGB"))
#     debi_tensor = transform(Image.open(debi_path).convert("RGB"))
#     pairs.append((orig_tensor, debi_tensor))
#     labels.append(fname)

# # --- Save grid ---
# save_path = os.path.join(output_dir, "selected_grid.png")
# make_vertical_pair_grid(pairs, labels, save_path)
# print(f"✅ Saved selected grid to {save_path}")
# print(f"📁 Copied selected image files to {copied_dir}")


import os
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import torchvision.utils as vutils

# --- Config ---
selected_dir = "comparison_grids_custom/original"
output_path = "comparison_grids_custom/pet_grid.png"

# --- Font ---
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

def make_vertical_pair_grid(pairs, labels, save_path):
    stacked_pairs = []
    for (orig_tensor, debi_tensor), label in zip(pairs, labels):
        orig_img = T.ToPILImage()(orig_tensor)
        debi_img = T.ToPILImage()(debi_tensor)

        combined = Image.new("RGB", (256, 512))
        combined.paste(orig_img, (0, 0))
        combined.paste(debi_img, (0, 256))

        stacked_pairs.append(T.ToTensor()(combined))

    grid = vutils.make_grid(stacked_pairs, nrow=len(stacked_pairs))
    vutils.save_image(grid, save_path)

# --- Get sorted filenames ---
all_files = sorted(f for f in os.listdir(selected_dir) if f.endswith(".png"))
base_names = sorted(set(f.replace("original_", "").replace("debiased_", "") for f in all_files))

# --- Read and pair images ---
pairs, labels = [], []
for base in base_names:
    orig_path = os.path.join(selected_dir, f"original_{base}")
    debi_path = os.path.join(selected_dir, f"debiased_{base}")

    if not os.path.exists(orig_path) or not os.path.exists(debi_path):
        print(f"⚠️ Skipping {base}, missing file")
        continue

    orig_tensor = transform(Image.open(orig_path).convert("RGB"))
    debi_tensor = transform(Image.open(debi_path).convert("RGB"))
    pairs.append((orig_tensor, debi_tensor))
    labels.append(base)

# --- Make and save grid ---
make_vertical_pair_grid(pairs, labels, output_path)
print(f"✅ Grid saved to {output_path}")

import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to parent folder containing 'original' and 'debiased' folders
parent_folder = "final_results/faces"
top_folder = os.path.join(parent_folder, "original")
bottom_folder = os.path.join(parent_folder, "debiased")

# List and sort image filenames
top_images = sorted(os.listdir(top_folder))
bottom_images = sorted(os.listdir(bottom_folder))

# Limit to first 5 pairs (or fewer if needed)
num_images = min(5, len(top_images), len(bottom_images))

# Create grid: 2 rows (original, debiased), 5 columns (one per image)
fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))

for i in range(num_images):
    # Load images
    top_img = Image.open(os.path.join(top_folder, top_images[i]))
    bottom_img = Image.open(os.path.join(bottom_folder, bottom_images[i]))

    # Display original (top row)
    axes[0, i].imshow(top_img)
    axes[0, i].set_title(f"Original {i}")
    axes[0, i].axis("off")

    # Display debiased (bottom row)
    axes[1, i].imshow(bottom_img)
    axes[1, i].set_title(f"Debiased {i}")
    axes[1, i].axis("off")

# Layout and save
plt.tight_layout()
plt.savefig("org_vs_deb_faces_grid.png", dpi=300)
plt.show()

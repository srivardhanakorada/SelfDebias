import os
from PIL import Image
import matplotlib.pyplot as plt

original_dir = 'pinned_results/their_edited/original'
debiased_dir = 'pinned_results/their_edited/debiased'
output_dir = 'pinned_results/output_grids_final_theirs'
os.makedirs(output_dir, exist_ok=True)

# Sorted list of indices from filenames
image_pairs = sorted(os.listdir(original_dir))
indices = [int(fname.split('_')[-1].replace('.png', '')) for fname in image_pairs]

for i in range(0, len(indices), 5):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for j in range(5):
        if i + j >= len(indices):
            axes[0, j].axis('off')
            axes[1, j].axis('off')
            continue
        idx = indices[i + j]
        idx_str = f"{idx:04d}"
        orig_path = os.path.join(original_dir, f'original_{idx_str}.png')
        debiased_path = os.path.join(debiased_dir, f'debiased_{idx_str}.png')

        if not os.path.exists(orig_path) or not os.path.exists(debiased_path):
            axes[0, j].axis('off')
            axes[1, j].axis('off')
            continue

        orig_img = Image.open(orig_path).convert('RGB')
        debiased_img = Image.open(debiased_path).convert('RGB')

        axes[0, j].imshow(orig_img)
        axes[0, j].set_title(f'Original {idx_str}')
        axes[0, j].axis('off')
        axes[1, j].imshow(debiased_img)
        axes[1, j].set_title(f'Debiased {idx_str}')
        axes[1, j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'grid_{i//5:02d}.png'))
    plt.close()

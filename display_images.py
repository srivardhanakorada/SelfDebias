import os
from PIL import Image
import matplotlib.pyplot as plt

original_dir = 'our_outputs/original'
debiased_dir = 'our_outputs/debiased'
output_dir = 'pinned_outputs/output_grids'
os.makedirs(output_dir, exist_ok=True)

image_pairs = sorted(os.listdir(original_dir))
for i in range(0, len(image_pairs), 5):
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))
    for j in range(5):
        if i + j >= len(image_pairs): break
        orig_path = os.path.join(original_dir, f'original_{i+j:04d}.png')
        debiased_path = os.path.join(debiased_dir, f'debiased_{i+j:04d}.png')
        orig_img = Image.open(orig_path).convert('RGB')
        debiased_img = Image.open(debiased_path).convert('RGB')
        axes[j, 0].imshow(orig_img)
        axes[j, 0].set_title(f'Original {i+j:04d}')
        axes[j, 1].imshow(debiased_img)
        axes[j, 1].set_title(f'Debiased {i+j:04d}')
        axes[j, 0].axis('off')
        axes[j, 1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'grid_{i//5:02d}.png'))
    plt.close()

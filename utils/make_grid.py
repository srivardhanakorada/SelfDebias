import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def get_image_files(folder):
    """Get all image files from the folder"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(exts)]
    return sorted(files)  # Sorting for consistent ordering

def create_image_grid(images, rows=8, cols=4, figsize=(20, 40)):
    """Create a single grid of images"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for i, (ax, img_path) in enumerate(zip(axes.flat, images)):
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            ax.axis('off')  # Show empty box if image fails to load
    
    plt.tight_layout()
    return fig

def create_all_grids(folder, output_dir="grid_outputs"):
    """Create multiple grids from all images in folder"""
    all_images = get_image_files(folder)
    total_images = len(all_images)
    
    if total_images < 128:
        print(f"⚠️ Warning: Only {total_images} images found. Need 128 for 4 full grids.")
    
    # Calculate how many grids we can make (32 images per grid)
    num_grids = min(4, total_images // 32)
    if total_images % 32 != 0:
        num_grids = min(4, ceil(total_images / 32))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_grids):
        start_idx = i * 32
        end_idx = start_idx + 32
        grid_images = all_images[start_idx:end_idx]
        
        print(f"Creating grid {i+1} with images {start_idx+1} to {min(end_idx, total_images)}")
        
        fig = create_image_grid(grid_images)
        output_path = os.path.join(output_dir, f"grid_{i+1}.png")
        fig.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        print(f"Saved {output_path}")

# Example usage
create_all_grids("pet_outputs/debiased_simple")
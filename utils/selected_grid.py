# import os
# from PIL import Image
# import argparse

# def create_image_grid(image_folder, indices, output_file="grid.jpg", grid_size=(4, 4)):
#     """
#     Create a grid of images from a folder based on specified indices.
    
#     Args:
#         image_folder (str): Path to folder containing images
#         indices (list): List of image indices to include (0-based)
#         output_file (str): Output filename for the grid
#         grid_size (tuple): (rows, cols) for the grid
#     """
#     # Get all image files from folder
#     image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
#     # Check if we have enough images
#     if len(image_files) < max(indices)+1:
#         raise ValueError(f"Not enough images in folder. Requested index {max(indices)} but only {len(image_files)} images found.")
    
#     # Open images and resize to same size (using first image's size)
#     images = []
#     for idx in indices:
#         img_path = os.path.join(image_folder, image_files[idx])
#         img = Image.open(img_path)
#         images.append(img)
    
#     # Use size of first image for all
#     width, height = images[0].size
    
#     # Create blank grid image
#     grid_img = Image.new('RGB', (width * grid_size[1], height * grid_size[0]))
    
#     # Paste images into grid
#     for i, img in enumerate(images):
#         row = i // grid_size[1]
#         col = i % grid_size[1]
#         grid_img.paste(img, (col * width, row * height))
    
#     # Save grid
#     grid_img.save(output_file)
#     print(f"Saved grid to {output_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--folder", type=str, required=True, help="Folder containing images")
#     parser.add_argument("--indices", type=str, required=True, help="Comma-separated list of indices (0-based)")
#     parser.add_argument("--output", type=str, default="grid.jpg", help="Output filename")
#     args = parser.parse_args()
    
#     # Convert indices string to list of integers
#     indices = [int(i) for i in args.indices.split(",")]
    
#     # Check we have exactly 16 indices for 4x4 grid
#     if len(indices) != 16:
#         raise ValueError(f"Need exactly 16 indices for 4x4 grid, got {len(indices)}")
    
#     create_image_grid(args.folder, indices, args.output)

import os
import matplotlib.pyplot as plt
from PIL import Image

def make_2x4_grid(
    folder='ddim_outputs/original_ddim',                     # Default folder
    indices=[4,220,16,126,8,12,29,202],                  # Default 8 image indices
    output_file='results/ddim_faces_grid_org.pdf',        # Output file
    image_size=(256, 256),
    dpi=300
):
    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    selected_files = [image_files[i] for i in indices if i < len(image_files)]

    fig, axes = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi)

    for ax, fname in zip(axes.flatten(), selected_files):
        path = os.path.join(folder, fname)
        img = Image.open(path).convert('RGB').resize(image_size)
        ax.imshow(img)
        ax.axis('off')

    for ax in axes.flatten()[len(selected_files):]:  # Empty slots (if <8 images)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Example usage:
make_2x4_grid()

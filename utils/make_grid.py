import os
from PIL import Image

def create_image_grid(base_path, indices_row1, indices_row2, output_path="grid.png"):
    """
    Create a 2-row image grid from two lists of indices.
    
    Args:
        base_path (str): Base directory where images are stored
        indices_row1 (list): List of indices for the first row
        indices_row2 (list): List of indices for the second row
        output_path (str): Path to save the resulting grid image
    """
    # Load all images
    images_row1 = []
    for idx in indices_row1:
        img_path = os.path.join(base_path, f"prompt00_{str(idx).zfill(5)}.png")
        images_row1.append(Image.open(img_path))
    
    images_row2 = []
    for idx in indices_row2:
        img_path = os.path.join(base_path, f"prompt00_{str(idx).zfill(5)}.png")
        images_row2.append(Image.open(img_path))
    
    # Check we have at least one image in each row
    if not images_row1 or not images_row2:
        raise ValueError("Each row must have at least one image")
    
    # Get dimensions
    img_width, img_height = images_row1[0].size
    grid_width = max(len(images_row1), len(images_row2)) * img_width
    grid_height = 2 * img_height
    
    # Create new image
    grid = Image.new('RGB', (grid_width, grid_height))
    
    # Paste first row
    for i, img in enumerate(images_row1):
        grid.paste(img, (i * img_width, 0))
    
    # Paste second row
    for i, img in enumerate(images_row2):
        grid.paste(img, (i * img_width, img_height))
    
    # Save the grid
    grid.save(output_path)
    print(f"Grid saved to {output_path}")

# Example usage:
base_path = "/home/teja/vardhan/adp_replics/demo/results/debiased"
indices_row1 = [85 ,14 ,12 ,16 ,76]  # First row will show images 1, 2, 3
indices_row2 = [2 ,7 ,69 ,15 ,27]  # Second row will show images 4, 5, 6
create_image_grid(base_path, indices_row1, indices_row2)
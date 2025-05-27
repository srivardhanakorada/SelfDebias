import os
from PIL import Image
import math

def make_image_grid(folder_path, output_path='grid_output.png', images_per_row=4, image_size=(256, 256)):
    image_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # image_files = [image_files[0],image_files[1],image_files[5],image_files[7],image_files[15],image_files[16]]
    num_images = len(image_files)
    num_rows = math.ceil(num_images / images_per_row)

    # Resize images and store in a list
    images = []
    for fname in image_files:
        img = Image.open(os.path.join(folder_path, fname)).resize(image_size)
        images.append(img)

    # Create the grid
    grid_width = images_per_row * image_size[0]
    grid_height = num_rows * image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        grid_image.paste(img, (col * image_size[0], row * image_size[1]))

    # Save the result
    grid_image.save(output_path)
    print(f"Saved image grid to: {output_path}")

make_image_grid('random_faces.png')

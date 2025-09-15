import os
import math
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
clip_model.eval()

# Text prompts for classification
text_prompts = ["man", "woman"]
text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

# Image transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

def classify_image(image: Image.Image) -> str:
    """Return 'male' or 'female' using CLIP zero-shot classification."""
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs, **text_inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, 2]
        probs = logits_per_image.softmax(dim=1).cpu()
    return torch.argmax(probs)

def make_image_grids(image_folder, grid_rows=4, grid_cols=8, images_per_grid=32, output_folder='grids'):
    output_folder = os.path.join(image_folder, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    total_images = len(image_paths)
    total_grids = math.ceil(total_images / images_per_grid)

    for g in range(total_grids):
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        axes = axes.flatten()

        for i in range(images_per_grid):
            idx = g * images_per_grid + i
            if idx >= total_images:
                axes[i].axis('off')
                continue

            path = image_paths[idx]
            img = Image.open(path).convert("RGB")

            # Classify image
            label = classify_image(img)

            # Add red border if female
            if label.item() == 0:
                img = ImageOps.expand(img, border=24, fill='blue')

            axes[i].imshow(img)
            axes[i].axis('off')

        plt.tight_layout()
        grid_filename = os.path.join(output_folder, f'grid_{g + 1}.png')
        plt.savefig(grid_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f'Saved: {grid_filename}')

if __name__ == '__main__':
    make_image_grids(image_folder="/home/teja/three/final_results/faces_imb/debiased")

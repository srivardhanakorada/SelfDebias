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
text_prompts = ["a male", "a female"]
text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

# Image transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

def classify_image(image: Image.Image) -> str:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs, **text_inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, 2]
        probs = logits_per_image.softmax(dim=1).cpu()
    return "male" if probs[0, 0] > probs[0, 1] else "female"

def make_image_grids(image_folder):
    image_paths = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    mc,fc = 0,0
    for idx in range(len(image_paths)):
        path = image_paths[idx]
        img = Image.open(path).convert("RGB")
        label = classify_image(img)
        if label == "male": mc += 1
        else: fc += 1
    return mc,fc

if __name__ == '__main__':
    mc,fc = make_image_grids(image_folder="final_results/librarian/debiased")
    print(mc,fc)
    prob = mc/(mc+fc)
    print(abs(0.5-prob)/0.5)

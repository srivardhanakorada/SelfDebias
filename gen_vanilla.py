import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

def save_batch(images, folder, start_idx, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(folder, f"{prefix}_{start_idx + i:05d}.png"))

# --- CONFIG ---
seed = 6325
device = 'cuda'
NUM_IMAGES = 8
BATCH_SIZE = 4
PROMPT = "Photo of a pet"
NEG_PROMPT = "multiple, cartoonish, sketch, drawing, blurred, distorted"
OUT_DIR = "pet_outputs/pet"
# ---------------

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

# --- Streamed Generation and Storage ---
for i in tqdm(range(0, NUM_IMAGES, BATCH_SIZE)):
    result, _ = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device).manual_seed(seed + i),
        guidance_scale=7.5,
        num_images_per_prompt=BATCH_SIZE,
        negative_prompt=NEG_PROMPT,
        mode="distribution",
    )
    save_batch(result.images, OUT_DIR, start_idx=i, prefix="original")
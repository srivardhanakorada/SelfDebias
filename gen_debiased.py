import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

def save_batch(images, folder, start_idx, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(folder, f"{prefix}_{start_idx + i:05d}.png"))

# --- CONFIG ---
seed = 42
device = 'cuda'
NUM_IMAGES = 100 
BATCH_SIZE = 4
CHECKPOINT = "pretrained/our_vhl.pt"
PROMPT = "Realistic photo of a single [vehicle], full body visible, neutral background"
NEG_PROMPT = "multiple objects, partial view, cropped, human, animal, painting, cartoon, abstract, distorted, unrealistic rendering"
OUT_DIR = "vhl_outputs/debiased_rec"
# ---------------

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

# DEBIASED generation with streamed saving
for i in tqdm(range(0, NUM_IMAGES, BATCH_SIZE),desc="Debiased Generating"):
    result, grad_list = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device).manual_seed(seed + i),
        guidance_scale=7.5,
        num_images_per_prompt=BATCH_SIZE,
        negative_prompt=NEG_PROMPT,
        loss_strength=1500,
        scaling_strength=400,
        checkpoint_path=CHECKPOINT,
        mode="distribution",
    )
    save_batch(result.images, OUT_DIR, start_idx=i, prefix="debiased")
import argparse
import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import clip
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
args = parser.parse_args()

device = 'cuda:2'
base_path = "/home/teja/three/vardhan"
seed = 8000 + args.start
BATCH_SIZE = 100
PROMPT = "a computer"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
os.makedirs(f"{base_path}/rebuttal/lands",exist_ok=True)
os.makedirs(f"{base_path}/rebuttal/lands/images",exist_ok=True)
os.makedirs(f"{base_path}/rebuttal/lands/h",exist_ok=True)
os.makedirs(f"{base_path}/rebuttal/lands/clip",exist_ok=True)
pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
for img_id in tqdm(range(args.start, args.end),desc="Vanilla"):
    result, _ = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device).manual_seed(seed + img_id),
        guidance_scale=7.5,
        num_images_per_prompt=1,
        ret_h=True
    )
    img = result.images[0]
    img.save(f"{base_path}/rebuttal/lands/images/{img_id:05d}.png")
    print("Saved image to:",f"{base_path}/rebuttal/lands/images/{img_id:05d}.png")
    image_input = clip_preprocess(Image.open(f"{base_path}/rebuttal/lands/images/{img_id:05d}.png")).unsqueeze(0).to(device)
    with torch.no_grad(): clip_vec = clip_model.encode_image(image_input).squeeze().half().cpu()
    torch.save(clip_vec, f"{base_path}/rebuttal/lands/clip/{img_id:05d}.pt")
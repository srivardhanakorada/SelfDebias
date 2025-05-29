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

device = 'cuda'
seed = 8000 + args.start
BATCH_SIZE = 100
PROMPT = "Realistic photo of a single [vehicle], full body visible, neutral background"
NEG_PROMPT = "multiple objects, partial view, cropped, human, animal, painting, cartoon, abstract, distorted, unrealistic rendering"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

os.makedirs("vhl_data",exist_ok=True)
os.makedirs("vhl_data/images",exist_ok=True)
os.makedirs("vhl_data/h",exist_ok=True)
os.makedirs("vhl_data/clip",exist_ok=True)

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
for img_id in tqdm(range(args.start, args.end),desc="Vanilla"):
    result, h_vecs = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device).manual_seed(seed + img_id),
        guidance_scale=7.5,
        num_images_per_prompt=1,
        negative_prompt=NEG_PROMPT,
        ret_h=True
    )
    img = result.images[0]
    img.save(f"vhl_data/images/{img_id:05d}.png")
    h_list = [h_vecs[t][0].half().cpu() for t in range(len(h_vecs))]
    temp = [h_vecs[t][1].half().cpu() for t in range(len(h_vecs))]
    h_list.extend(temp)
    torch.save(h_list, f"vhl_data/h/{img_id:05d}.pt")
    image_input = clip_preprocess(Image.open(f"vhl_data/images/{img_id:05d}.png")).unsqueeze(0).to(device)
    with torch.no_grad(): clip_vec = clip_model.encode_image(image_input).squeeze().half().cpu()
    torch.save(clip_vec, f"vhl_data/clip/{img_id:05d}.pt")
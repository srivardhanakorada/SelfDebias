import argparse
import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import clip
from PIL import Image
import time
from datetime import datetime

print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
args = parser.parse_args()

device = 'cuda:0'
base_path = "/home/teja/three/vardhan"
seed = 8000 + args.start
BATCH_SIZE = 100
PROMPT = "a photo of food on a table, high quality, 8k"
NEG_PROMPT = "(((deformed))), grayscale, closeup, cartoonish, unrealistic, blurry, sketch, drawing"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
os.makedirs(f"{base_path}/rebuttal/foods",exist_ok=True)
os.makedirs(f"{base_path}/rebuttal/foods/images",exist_ok=True)
os.makedirs(f"{base_path}/rebuttal/foods/h",exist_ok=True)
os.makedirs(f"{base_path}/rebuttal/foods/clip",exist_ok=True)

pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
print(pipeline.device)
for img_id in tqdm(range(args.start, args.end),desc="Vanilla"):
    result, h_vecs = pipeline(
        prompt=PROMPT,
        negative_prompt = NEG_PROMPT,
        generator=torch.Generator(device).manual_seed(seed + img_id),
        guidance_scale=7.5,
        num_images_per_prompt=1,
        ret_h=True
    )
    img = result.images[0]
    img.save(f"{base_path}/rebuttal/foods/images/{img_id:05d}.png")
    print("Saved image to:",f"{base_path}/rebuttal/foods/images/{img_id:05d}.png")
    h_list = [h_vecs[t][0].half().cpu() for t in range(len(h_vecs))]
    temp = [h_vecs[t][1].half().cpu() for t in range(len(h_vecs))]
    h_list.extend(temp)
    torch.save(h_list, f"{base_path}/rebuttal/foods/h/{img_id:05d}.pt")
    image_input = clip_preprocess(Image.open(f"{base_path}/rebuttal/foods/images/{img_id:05d}.png")).unsqueeze(0).to(device)
    with torch.no_grad(): clip_vec = clip_model.encode_image(image_input).squeeze().half().cpu()
    torch.save(clip_vec, f"{base_path}/rebuttal/foods/clip/{img_id:05d}.pt")

end_time = time.time()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time taken: {(end_time - start_time)/60:.2f} minutes")
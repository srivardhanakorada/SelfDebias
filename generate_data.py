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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #KAGGLE: use this line to detect the GPU device in Kaggle
# parser.add_argument('--gpu_id', type=int, required=True)
args = parser.parse_args()

# device = f'cuda:{args.gpu_id}'
seed = 8000 + args.start
BATCH_SIZE = 100
PROMPT = "a photo of a person, single person, single face, ultra detailed, raw photo, realistic face"
NEG_PROMPT = "(((deformed))), grayscale, closeup, cartoonish, unrealistic, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# KAGGLE: the below directories were created in /kaggle/working
# os.makedirs("data",exist_ok=True)
# os.makedirs("data/images",exist_ok=True)
# os.makedirs("data/h",exist_ok=True)
# os.makedirs("data/clip",exist_ok=True)

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
for img_id in tqdm(range(args.start, args.end),desc="Vanilla Generation"):
    result, h_vecs = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device).manual_seed(seed + img_id),
        guidance_scale=7.5,
        num_images_per_prompt=1,
        negative_prompt=NEG_PROMPT,
        ret_h=True
    )
    img = result.images[0]
    img.save(f"/kaggle/temp/data/images/{img_id:05d}.png") # KAGGLE: save the image in /kaggle/temp/data/images
    h_list = [h_vecs[t][0].half().cpu() for t in range(len(h_vecs))]
    temp = [h_vecs[t][1].half().cpu() for t in range(len(h_vecs))]
    h_list.extend(temp)
    torch.save(h_list, f"/kaggle/temp/data/h/{img_id:05d}.pt") # KAGGLE: save the h_vecs in /kaggle/temp/data/h
    image_input = clip_preprocess(Image.open(f"/kaggle/temp/data/images/{img_id:05d}.png")).unsqueeze(0).to(device) # KAGGLE: preprocess the image and move it to GPU
    with torch.no_grad(): clip_vec = clip_model.encode_image(image_input).squeeze().half().cpu()
    torch.save(clip_vec, f"/kaggle/temp/data/clip/{img_id:05d}.pt") # KAGGLE: save the clip_vec in /kaggle/temp/data/clip
import argparse
import os
import torch
from diffusers import DDIMPipeline
from tqdm import tqdm
# import clip
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
seed = 8000 + args.start
BATCH_SIZE = 100
PROMPT = "Photo of a pet"
NEG_PROMPT = "multiple, cartoonish, sketch, drawing, blurred, distorted"

# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# clip_model.eval()

# os.makedirs("pet_data",exist_ok=True)
os.makedirs("/kaggle/temp/data/images",exist_ok=True)
os.makedirs("/kaggle/temp/data/h",exist_ok=True)
# os.makedirs("/kaggle/temp/data/clip",exist_ok=True)

pipeline = DDIMPipeline.from_pretrained("google/ddpm-celebahq-256", torch_dtype=torch.float16).to(device)
for img_id in tqdm(range(args.start, args.end),desc="Vanilla"):
    result, h_vecs = pipeline(
        generator=torch.Generator(device).manual_seed(seed + img_id),
        ret_h=True
    )
    # print(len(h_vecs)) 50
    # print(type(h_vecs[0])) 
    # print(h_vecs[0].shape) [1, 512, 8, 8]
    img = result.images[0]
    img.save(f"/kaggle/temp/data/images/{img_id:05d}.png")
    h_list = [h_vecs[t][0].half().cpu() for t in range(len(h_vecs))] #50
    # temp = [h_vecs[t][1].half().cpu() for t in range(len(h_vecs))]
    # h_list.extend(temp)
    torch.save(h_list, f"/kaggle/temp/data/h/{img_id:05d}.pt")
    # image_input = clip_preprocess(Image.open(f"/kaggle/temp/data/images/{img_id:05d}.png")).unsqueeze(0).to(device)
    # with torch.no_grad(): clip_vec = clip_model.encode_image(image_input).squeeze().half().cpu()
    # torch.save(clip_vec, f"/kaggle/temp/data/images/{img_id:05d}.pt")
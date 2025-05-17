import os
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_base = clip.load("ViT-B/32", device=device)

# Augmentation
preprocess_aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.1),
    T.ToTensor(),
    T.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
])

# Paths
H_DIR = "/kaggle/input/h-part1-generated/kaggle/temp/data/h_part1"
IMG_DIR = "/kaggle/input/images-generated"
OUT_DIR = "/kaggle/temp/contrastive_triplets"
os.makedirs(OUT_DIR, exist_ok=True)

num_timesteps = 51
versions = ["cond", "uncond"]  # i=0 for cond, i=1 for uncond

for fname in tqdm(sorted(os.listdir(H_DIR))):
    if not fname.endswith(".pt"):
        continue

    idx = fname.replace(".pt", "")
    h_full = torch.load(os.path.join(H_DIR, fname))  # Interleaved: cond_0, uncond_0, ..., cond_50, uncond_50
    assert len(h_full) == 102, f"{fname} does not have 102 entries."

    img_path = os.path.join(IMG_DIR, f"{idx}.png")
    if not os.path.exists(img_path):
        print(f"Missing image for {idx}")
        continue

    img = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        img_base = preprocess_base(img).unsqueeze(0).to(device)
        img1 = preprocess_aug(img).unsqueeze(0).to(device)
        img2 = preprocess_aug(img).unsqueeze(0).to(device)

        clip_base = clip_model.encode_image(img_base).squeeze(0).cpu()
        clip1 = clip_model.encode_image(img1).squeeze(0).cpu()
        clip2 = clip_model.encode_image(img2).squeeze(0).cpu()

    sample_dict = {"cond": [], "uncond": []}

    for t in range(num_timesteps):
        for i, version in enumerate(versions):  # 0=cond, 1=uncond
            h_t = h_full[t * 2 + i].to(torch.float32)

            sample_dict[version].append({
                "h": h_t.cpu(),
                "clip_base": clip_base,
                "clip1": clip1,
                "clip2": clip2,
                "t": t
            })

    torch.save(sample_dict, os.path.join(OUT_DIR, f"{idx}.pt"))

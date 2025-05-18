import os
import torch
from PIL import Image
import clip
import random

device = "cuda:2" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def count_gender(folder):
    male, female = 0, 0
    ls = sorted(os.listdir(folder))
    for f in ls[:625]:
        if f.endswith(".png"):
            img = preprocess(Image.open(os.path.join(folder, f)).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(img)
                text = clip.tokenize(["a photo of a man", "a photo of a woman"]).to(device)
                txt_feat = clip_model.encode_text(text)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ txt_feat.T).squeeze()
                pred = int(sim[0] > sim[1])
                male += pred
                female += 1 - pred
    return female, male

def save_comparisons(folder1, folder2, out_dir="outputs/comparison", n=10):
    os.makedirs(out_dir, exist_ok=True)
    imgs1 = sorted(f for f in os.listdir(folder1) if f.endswith(".png"))
    imgs2 = sorted(f for f in os.listdir(folder2) if f.endswith(".png"))
    indices = random.sample(range(min(len(imgs1), len(imgs2))), n)
    for idx in indices:
        img1 = Image.open(os.path.join(folder1, imgs1[idx])).convert("RGB")
        img2 = Image.open(os.path.join(folder2, imgs2[idx])).convert("RGB")
        combo = Image.new("RGB", (img1.width * 2, img1.height))
        combo.paste(img1, (0, 0))
        combo.paste(img2, (img1.width, 0))
        combo.save(os.path.join(out_dir, f"compare_{idx:04d}.png"))

f_orig, m_orig = count_gender("our_outputs/original")
f_deb, m_deb = count_gender("our_outputs/debiased")

print(f"Original → Female: {f_orig} | Male: {m_orig} | Ratio (M|F) : {m_orig/f_orig}")
print(f"Debiased → Female: {f_deb} | Male: {m_deb} | Ratio (M|F) : {m_deb/f_deb}")

# save_comparisons("our_outputs/original", "our_outputs/debiased", n=4)

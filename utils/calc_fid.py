import os
import torch
import numpy as np
from PIL import Image
from torchvision.models import inception_v3
from torchvision import transforms
from tqdm import tqdm
from scipy import linalg

def get_inception_model(device="cuda"):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    # Remove last FC layer → keep pool3 features
    model.fc = torch.nn.Identity()
    return model

@torch.no_grad()
def extract_features(folder, model, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    features = []

    for i in tqdm(range(0, len(imgs), batch_size), desc=f"Extracting from {folder}"):
        batch_paths = imgs[i:i+batch_size]
        batch = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(batch).to(device)
        feats = model(batch)  # [B, 2048]
        features.append(feats.cpu())

    return torch.cat(features, dim=0).numpy()

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_fid(folder1, folder2, device="cuda"):
    model = get_inception_model(device)

    feats1 = extract_features(folder1, model, device)
    feats2 = extract_features(folder2, model, device)

    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_score

# Example usage:
fid = compute_fid("our_outputs/original", "their_outputs/debiased")
print("FID:", fid)

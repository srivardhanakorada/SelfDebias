import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def compute_fd_score(image_folder, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Text prompts: "a photo of a man" and "a photo of a woman"
    with torch.no_grad():
        text_tokens = clip.tokenize(["a photo of a man", "a photo of a woman"]).to(device)
        text_features = model.encode_text(text_tokens)  # [2, 512]
        text_features = F.normalize(text_features, dim=-1)

    # Process images
    probs_list = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])

    for fname in tqdm(image_files, desc="Evaluating images for FD"):
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feat = model.encode_image(img_tensor)  # [1, 512]
            image_feat = F.normalize(image_feat, dim=-1)
            logits = image_feat @ text_features.T  # [1, 2]
            probs = F.softmax(logits, dim=-1).squeeze(0)  # [2]
            probs_list.append(probs.cpu())

    # Stack and compute mean probability
    probs_tensor = torch.stack(probs_list)  # [N, 2]
    mean_probs = probs_tensor.mean(dim=0)   # [2]

    # Uniform target distribution
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.size(0))

    # FD = L2 distance between uniform and predicted class distribution
    fd_score = torch.norm(uniform - mean_probs, p=2).item()
    return fd_score

fd = compute_fd_score("their_outputs/debiased")
# fd = compute_fd_score("our_outputs/debiased")
print(f"Fairness Discrepancy (FD) Score: {fd:.4f}")
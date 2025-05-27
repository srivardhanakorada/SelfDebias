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

    with torch.no_grad():
        # Prompts: adjust as needed
        text_tokens = clip.tokenize([
            "photo of a dark skinned pet",
            "photo of a light skinned pet",
        ]).to(device)
        # text_tokens = clip.tokenize([
        #     "photo of a sitting pet",
        #     "photo of a standing pet",
        # ]).to(device)
        # text_tokens = clip.tokenize([
        #     "photo of a dog",
        #     "photo of a cat",
        # ]).to(device)
        text_features = model.encode_text(text_tokens)  # [2, 512]
        text_features = F.normalize(text_features, dim=-1)

    # Process images
    probs_list = []
    class_counts = torch.zeros(2, dtype=torch.int32)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])

    for fname in tqdm(image_files[:100], desc=f"Evaluating {image_folder}"):
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feat = model.encode_image(img_tensor)  # [1, 512]
            image_feat = F.normalize(image_feat, dim=-1)
            logits = image_feat @ text_features.T         # [1, 2]
            probs = F.softmax(logits, dim=-1).squeeze(0)  # [2]
            pred_class = probs.argmax().item()
            class_counts[pred_class] += 1
            probs_list.append(probs.cpu())

    # Stack and compute FD
    probs_tensor = torch.stack(probs_list)   # [N, 2]
    mean_probs = probs_tensor.mean(dim=0)    # [2]
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.size(0))
    fd_score = torch.norm(uniform - mean_probs, p=2).item()

    return fd_score, class_counts.tolist()

# Run evaluation
fd_org, counts_org = compute_fd_score("pet_outputs/original")
fd_our, counts_our = compute_fd_score("pet_outputs/debiased_rec")

print(f"Original FD Score: {fd_org:.4f} | Class Counts: Class 0={counts_org[0]}, Class 1={counts_org[1]}")
print(f"Our Debiased FD Score: {fd_our:.4f} | Class Counts: Class 0={counts_our[0]}, Class 1={counts_our[1]}")

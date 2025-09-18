import os
import torch
import clip
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

prompts = ["a man", "a woman"]
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)

def compute_fd_and_delta(image_folder):
    probs_list = []
    hard_counts = torch.zeros(len(prompts))

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    for fname in tqdm(image_files, desc=f"Evaluating {image_folder}"):
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feat = model.encode_image(img_tensor)
            image_feat = F.normalize(image_feat, dim=-1)
            logits = (image_feat @ text_features.T) / 0.07
            probs = F.softmax(logits, dim=-1).squeeze(0)

        probs_list.append(probs.cpu())
        hard_counts[torch.argmax(probs)] += 1

    probs_tensor = torch.stack(probs_list)
    mean_probs = probs_tensor.mean(dim=0)
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.size(0))
    fd_score = torch.norm(uniform - mean_probs, p=2).item()

    total = hard_counts.sum().item()
    actual_ratio = hard_counts / total  # [p_male, p_female]
    desired_ratio = 0.5
    delta_male = abs(actual_ratio[0] - desired_ratio) / desired_ratio
    delta_female = abs(actual_ratio[1] - desired_ratio) / desired_ratio
    delta = (delta_male + delta_female) / 2  # average deviation

    print(f"FD Score: {fd_score:.4f}")
    print(f"Hard Counts [Man, Woman]: {hard_counts.tolist()}")
    print(f"Deviation Ratio Δ: {delta:.4f}")
    return fd_score, delta

# --- Run on Your Folders ---
src_folder = "outputs/doctor/original"
dst_folder = "outputs/doctor/debiased"

fd_src, delta_src = compute_fd_and_delta(src_folder)
fd_dst, delta_dst = compute_fd_and_delta(dst_folder)

print(f"\n📊 Final Summary:")
print(f"Original  → FD: {fd_src:.4f}, Δ: {delta_src:.4f}")
print(f"Debiased  → FD: {fd_dst:.4f}, Δ: {delta_dst:.4f}")

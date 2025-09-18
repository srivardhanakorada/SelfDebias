import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def compute_fd_score(image_folder, text_features, model, preprocess, device, categories):
    probs_list = []
    counts = torch.zeros(len(categories), dtype=torch.int32)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])

    for fname in tqdm(image_files, desc=f"Evaluating {image_folder}"):
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feat = model.encode_image(img_tensor)
            image_feat = F.normalize(image_feat, dim=-1)
            logits = image_feat @ text_features.T
            probs = F.softmax(logits, dim=-1).squeeze(0)
            probs_list.append(probs.cpu())

            pred_idx = torch.argmax(probs).item()
            counts[pred_idx] += 1

    probs_tensor = torch.stack(probs_list)
    mean_probs = probs_tensor.mean(dim=0)
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.size(0))
    fd_score = torch.norm(uniform - mean_probs, p=2).item()
    return fd_score, counts.tolist()

def evaluate_all_combinations(image_folders, log_path="simple_pet_fd_log.log"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    categories = {
        "Species": [
            "a photo of a dog",
            "a photo of a cat"
        ]
    }

    logs = ["📊 Pet Fairness Evaluation Results:\n"]
    logs.append("="*60)
    
    for category, prompts in categories.items():
        text_tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

        logs.append(f"\n🔍 Category: {category}")
        logs.append("-"*50)
        
        for label, path in image_folders.items():
            score, counts = compute_fd_score(path, text_features, model, preprocess, device, prompts)
            count_str = ", ".join([f"{p.split()[-1]}:{c}" for p, c in zip(prompts, counts)])
            logs.append(f"{label:>10} - FD: {score:.4f}\nCounts: {count_str}")
        logs.append("="*60)

    with open(log_path, "w") as f:
        f.write("\n".join(logs))
    print(f"✅ Results logged to: {log_path}")

# Define your input image directories
image_folders = {
    "Original": "pet_outputs/original",
    "Debiased": "pet_outputs/debiased_simple"
}

evaluate_all_combinations(image_folders)
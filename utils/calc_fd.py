import os
import torch
import clip
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
prompts = ["a man", "a woman"]
# prompts = ["a white person", "a black person", "an indian person", "an asian person"]
# prompts = ["a young person", "an adult person","an old person"]
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
def compute_fd(image_folder):
    probs_list = []
    hard_counts = torch.zeros(len(prompts))
    image_files = []
    for root, _, files in os.walk(image_folder):
        for f in files:
            if f.endswith(('.jpg', '.png')):
                image_files.append(os.path.join(root, f))
    image_files = sorted(image_files)
    for fname in tqdm(image_files):
        img = Image.open(fname).convert("RGB")
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
    print("Hard Counts:", hard_counts.tolist())
    return fd_score

# --- Run on Your Folder ---
path = "rebuttal/faces/debiased_3"
fd = compute_fd(path)
print(f"FD: {fd} ")
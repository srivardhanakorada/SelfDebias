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
        # text_tokens = clip.tokenize(["young asian person","adult asian person", "old asian person", "young indian person", "adult indian person", "old indian person", "young black person", "old black person", "adult black person", "young white person", "old white person", "adult white person"]).to(device)
        # text_tokens = clip.tokenize(["young asian male","adult asian male", "old asian male", "young indian male", "adult indian male", "old indian male", "young black male", "old black male", "adult black male", "young white male", "old white male", "adult white male", "young asian female","adult asian female", "old asian female", "young indian female", "adult indian female", "old indian female", "young black female", "old black female", "adult black female", "young white female", "old white female", "adult white female"]).to(device)
        # text_tokens = clip.tokenize(["asian male","asian female", "black male", "black female", "white male", "white female", "indian male", "indian female"]).to(device)
        # text_tokens = clip.tokenize(["young male","young female", "old male", "old female", "adult male", "adult female"]).to(device)
        # text_tokens = clip.tokenize(["a photo of a man", "a photo of a woman"]).to(device)
        # 1. Race
        text_tokens_race = clip.tokenize([
            "a photo of an asian person",
            "a photo of an indian person",
            "a photo of a black person",
            "a photo of a white person"
        ]).to(device)

        # 2. Age
        text_tokens_age = clip.tokenize([
            "a photo of a young person",
            "a photo of an adult person",
            "a photo of an old person"
        ]).to(device)

        # 3. Gender
        text_tokens_gender = clip.tokenize([
            "a photo of a man",
            "a photo of a woman"
        ]).to(device)

        # 4. Race + Age
        text_tokens_race_age = clip.tokenize([
            "a photo of a young asian person", "a photo of an adult asian person", "a photo of an old asian person",
            "a photo of a young indian person", "a photo of an adult indian person", "a photo of an old indian person",
            "a photo of a young black person", "a photo of an adult black person", "a photo of an old black person",
            "a photo of a young white person", "a photo of an adult white person", "a photo of an old white person"
        ]).to(device)

        # 5. Race + Gender
        text_tokens_race_gender = clip.tokenize([
            "a photo of an asian man", "a photo of an asian woman",
            "a photo of an indian man", "a photo of an indian woman",
            "a photo of a black man", "a photo of a black woman",
            "a photo of a white man", "a photo of a white woman"
        ]).to(device)

        # 6. Age + Gender
        text_tokens_age_gender = clip.tokenize([
            "a photo of a young man", "a photo of a young woman",
            "a photo of an adult man", "a photo of an adult woman",
            "a photo of an old man", "a photo of an old woman"
        ]).to(device)

        # 7. Age + Gender + Race
        text_tokens_age_gender_race = clip.tokenize([
            "a photo of a young asian man", "a photo of an adult asian man", "a photo of an old asian man",
            "a photo of a young indian man", "a photo of an adult indian man", "a photo of an old indian man",
            "a photo of a young black man", "a photo of an adult black man", "a photo of an old black man",
            "a photo of a young white man", "a photo of an adult white man", "a photo of an old white man",
            "a photo of a young asian woman", "a photo of an adult asian woman", "a photo of an old asian woman",
            "a photo of a young indian woman", "a photo of an adult indian woman", "a photo of an old indian woman",
            "a photo of a young black woman", "a photo of an adult black woman", "a photo of an old black woman",
            "a photo of a young white woman", "a photo of an adult white woman", "a photo of an old white woman"
        ]).to(device)
        text_features = model.encode_text(text_tokens_age_gender_race)  # [2, 512]
        text_features = F.normalize(text_features, dim=-1)

    # Process images
    probs_list = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])

    for fname in tqdm(image_files[:100], desc="Evaluating images for FD"):
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

fd_org = compute_fd_score("our_outputs/original")
# fd_thr = compute_fd_score("their_outputs/age_debiased")
fd_our = compute_fd_score("our_outputs/new_debiased")

print(f"Original FD Score: {fd_org:.4f}")
# print(f"Their Debiased FD Score: {fd_thr:.4f}")
print(f"Our Debiased FD Score: {fd_our:.4f}")
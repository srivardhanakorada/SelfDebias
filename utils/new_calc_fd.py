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

def evaluate_all_combinations(image_folders, log_path="simple_face_fd_log.log"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    categories = {
        # Single attribute categories
        "Race": [
            "a photo of an asian person",
            "a photo of an indian person",
            "a photo of a black person",
            "a photo of a white person"
        ],
        "Age": [
            "a photo of a young person",
            "a photo of an adult person",
            "a photo of an old person"
        ],
        "Gender": [
            "a photo of a man",
            "a photo of a woman"
        ],
        
        # Two-attribute combinations
        "Race+Age": [
            "a photo of a young asian person", "a photo of an adult asian person", "a photo of an old asian person",
            "a photo of a young indian person", "a photo of an adult indian person", "a photo of an old indian person",
            "a photo of a young black person", "a photo of an adult black person", "a photo of an old black person",
            "a photo of a young white person", "a photo of an adult white person", "a photo of an old white person"
        ],
        "Race+Gender": [
            "a photo of an asian man", "a photo of an asian woman",
            "a photo of an indian man", "a photo of an indian woman",
            "a photo of a black man", "a photo of a black woman",
            "a photo of a white man", "a photo of a white woman"
        ],
        "Age+Gender": [
            "a photo of a young man", "a photo of a young woman",
            "a photo of an adult man", "a photo of an adult woman",
            "a photo of an old man", "a photo of an old woman"
        ],
        
        # Three-attribute combination
        "Race+Age+Gender": [
            "a photo of a young asian man", "a photo of an adult asian man", "a photo of an old asian man",
            "a photo of a young indian man", "a photo of an adult indian man", "a photo of an old indian man",
            "a photo of a young black man", "a photo of an adult black man", "a photo of an old black man",
            "a photo of a young white man", "a photo of an adult white man", "a photo of an old white man",
            "a photo of a young asian woman", "a photo of an adult asian woman", "a photo of an old asian woman",
            "a photo of a young indian woman", "a photo of an adult indian woman", "a photo of an old indian woman",
            "a photo of a young black woman", "a photo of an adult black woman", "a photo of an old black woman",
            "a photo of a young white woman", "a photo of an adult white woman", "a photo of an old white woman"
        ]
    }

    logs = ["📊 Face Fairness Evaluation Results:\n"]
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
            
            # Format the counts output based on category complexity
            if "Race+Age+Gender" in category:
                # Complex formatting for 3-way combination
                count_str = "\n"
                races = ["Asian", "Indian", "Black", "White"]
                ages = ["Young", "Adult", "Old"]
                genders = ["Male", "Female"]
                
                idx = 0
                for race in races:
                    count_str += f"{race:>6} | "
                    for age in ages:
                        for gender in genders:
                            count_str += f"{gender[0]}:{counts[idx]:<3} "
                            idx += 1
                    count_str += "\n"
            elif "+" in category:
                # Formatting for 2-way combinations
                att1, att2 = category.split("+")
                if att1 == "Race":
                    groups = ["Asian", "Indian", "Black", "White"]
                    subgroups = ["Young", "Adult", "Old"] if att2 == "Age" else ["Male", "Female"]
                else:  # Age+Gender
                    groups = ["Young", "Adult", "Old"]
                    subgroups = ["Male", "Female"]
                
                count_str = "\n"
                idx = 0
                for group in groups:
                    count_str += f"{group:>6} | "
                    for subgroup in subgroups:
                        count_str += f"{subgroup[0]}:{counts[idx]:<3} "
                        idx += 1
                    count_str += "\n"
            else:
                # Simple formatting for single attribute
                count_str = ", ".join([f"{p.split()[-1]}:{c}" for p, c in zip(prompts, counts)])
            
            logs.append(f"{label:>10} - FD: {score:.4f}\nCounts: {count_str}")
        
        logs.append("="*60)

    with open(log_path, "w") as f:
        f.write("\n".join(logs))
    print(f"✅ Results logged to: {log_path}")

# Define your input image directories
image_folders = {
    "Original": "face_outputs/original",
    "Debiased": "face_outputs/debiased_simple"
}

evaluate_all_combinations(image_folders)
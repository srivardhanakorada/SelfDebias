import os
import cv2

import matplotlib.pyplot as plt

def get_image_files(folder, max_images=10,x=0):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(exts)]
    return sorted(files)[x:max_images+x]

def show_images_grid(folder1, folder2, max_images=10, x=0):
    imgs1 = get_image_files(folder1, max_images, x)
    imgs2 = get_image_files(folder2, max_images, x)
    n = min(len(imgs1), len(imgs2), max_images)

    if n == 0:
        print(f"⚠️ Skipping x={x} — no images found.")
        return

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        img1 = cv2.cvtColor(cv2.imread(imgs1[i]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(imgs2[i]), cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Top {i+1}")
        axes[1, i].imshow(img2)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Bottom {i+1}")
    plt.tight_layout()
    plt.savefig(f"scratch/faces_{x}.png")
    plt.close()


for x in range(0,100,10): show_images_grid('pet_outputs/original', 'pet_outputs/check',x=x)

# import os
# import cv2
# import matplotlib.pyplot as plt

# def get_image_files(folder, max_images=100):
#     exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
#     files = [os.path.join(folder, f) for f in os.listdir(folder)
#              if f.lower().endswith(exts)]
#     return sorted(files)[:max_images]

# def show_big_grid(folder1, folder2, total_images=100, per_row=10):
#     imgs1 = get_image_files(folder1, total_images)
#     imgs2 = get_image_files(folder2, total_images)
#     n = min(len(imgs1), len(imgs2), total_images)

#     num_blocks = n // per_row
#     fig, axes = plt.subplots(2 * num_blocks, per_row, figsize=(per_row * 2, num_blocks * 2.5))

#     for block in range(num_blocks):
#         for i in range(per_row):
#             idx = block * per_row + i
#             img1 = cv2.cvtColor(cv2.imread(imgs1[idx]), cv2.COLOR_BGR2RGB)
#             img2 = cv2.cvtColor(cv2.imread(imgs2[idx]), cv2.COLOR_BGR2RGB)

#             axes[2 * block, i].imshow(img1)
#             axes[2 * block, i].axis('off')
#             axes[2 * block, i].set_title(f"Orig {idx+1}", fontsize=8)

#             axes[2 * block + 1, i].imshow(img2)
#             axes[2 * block + 1, i].axis('off')
#             axes[2 * block + 1, i].set_title(f"Debias {idx+1}", fontsize=8)

#     plt.tight_layout()
#     os.makedirs("scratch", exist_ok=True)
#     plt.savefig("scratch/pets_all_100.png", dpi=200)
#     plt.close()

# # Run it
# show_big_grid('pet_outputs/original', 'pet_outputs/debiased', total_images=100)
# import os
# import cv2
# import torch
# import clip
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load CLIP model
# model, preprocess = clip.load("ViT-B/32", device=device)

# selected_indices = [1, 8, 13, 18, 20, 64, 71, 78]  # 1-based

# def get_selected_images(folder, indices):
#     exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
#     files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
#     return [os.path.join(folder, files[i - 1]) for i in indices if i - 1 < len(files)]

# def compute_clip_similarity(image_paths, prompts):
#     with torch.no_grad():
#         text_tokens = clip.tokenize(prompts).to(device)
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         similarities = []
#         for path in image_paths:
#             image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
#             image_features = model.encode_image(image)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             sim = image_features @ text_features.T  # cosine similarity
#             similarities.append(sim.squeeze().cpu())  # shape: (2,) for 2 prompts

#     return torch.stack(similarities)  # shape: (N, 2)

# def show_horizontal_strip_grid(folder1, folder2, indices):
#     imgs1 = get_selected_images(folder1, indices)
#     imgs2 = get_selected_images(folder2, indices)

#     # Compute CLIP similarities
#     prompts = ["a photo of a dog", "a photo of a cat"]
#     sims1 = compute_clip_similarity(imgs1, prompts)
#     sims2 = compute_clip_similarity(imgs2, prompts)

#     # Normalize scores row-wise (per image)
#     sims1_norm = torch.nn.functional.softmax(sims1, dim=1)
#     sims2_norm = torch.nn.functional.softmax(sims2, dim=1)

#     print("\nCLIP Similarities (normalized softmax over dog/cat):")
#     for i, idx in enumerate(indices):
#         print(f"Image {idx}:")
#         print(f"  Original:  dog: {sims1_norm[i,0]:.3f}, cat: {sims1_norm[i,1]:.3f}")
#         print(f"  Debiased:  dog: {sims2_norm[i,0]:.3f}, cat: {sims2_norm[i,1]:.3f}")

#     # Plotting
#     num_images = len(indices)
#     fig = plt.figure(figsize=(num_images * 2.5, 5))
#     spec = gridspec.GridSpec(2, num_images + 1, width_ratios=[0.5] + [1]*num_images, wspace=0.05, hspace=0.1)

#     ax_label1 = fig.add_subplot(spec[0, 0])
#     ax_label1.text(0.5, 0.5, "Original", fontsize=14, ha='center', va='center')
#     ax_label1.axis('off')

#     ax_label2 = fig.add_subplot(spec[1, 0])
#     ax_label2.text(0.5, 0.5, "Debiased", fontsize=14, ha='center', va='center')
#     ax_label2.axis('off')

#     for i in range(num_images):
#         ax1 = fig.add_subplot(spec[0, i + 1])
#         img1 = cv2.cvtColor(cv2.imread(imgs1[i]), cv2.COLOR_BGR2RGB)
#         ax1.imshow(img1)
#         ax1.axis('off')

#         ax2 = fig.add_subplot(spec[1, i + 1])
#         img2 = cv2.cvtColor(cv2.imread(imgs2[i]), cv2.COLOR_BGR2RGB)
#         ax2.imshow(img2)
#         ax2.axis('off')

#     os.makedirs("scratch", exist_ok=True)
#     plt.savefig("scratch/pets_selected.png", dpi=200, bbox_inches='tight')
#     plt.close()

# # Run
# show_horizontal_strip_grid('pet_outputs/original', 'pet_outputs/debiased', selected_indices)


# import random

# def evaluate_random_originals(folder, all_indices, exclude_indices, num_samples=5):
#     exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
#     files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
#     available_indices = [i for i in range(1, len(files)+1) if i not in exclude_indices]

#     sampled_indices = random.sample(available_indices, min(num_samples, len(available_indices)))
#     sampled_paths = [os.path.join(folder, files[i - 1]) for i in sampled_indices]

#     prompts = ["a photo of a dog", "a photo of a cat"]
#     sims = compute_clip_similarity(sampled_paths, prompts)
#     sims_norm = torch.nn.functional.softmax(sims, dim=1)

#     print("\nCLIP Similarities (normalized) for randomly selected ORIGINAL images:")
#     for i, idx in enumerate(sampled_indices):
#         print(f"Image {idx}: dog: {sims_norm[i,0]:.3f}, cat: {sims_norm[i,1]:.3f}")

# # Run additional analysis
# evaluate_random_originals(
#     folder='pet_outputs/original',
#     all_indices=list(range(1, 100)),  # adjust if you know the actual file count
#     exclude_indices=selected_indices,
#     num_samples=5
# )
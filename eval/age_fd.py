import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def age_classifier(image_paths, model_path):
    # FairFace Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load FairFace model
    fairface = models.resnet34(pretrained=False)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)
    fairface.load_state_dict(torch.load(model_path))
    fairface = fairface.to("cuda:0" if torch.cuda.is_available() else "cpu")
    fairface.eval()

    device = next(fairface.parameters()).device
    all_age_outputs = []

    with torch.no_grad():
        for image_path in tqdm(image_paths):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load {image_path}: {e}")
                continue

            image = transform(image).unsqueeze(0).to(device)
            outputs = fairface(image)

            age_outputs = outputs[:, 9:]  # Age logits
            age_outputs = torch.nn.functional.softmax(age_outputs, dim=1)
            all_age_outputs.append(age_outputs.cpu().numpy())

    all_age_outputs = np.vstack(all_age_outputs)

    # Map age bins to age groups
    age_mapping = {
        'Young': [0, 1, 2],       # 0-2, 3-9, 10-19
        'Adult': [3, 4, 5, 6],    # 20-59
        'Old': [7, 8]             # 60-69, 70+
    }

    logits_mean = np.mean(all_age_outputs, axis=0)
    mapped_logits_mean = np.zeros(3)
    for new_label, old_labels in age_mapping.items():
        idx = list(age_mapping.keys()).index(new_label)
        mapped_logits_mean[idx] = np.sum(logits_mean[old_labels])

    # FD score
    fd_logits_mean = np.sqrt(np.sum((mapped_logits_mean - 1/3) ** 2))

    print("\nAge Group Statistics:")
    for i, group in enumerate(['Young', 'Adult', 'Old']):
        print(f"{group}: {mapped_logits_mean[i]:.4f}")
    print(f"FD (logits mean method): {fd_logits_mean:.4f}")

    # Per-image mapping
    age_classes = np.argmax(all_age_outputs, axis=1)
    mapped_ages = np.zeros_like(age_classes)
    for new_label, old_labels in age_mapping.items():
        for old_label in old_labels:
            mapped_ages[age_classes == old_label] = list(age_mapping.keys()).index(new_label)

    return all_age_outputs, mapped_ages, fd_logits_mean


def main():
    parser = argparse.ArgumentParser(description="FairFace Age Classifier + FD Metric")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing images (.png/.jpg)")
    parser.add_argument('--output_file', type=str, required=True, help="File to save age classification results")
    parser.add_argument('--model_path', type=str, default='./res34_fair_align_multi_7_20190809.pt',
                        help="Path to FairFace model weights (.pt)")

    args = parser.parse_args()

    image_paths = glob.glob(os.path.join(args.input_folder, "*.png")) + \
                  glob.glob(os.path.join(args.input_folder, "*.jpg"))

    if len(image_paths) == 0:
        print(f"No images found in {args.input_folder}")
        return

    age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    mapped_labels = ['Young', 'Adult', 'Old']

    age_results, mapped_ages, fd = age_classifier(image_paths, args.model_path)

    with open(args.output_file, 'w') as file:
        for idx, (age_result, mapped_age) in enumerate(zip(age_results, mapped_ages)):
            original_age = age_labels[np.argmax(age_result)]
            mapped_age_label = mapped_labels[mapped_age]
            file.write(f"Image {idx + 1}: original_age: {original_age}, mapped_age: {mapped_age_label}, "
                       f"probabilities: {', '.join([f'{p:.4f}' for p in age_result])}\n")
        file.write(f"\nFD (logits mean method): {fd:.4f}\n")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

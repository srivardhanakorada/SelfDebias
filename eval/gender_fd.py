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
import warnings
warnings.filterwarnings("ignore")


def gender_classifier(image_paths, model_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load FairFace gender classifier
    fairface = models.resnet34(pretrained=False)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)
    fairface.load_state_dict(torch.load(model_path))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fairface = fairface.to(device)
    fairface.eval()

    all_gender_outputs = []
    gender_outputs_all = torch.zeros(2).to(device)

    with torch.no_grad():
        for image_path in tqdm(image_paths):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load {image_path}: {e}")
                continue

            image = transform(image).unsqueeze(0).to(device)

            outputs = fairface(image)
            gender_outputs = outputs[:, 7:9]
            gender_outputs = torch.nn.functional.softmax(gender_outputs, dim=1)

            all_gender_outputs.append(gender_outputs.cpu().numpy())
            gender_outputs_all += gender_outputs.squeeze(0)

    all_gender_outputs = np.vstack(all_gender_outputs)

    # Compute FD
    avg_gender_probs = gender_outputs_all.cpu().numpy() / len(all_gender_outputs)
    fd = np.sqrt(np.sum((avg_gender_probs - 0.5) ** 2))

    male_pred = np.sum(all_gender_outputs[:, 0] > 0.5) / len(all_gender_outputs)
    female_pred = np.sum(all_gender_outputs[:, 1] > 0.5) / len(all_gender_outputs)
    uncertain_pred = 1.0 - (
        np.sum(all_gender_outputs[:, 0] > 0.75) + np.sum(all_gender_outputs[:, 1] > 0.75)
    ) / len(all_gender_outputs)

    print("\nGender Group Statistics:")
    print(f"Male   (>0.5): {male_pred:.4f}")
    print(f"Female (>0.5): {female_pred:.4f}")
    print(f"Uncertain (>0.75 neither): {uncertain_pred:.4f}")
    print(f"FD (logits mean method): {fd:.4f}")

    return all_gender_outputs, fd


def main():
    parser = argparse.ArgumentParser(description="FairFace Gender Classifier + FD Metric")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing images (.png/.jpg)")
    parser.add_argument('--output_file', type=str, required=True, help="File to save classification results")
    parser.add_argument('--model_path', type=str, default='eval/res34_fair_align_multi_4_20190809.pt',
                        help="Path to FairFace model weights (.pt)")

    args = parser.parse_args()
    
    image_paths = sorted(
        glob.glob(os.path.join(args.input_folder, "**", "*.png"), recursive=True) +
        glob.glob(os.path.join(args.input_folder, "**", "*.jpg"), recursive=True)
    )

    if len(image_paths) == 0:
        print(f"No images found in {args.input_folder}")
        return

    gender_outputs, fd = gender_classifier(image_paths, args.model_path)

    with open(args.output_file, 'w') as f:
        for path, output in zip(image_paths, gender_outputs):
            fname = os.path.basename(path)
            f.write(f"Image {fname}: male: {output[0]:.4f}, female: {output[1]:.4f}\n")
        f.write(f"\nFD (logits mean method): {fd:.4f}\n")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

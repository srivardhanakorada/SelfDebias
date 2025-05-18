import os
import shutil

# Configuration
indices = [5,19,25,46,66,78,82,91,93,95]
original_dir = "our_outputs/original"
debiased_dir = "their_outputs/debiased"
original_out = "pinned_results/their_edited/original"
debiased_out = "pinned_results/their_edited/debiased"

os.makedirs(original_out, exist_ok=True)
os.makedirs(debiased_out, exist_ok=True)

for idx in indices:
    idx_str_two = f"{idx:04d}"
    idx_str_one = f"{idx:05d}"

    src_orig = os.path.join(original_dir, f"original_{idx_str_one}.png")
    dst_orig = os.path.join(original_out, f"original_{idx_str_one}.png")

    src_debiased = os.path.join(debiased_dir, f"debiased_{idx_str_two}.png")
    dst_debiased = os.path.join(debiased_out, f"debiased_{idx_str_two}.png")

    if os.path.exists(src_orig):
        shutil.copy(src_orig, dst_orig)
    else:
        print(f"Original image not found: {src_orig}")

    if os.path.exists(src_debiased):
        shutil.copy(src_debiased, dst_debiased)
    else:
        print(f"Debiased image not found: {src_debiased}")

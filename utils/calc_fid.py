# from cleanfid import fid #type:ignore
# import os

# # path to CelebA-HQ dataset
# base_path = "new_final_results/gender/original"
# root_dir = "new_final_results/gender/val_debiased"

# src_paths = [
#     os.path.join(base_path, d)
#     for d in os.listdir(base_path)
#     if os.path.isdir(os.path.join(base_path, d))
# ]

# test_paths = [
#     os.path.join(root_dir, d)
#     for d in os.listdir(root_dir)
#     if os.path.isdir(os.path.join(root_dir, d))
# ]

# fid_scores = []

# for base,path in zip(src_paths,test_paths):
#     score = fid.compute_fid(base, path)
#     fid_scores.append((path, score))
#     print(f"FID score for {path}: {score}")

from cleanfid import fid #type:ignore
import os
base_path = "rebuttal/faces/original_32"
root_dir = "rebuttal/faces/debiased_3"
score = fid.compute_fid(base_path, root_dir)
print(f"FID score for {root_dir}: {score}")

# from cleanfid import fid
# score = fid.compute_fid("final_results/ddim_outputs/debiased_ddim", dataset_name="Cel", dataset_res=1024, dataset_split="trainval")
# print(score)
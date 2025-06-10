import os
import json
import shutil

# --- Paths ---
json_path = "cluster_to_filenames.json"
source_images_dir = "/home/teja/three/vardhan/new_faces/data/images"  # this is where .png images are
dest_root_dir = "clustered_images_faces"

# --- Load mapping ---
with open(json_path, "r") as f:
    cluster_dict = json.load(f)

# --- Copy images ---
for version in ["cond", "uncond"]:
    for t_str, clusters in cluster_dict[version].items():
        timestep_dir = os.path.join(dest_root_dir, version, f"t{int(t_str):02d}")
        for cluster_id, filenames in clusters.items():
            cluster_dir = os.path.join(timestep_dir, str(cluster_id))
            os.makedirs(cluster_dir, exist_ok=True)

            for fname in filenames:
                src = os.path.join(source_images_dir, fname)
                dst = os.path.join(cluster_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    print(f"⚠️ Missing image: {src}")

print(f"✅ Images copied to: {dest_root_dir}/")

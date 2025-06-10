import json
import torch
from collections import defaultdict

# --- Input files ---
CENTROID_PATH = "cluster_centroids.pt"
ATTR_PATH = "cluster_attributes.json"
SAVE_PATH = "attribute_value_representations.pt"

# --- Load data ---
print("📥 Loading centroid and attribute files...")
centroids_data = torch.load(CENTROID_PATH)  # dict[version][timestep][cluster_id] → {centroid, filenames}
with open(ATTR_PATH, "r") as f:
    attributes_data = json.load(f)  # dict[version][timestep][cluster_id] → {attr_key: attr_value}

# --- Output dict ---
final_representations = {"cond": {}, "uncond": {}}

# --- Process ---
for version in ["cond", "uncond"]:
    print(f"🔍 Processing version: {version}")
    for t_str, cluster_dict in attributes_data[version].items():
        t = int(t_str[1:])  # Convert 't00' → 0
        attr_centroids = defaultdict(lambda: defaultdict(list))  # key → value → list of centroids

        for cluster_id_str, attr_dict in cluster_dict.items():
            cluster_id = int(cluster_id_str)
            try:
                centroid = centroids_data[version][t][cluster_id]["centroid"]
            except KeyError:
                continue  # skip if centroid missing

            for attr_key, attr_val in attr_dict.items():
                attr_centroids[attr_key][attr_val].append(centroid)

        # average the centroids per attribute value
        final_representations[version][t] = {}
        for attr_key, value_dict in attr_centroids.items():
            final_representations[version][t][attr_key] = {}
            for attr_val, vectors in value_dict.items():
                stacked = torch.stack(vectors, dim=0)
                avg_vec = stacked.mean(dim=0)
                final_representations[version][t][attr_key][attr_val] = avg_vec

# --- Save ---
torch.save(final_representations, SAVE_PATH)
print(f"✅ Saved attribute value representations to: {SAVE_PATH}")

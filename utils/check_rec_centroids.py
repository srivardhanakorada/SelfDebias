import torch
import os

# --- Config ---
centroid_path = "centroids_recursive/sweighted_centroids_pets.pt"
log_file_path = "centroids_recursive/cluster_weights_log.txt"

# --- Load centroids ---
data = torch.load(centroid_path)
weights = data["weights"]  # {"cond": {t: Tensor}, "uncond": {t: Tensor}}

log_lines = []

for version in ["cond", "uncond"]:
    log_lines.append(f"\n===== {version.upper()} =====")
    for t in range(51):
        if t not in weights[version]:
            log_lines.append(f"[t={t:02d}] No weights found.")
            continue
        w = weights[version][t]
        w_str = ", ".join([f"{v:.4f}" for v in w.tolist()])
        log_lines.append(f"[t={t:02d}] {len(w)} clusters → Weights: [{w_str}]")

# --- Save log ---
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
with open(log_file_path, "w") as f:
    f.write("\n".join(log_lines))

print(f"📘 Cluster weights logged to: {log_file_path}")

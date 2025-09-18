import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("centroid_clip_similarity.csv")

plt.figure(figsize=(10, 4))
for c in [0, 1]:
    subset = df[df["centroid"] == c]
    diff = subset["similarity_dog"] - subset["similarity_cat"]
    plt.plot(subset["timestep"], diff, label=f"Centroid {c} (dog - cat)")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Timestep")
plt.ylabel("Similarity Difference (dog - cat)")
plt.title("Centroid CLIP Similarity Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("scratch/const.png")
# Fully Unsupervised Self-Debiasing of Text-to-Image Diffusion Models

This repository contains the official implementation of our paper:

> **Fully Unsupervised Self-Debiasing of Text-to-Image Diffusion Models**  
> WACV 2026 (Algorithms Track)

---

## 📖 Overview

Text-to-image (T2I) diffusion models such as Stable Diffusion achieve impressive realism, but they also reproduce and sometimes amplify societal and representational biases learned from large-scale web data. Existing debiasing methods are mostly **supervised** — requiring attribute labels, curated datasets, or external classifiers. These approaches are limited when biases are abstract, unknown, or hard to define.

We introduce **SelfDebias**, the first **fully unsupervised, test-time debiasing framework** for T2I diffusion models. Unlike prior methods, SelfDebias does not rely on labeled data, bias axes, or model retraining. Instead, it automatically identifies and mitigates biases during inference.

---

## 🚀 Method

SelfDebias operates in the semantic latent space (*h-space*) of diffusion models and consists of three key modules:

1. **Semantic Projection Module**  
   - Learns a mapping from intermediate UNet activations (h-space) to CLIP image embeddings.  
   - Aligns activations with semantic space using contrastive learning.

2. **Semantic Mode Discovery Module**  
   - Applies **two-stage clustering** (silhouette-based + recursive spectral clustering) to discover semantic modes without prior labels.  
   - Captures both dominant and fine-grained modes (e.g., gender, race, age, or abstract concepts).

3. **Self-Debiasing Module**  
   - During inference, performs **soft cluster assignment** of activations.  
   - Minimizes KL divergence between predicted distributions and a depth-weighted uniform target.  
   - Backpropagates this signal into the h-space mid-generation, steering outputs toward balanced distributions.

This approach requires **no retraining**, **no external supervision**, and is compatible with any UNet-based diffusion model.

---

## ✨ Key Contributions

- **First fully unsupervised debiasing method** for diffusion models at test time.  
- Generalizes across **demographics** (gender, race, age) and **abstract concepts** (e.g., “a peaceful moment”, “fantasy creature”).  
- Outperforms supervised baselines in reducing **Fairness Discrepancy (FD)** while maintaining high image quality (FID).  
- Works across different prompts, diffusion architectures, and even unconditional models.  
- Enables steering toward **arbitrary target distributions** (e.g., simulating demographic skews).

---

## 📊 Results

- On **face generation**, SelfDebias achieves **best FD scores** while preserving FID.  
- On **occupation prompts** (Winobias), it significantly reduces gender skew.  
- On **abstract prompts**, it balances outputs without predefined concept axes.  
- Demonstrates **encoder- and model-agnostic generalization** (e.g., works with OpenCLIP, unconditional DDIM).

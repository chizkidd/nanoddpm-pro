[![CI](https://github.com/chizkidd/nanoddpm-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/chizkidd/nanoddpm-pro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=chizkidd.nanoddpm-pro)<br>
**nanoddpm:** [![Base Repo](https://img.shields.io/badge/%20Base%20Repo-nanoddpm-2ea44f?style=flat-square&logo=github)](https://github.com/chizkidd/nanoddpm)<br>
**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm-pro/blob/main/nanoddpm-pro.ipynb)

# nanoddpm-pro

From‑scratch implementation of **Denoising Diffusion Probabilistic Models** for **MNIST** and **CIFAR‑10** with a **Mini‑UNet, Classifier‑Free Guidance (CFG), DDIM/EDM sampling, Euler/Heun ODE solvers, v‑prediction, PCA‑FID, Sobel sharpness, and KL intensity divergence** – all in **~400 lines** of clean PyTorch.

This project extends the original ~170‑line MNIST implementation ([chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm)) to a **unified, educational powerhouse** that lets you explore the full modern diffusion design space with simple CLI flags.

>**Note:** 
>- **DDIM:** Denoising Diffusion Implicit Models. [[Paper Link](https://arxiv.org/pdf/2010.02502)]
>- **EDM:** Elucidating Diffusion Models. [[Paper Link](https://arxiv.org/pdf/2206.00364)]
>- **v-prediction:** Alternative training velocity target used in [SD2+](https://arxiv.org/pdf/2112.10752)/[Imagen](https://arxiv.org/pdf/2210.02303).  [[Paper Link](https://arxiv.org/pdf/2202.00512)] 

## Features
| Feature | Implementation |
|---------|---------------|
| **Dual dataset support** | MNIST (28×28 grayscale) and CIFAR‑10 (32×32 RGB) |
| **Mini‑UNet** | Residual blocks, sinusoidal time/class embeddings |
| **Classifier‑Free Guidance** | Joint conditioned/unconditioned training, steer generation at inference |
| **DDIM sampling** | Discrete timesteps, deterministic reverse loop (`--sampler ddim`) |
| **EDM sampling** | Continuous noise, preconditioning, Euler/Heun ODE solvers (`--sampler edm`) |
| **EDM ODE Solver Options** | 1st-order ODE (`--solver euler`) or 2nd-order ODE (`--solver heun`) |
| **Two training targets** | Noise prediction (`--target epsilon`) or v‑prediction (`--target v`) |
| **Two beta schedules** | Linear (`--beta_schedule linear`) or cosine (`--beta_schedule cosine`) |
| **Exponential Moving Average** | Improves sample quality (`--use_ema`) |
| **Lightweight metrics** | PCA‑FID, Sobel edge sharpness, intensity KL divergence (no InceptionV3) |
| **Single‑file design** | ~400 lines, CLI‑ready, Colab‑friendly |

## Project Lineage
| Project | Description | File |
|---------|-------------|------|
| **nanoddpm** | From‑scratch DDPM for MNIST (~170 lines) | [chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm) |
| **nanoddpm-pro** | Unified DDIM/EDM + CFG + v‑prediction + PCA‑FID + MNIST/CIFAR‑10 | `nanoddpm-pro.py` |

## Quick Start
```bash
git clone https://github.com/chizkidd/nanoddpm-pro.git
cd nanoddpm-pro
pip install -r requirements.txt

# EDM + v‑prediction + Heun on CIFAR‑10 (recommended for quality)
python nanoddpm-pro.py --dataset cifar10 --sampler edm --target v --solver heun --epochs 20 --cfg_scale 4.0

# DDIM baseline on MNIST (great for learning)
python nanoddpm-pro.py --dataset mnist --sampler ddim --target epsilon --epochs 20 --cfg_scale 3.0

# Colab free tier friendly (small resolution, few steps)
python nanoddpm-pro.py --dataset cifar10 --sampler edm --target epsilon --solver euler --epochs 10 --batch_size 16 --resize 16 --sample_steps 10
```

## How It Works
| Stage | DDIM (`--sampler ddim`) | EDM (`--sampler edm`) |
|-------|------------------------|----------------------|
| **Noise Schedule** | Linear or cosine `β_t`, cumulative `ᾱ_t` | Log‑normal `σ ~ exp(𝒩(-1.2, 1.2²))` |
| **Forward Process** | `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε` | `x_σ = x₀ + σ·ε` |
| **Training Target** | Predict `ε` or `v` | Predict `ε` (wrapped as x₀) or `v` |
| **Weighted Loss** | SNR‑weighted for `ε`, uniform for `v` | 1/`c_out`² for `ε`, uniform for `v` |
| **Sampling** | Deterministic reverse loop (`sample_ddim`) | ODE solver: Euler or Heun (`edm_sampler`) |
| **Metrics** | PCA‑FID, Sobel, KL, loss | Same |

## Project Structure
```
nanoddpm-pro/
├── .github/workflows/ci.yml          # CI: tests 8 configs on CPU
├── archive/                          # Legacy older version (v0) implementations
├── blog-pro.md                       # Math walkthrough (CFG, DDIM, EDM, v‑pred, PCA‑FID)
├── LICENSE
├── nanoddpm-pro.ipynb                # Colab notebook
├── nanoddpm-pro.py                   # Unified implementation
├── README.md
└── requirements.txt                  # torch, torchvision, numpy, matplotlib, tqdm
```

## CLI Reference

### Shared Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `cifar10` | `mnist` or `cifar10` |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Samples per batch |
| `--cfg_scale` | `4.0` | Guidance strength (1.0 = off, 3.0‑7.5 = strong) |
| `--resize` | `32` | Image resolution (used for CIFAR‑10 only) |
| `--device` | `auto` | `cpu` or `cuda` |
| `--target` | `epsilon` | Training objective: `epsilon` (noise) or `v` (v‑prediction) |
| `--beta_schedule` | `cosine` | `linear` or `cosine` (DDIM only) |
| `--use_ema` | `True` | Use exponential moving average |
| `--ema_decay` | `0.999` | EMA decay rate |

### Sampler Selection
| Flag | Default | Description |
|------|---------|-------------|
| `--sampler` | `edm` | `ddim` or `edm` |

### DDIM‑Specific Flags (`--sampler ddim`)
| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `1000` | Diffusion timesteps `T` |

### EDM‑Specific Flags (`--sampler edm`)
| Flag | Default | Description |
|------|---------|-------------|
| `--sample_steps` | `20` | ODE solver discretization steps |
| `--solver` | `euler` | `euler` (fast) or `heun` (quality) |

## Metrics Explained
| Metric | What it measures | Range / Interpretation |
|--------|-----------------|------------------------|
| **PCA‑FID** | Similarity between real and generated distributions | Lower = better (0 = identical) |
| **Sobel gradient** | Average edge sharpness | Higher = sharper (blurry → low) |
| **KL intensity** | Divergence of pixel intensity histograms | Lower = more similar to real data |

## Learn More
- `blog-pro.md` → Step‑by‑step math: CFG, DDIM reverse loop, EDM preconditioning, v‑prediction, Heun correction, PCA‑FID.
- `nanoddpm-pro.py` → Unified source code (read top‑to‑bottom like a textbook).
- `nanoddpm-pro.ipynb` → Colab notebook.
- `archive/` → Legacy older version implementations for reference.

## Acknowledgments
Inspired by Andrej Karpathy’s educational builds (`microGPT`) and the foundational papers: DDPM (Ho et al.), DDIM (Song et al.), EDM (Karras et al.), and v‑prediction (Salimans & Ho). Built for students, tinkerers, and anyone who believes diffusion shouldn’t be a black box.

## Philosophy
- Readability of code ⇔ learnable mathematics.
- No black boxes, no heavy wrappers. Just raw PyTorch, explicit diffusion equations, and a Mini‑UNet that fits in a single file.

## License
MIT

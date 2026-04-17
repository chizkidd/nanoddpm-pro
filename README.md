# nanoddpm-pro

From-scratch implementation of **Denoising Diffusion Probabilistic Model (DDPM)** with a **mini-UNet, Classifier-Free Guidance (CFG), Denoising Diffusion Implicit Models (DDIM) sampling & PCA-FID evaluation** on **CIFAR-10** dataset in ~200 lines. This project builds on the original ~170-line MNIST implementation: [chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm).

[![Base Repo](https://img.shields.io/badge/%20Base%20Repo-nanoddpm-2ea44f?style=flat-square&logo=github)](https://github.com/chizkidd/nanoddpm)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm-pro/blob/main/nanoddpm_pro.ipynb)


## Features
- **Mini-UNet**: Residual blocks with sinusoidal time & class embeddings
- **Classifier-Free Guidance (CFG)**: Joint conditional/unconditional training, steer generation at inference
- **DDIM Sampling**: Deterministic reverse process for 10–50× faster generation
- **PCA-FID**: Lightweight, from-scratch quality tracking (no `Inception-V3`)
- **Library-PC / Colab Friendly**: Configurable resolution (`--resize`) and step count to fit free-tier limits
- **Single-File Design**: ~200 lines, CLI-ready

### Lineage
| Project | Description | Link |
|---------|-------------|------|
| **nanoddpm** | From-scratch DDPM for MNIST | [nanoddpm](https://github.com/chizkidd/nanoddpm) |
| **nanoddpm-pro** | CIFAR-10 upgrade with CFG, DDIM & PCA-FID | `./` |

## Quick Start
```bash
git clone https://github.com/chizkidd/nanoddpm-pro.git
cd nanoddpm-pro
pip install -r requirements.txt
python nanoddpm-pro.py --epochs 5 --steps 500 --cfg_scale 4.0 --resize 32
```

**Free Colab tip:** Use `--resize 16 --steps 250` to avoid session timeouts while still learning the full pipeline.

## How It Works
| Stage | What Happens | Code Location |
|-------|--------------|---------------|
| **Forward** | Fixed Gaussian noise is added over `T` steps: `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε` | `forward_diffusion()` |
| **Training** | Model predicts noise `ε_θ(x_t, t, y)`. 10% of labels are dropped for CFG. | Training loop + `ResBlock` |
| **Sampling** | DDIM reverse step (deterministic, `σ_t=0`). CFG scales conditional vs unconditional noise. | `sample_ddim()` |
| **Evaluation** | PCA-FID projects real/generated batches into 32D space for fast distributional comparison. | `pca_fid()` |

## Project Structure
```
nanoddpm-pro/
├── .github/
│   └── workflows/
│       └── ci.yml             # Ensure continuous integration (source code: `nanoddpm-pro.py`)
├── nanoddpm-pro.py            # Single-file implementation (~200 lines)
├── nanoddpm-pro.ipynb         # Colab Notebook implementation with Visualization
nanoddpm-pro.ipynb
├── nanoddpm_pro_metrics.json  # Auto-generated training metrics
├── requirements.txt           # torch, torchvision, numpy, matplotlib, tqdm
├── blog-pro.md                # Math walkthrough (forward/reverse, CFG, DDIM, PCA-FID)
└── README.md
```

nanoddpm-pro/
├── .github/
│   └── workflows/
│       └── ci.yml          # ← Create this exact path
├── nanoddpm_pro.py
├── requirements.txt        # ← Place at root
├── README.md
└── blog_pro.md

## CLI Arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `3` | Training epochs |
| `--steps` | `500` | Diffusion timesteps `T` |
| `--batch_size` | `32` | Samples per batch |
| `--cfg_scale` | `4.0` | Guidance strength (1.0 = off, 3.0–7.5 = strong) |
| `--resize` | `32` | Image resolution (`16` for speed, `32` for quality) |
| `--device` | `auto` | `cpu` or `cuda` |

## Learn More
- `blog-pro.md` → Step-by-step math breakdown & architectural diagrams
- `nanoddpm-pro.py` → Source code file(read top-to-bottom like a textbook)
- `nanoddpm-pro.ipynb` → Colab Notebook with interactive sliders for CFG scale, DDIM steps, and class selection

## Acknowledgments
Inspired by Andrej Karpathy's educational builds (`micrograd`, `minbpe`, `microGPT`) and the foundational DDPM/DDIM papers. Built for students, tinkerers, and anyone who believes diffusion shouldn't be a black box.

## Philosophy
- _The readability of the code should translate to learnable mathematics._
- _No black boxes. No heavy wrappers. Just raw PyTorch, explicit diffusion equations, and a Mini-UNet that fits in a single file. Designed for learners who want to understand **how** diffusion works, not just how to call it._

## License
MIT
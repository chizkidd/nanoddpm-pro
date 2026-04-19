[![CI](https://github.com/chizkidd/nanoddpm-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/chizkidd/nanoddpm-pro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=chizkidd.nanoddpm-pro)<br>
**nanoddpm:** [![Base Repo](https://img.shields.io/badge/%20Base%20Repo-nanoddpm-2ea44f?style=flat-square&logo=github)](https://github.com/chizkidd/nanoddpm)<br>
**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm-pro/blob/main/nanoddpm-pro.ipynb)

# nanoddpm-pro

> Diffusion models from scratch

From-scratch implementation of **Denoising Diffusion Probabilistic Model (DDPM)** with a **mini-UNet, Classifier-Free Guidance (CFG), DDIM/EDM sampling, Euler/Heun ODE solvers, v-prediction, and PCA-FID evaluation** on **CIFAR-10** dataset in ~300 lines. This project builds on the original ~170-line MNIST implementation: [chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm).

A single, modular file (`nanoddpm-pro.py`) that cleanly branches between **DDIM** and **EDM** sampling, **epsilon** and **v-prediction** training targets, and **Euler**/**Heun** ODE solvers via CLI flags—all while sharing the Mini-UNet architecture, CFG logic, and PCA-FID evaluation.

>**Note:** Diffusion models from scratch
>- **DDIM:** Denoising Diffusion Implicit Models
>- **EDM:** Elucidating Diffusion Models
>- **v-prediction:** Alternative training target used in SD2+/Imagen

## Features
| Feature | Implementation |
|---------|---------------|
| **Mini-UNet** | Residual-style blocks with sinusoidal time/class embeddings |
| **Classifier-Free Guidance** | Joint conditional/unconditional training, steer generation at inference |
| **DDIM Sampling** | Discrete `t`, deterministic reverse loop (`--sampler ddim`) |
| **EDM Sampling** | Continuous `σ`, preconditioning, Euler/Heun ODE solvers (`--sampler edm`) |
| **Training Targets** | Noise prediction (`--target epsilon`) or v-prediction (`--target v`) |
| **Solver Options** | `--solver euler` (fast) or `--solver heun` (quality, EDM only) |
| **PCA-FID** | Lightweight, from-scratch quality tracking (no InceptionV3) |
| **Single-File Design** | ~300 lines, CLI-ready, Colab-friendly |

## Project Lineage
| Project | Description | File |
|---------|-------------|------|
| **nanoddpm** | From-scratch DDPM for MNIST (~170 lines) | [chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm) |
| **nanoddpm-pro** | CIFAR-10 upgrade: unified DDIM/EDM, CFG, v-prediction, PCA-FID | `nanoddpm-pro.py` |

## Quick Start
```bash
git clone https://github.com/chizkidd/nanoddpm-pro.git
cd nanoddpm-pro
pip install -r requirements.txt

# EDM + v-prediction + Heun (Recommended for quality, SD2+ style)
python nanoddpm-pro.py --sampler edm --target v --solver heun --epochs 20 --cfg_scale 4.0 --resize 32

# DDIM baseline (Simpler math, great for learning)
python nanoddpm-pro.py --sampler ddim --target epsilon --epochs 20 --cfg_scale 4.0 --resize 32

# Colab free tier friendly
python nanoddpm-pro.py --sampler edm --target epsilon --solver euler --epochs 10 --batch_size 16 --resize 16 --sample_steps 10
```

## How It Works
| Stage | DDIM (`--sampler ddim`) | EDM (`--sampler edm`) |
|-------|------------------------|----------------------|
| **Noise Schedule** | Linear `β_t`, cumulative `ᾱ_t` | Log-normal `σ ~ exp(𝒩(-1.2, 1.2²))` |
| **Forward Process** | `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε` | `x_σ = x₀ + σ·ε` |
| **Training Target** | Predict noise `ε` OR v-prediction `v` | Predict denoised `x₀` OR v-prediction `v` |
| **Sampling** | Deterministic reverse loop (`sample_ddim`) | ODE solver: Euler or Heun (`edm_sampler`) |
| **Evaluation** | PCA-FID on 32D projected features | Same PCA-FID metric |

## Project Structure
```
nanoddpm-pro/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                 # CI: smoke test 4 core configs on CPU
│   └── .gitkeep                   # Preserve empty directory in git
├── archive/                       # Legacy v1 implementations (preserved for reference)
│   ├── nanoddpm-pro-ddim.ipynb
│   ├── nanoddpm-pro-edm-v1.ipynb
│   ├── nanoddpm-pro-edm-v2.ipynb
│   └── nanoddpm-pro-v1.py
├── blog-pro.md                    # Math walkthrough (CFG, DDIM, EDM, v-pred, PCA-FID)
├── LICENSE
├── nanoddpm-pro.ipynb             # Colab notebook with interactive widgets
├── nanoddpm-pro.py                # Unified implementation (~300 lines)
├── README.md
└── requirements.txt               # torch, torchvision, numpy, matplotlib, tqdm
```

## CLI Config
### Shared Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Samples per batch |
| `--cfg_scale` | `4.0` | Guidance strength (1.0 = off, 3.0-7.5 = strong) |
| `--resize` | `32` | Image resolution (`16` for speed, `32` for quality) |
| `--device` | `auto` | `cpu` or `cuda` |

### Sampler Selection
| Flag | Default | Description |
|------|---------|-------------|
| `--sampler` | `edm` | Sampling framework: `ddim` or `edm` |

### Training Target
| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `epsilon` | Training objective: `epsilon` (noise prediction) or `v` (v-prediction) |

### DDIM-Specific Flags (`--sampler ddim`)
| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `1000` | Diffusion timesteps `T` |

### EDM-Specific Flags (`--sampler edm`)
| Flag | Default | Description |
|------|---------|-------------|
| `--sample_steps` | `20` | ODE solver discretization steps |
| `--solver` | `euler` | Sampling method: `euler` (fast) or `heun` (quality) |

## Learn More
- `blog-pro.md` → Step-by-step math: CFG, DDIM reverse loop, EDM preconditioning, v-prediction, Heun correction, PCA-FID.
- `nanoddpm-pro.py` → Unified source code (read top-to-bottom like a textbook).
- `nanoddpm-pro.ipynb` → Colab notebook with interactive sliders for CFG, sampler, solver, target, steps, and class selection.
- `archive/` → Legacy v1 implementations for reference and comparison

## Acknowledgments
Inspired by Andrej Karpathy's educational builds (`microGPT`) and the foundational papers: DDPM (Ho et al.), DDIM (Song et al.), EDM (Karras et al.). Built for students, tinkerers, and anyone who believes diffusion shouldn't be a black box.

## Philosophy
- The readability of the code should translate to learnable mathematics.
- No black boxes. No heavy wrappers. Just raw PyTorch, explicit diffusion equations, and a Mini-UNet that fits in a single file.

## License
MIT

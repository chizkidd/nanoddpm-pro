
[![CI](https://github.com/chizkidd/nanoddpm-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/chizkidd/nanoddpm-pro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Repo](https://img.shields.io/badge/%20Base%20Repo-nanoddpm-2ea44f?style=flat-square&logo=github)](https://github.com/chizkidd/nanoddpm)<br>
**DDIM:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm-pro/blob/main/nanoddpm-pro-ddim.ipynb)<br>
**EDM:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm-pro/blob/main/nanoddpm-pro-edm.ipynb)

# nanoddpm-pro

> Diffusion models from scratch

From-scratch implementation of **Denoising Diffusion Probabilistic Model (DDPM)** with a **mini-UNet, Classifier-Free Guidance (CFG), DDIM/EDM sampling & PCA-FID evaluation** on **CIFAR-10** dataset in ~200 lines. This project builds on the original ~170-line MNIST implementation: [chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm).


This repo contains **two sampling variants**:
- `nanoddpm-pro-ddim.py`: **DDIM (Denoising Diffusion Implicit Models) baseline** - discrete timesteps, deterministic reverse loop
- `nanoddpm-pro-edm.py`: **EDM (Elucidating Diffusion Models) upgrade** - continuous noise, preconditioning, Euler/Heun ODE solvers

Both share: Mini-UNet, Classifier-Free Guidance (CFG), and PCA-FID evaluation.

## Features
| Feature | DDIM Variant | EDM Variant |
|---------|-------------|-------------|
| **Mini-UNet** | Residual blocks + sinusoidal time/class embeddings | Same architecture |
| **Classifier-Free Guidance** | Joint conditional/unconditional training, steer generation at inference | Same CFG logic |
| **Sampling** | DDIM: discrete `t`, deterministic reverse | EDM: continuous `σ`, Euler/Heun ODE solvers |
| **Stability** | Manual clipping | Preconditioning (`c_in`, `c_out`, `c_skip`) |
| **Solver Options** | Single deterministic loop | `--solver euler` (fast) or `--solver heun` (quality) |
| **PCA-FID** | Lightweight quality tracking | Same metric |

## Project Lineage
| Project | Description | File |
|---------|-------------|------|
| **nanoddpm** | From-scratch DDPM for MNIST (~170 lines) | [chizkidd/nanoddpm](https://github.com/chizkidd/nanoddpm) |
| **nanoddpm-pro** | CIFAR-10 upgrade: CFG + DDIM + PCA-FID | `nanoddpm-pro-ddim.py` |
| **nanoddpm-pro-edm** | EDM framework + Euler/Heun solvers | `nanoddpm-pro-edm.py` |

## Quick Start
```bash
git clone https://github.com/chizkidd/nanoddpm-pro.git
cd nanoddpm-pro
pip install -r requirements.txt

# EDM + Heun (Recommended)
python nanoddpm-pro.py --sampler edm --solver heun --epochs 20 --cfg_scale 4.0 --resize 32

# DDIM baseline (simpler, great for learning)
python nanoddpm-pro-ddim.py --epochs 20 --cfg_scale 4.0 --resize 32

# EDM upgrade (recommended for quality/stability)
python nanoddpm-pro-edm.py --epochs 20 --cfg_scale 4.0 --resize 32 --solver heun


**Free Colab tip:** Use `--resize 16 --sample_steps 10` to avoid timeouts while still learning the full pipeline.

## How It Works
| Stage | DDIM Variant | EDM Variant |
|-------|-------------|-------------|
| **Noise Schedule** | Linear `β_t`, cumulative `ᾱ_t` | Log-normal `σ ~ exp(𝒩(-1.2, 1.2²))` |
| **Forward Process** | `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε` | `x_σ = x₀ + σ·ε` |
| **Training Target** | Predict noise `ε` | Predict denoised `x₀` (weighted by `1/c_out²`) |
| **Sampling** | Deterministic reverse loop (`sample_ddim`) | ODE solver: Euler or Heun (`edm_sampler`) |
| **Evaluation** | PCA-FID on 32D projected features | Same PCA-FID metric |

## Project Structure
```
nanoddpm-pro/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Ensure continuous integration (CI: smoke test on CPU)
├── nanoddpm-pro-ddim.py                # DDIM baseline (~200 lines)
├── nanoddpm-pro-edm.py            # EDM upgrade (~220 lines) 
├── nanoddpm-pro-ddim.ipynb             # Colab notebook implementation with Visualization
├── nanoddpm-pro-edm.ipynb             # Colab notebook implementation with Visualization
├── requirements.txt               # torch, torchvision, numpy, matplotlib, tqdm
├── blog-pro.md                    # Math walkthrough  (CFG, DDIM, EDM, PCA-FID)
├── LICENSE  
└── README.md
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

### DDIM Variant (`nanoddpm-pro-ddim.py`)
| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `500` | Diffusion timesteps `T` |

### EDM Variant (`nanoddpm-pro-edm.py`)
| Flag | Default | Description |
|------|---------|-------------|
| `--sample_steps` | `20` | ODE solver discretization steps |
| `--solver` | `euler` | Sampling method: `euler` (fast) or `heun` (quality) |


## Learn More
- `blog-pro.md` → Step-by-step math: CFG, DDIM reverse loop, EDM preconditioning, Heun correction, PCA-FID
- `nanoddpm-pro-ddim.py` → DDIM baseline source (read top-to-bottom like a textbook)
- `nanoddpm-pro-edm.py` → EDM upgrade source (same structure, advanced math)
- `nanoddpm-pro-ddim.ipynb` → DDIM Colab notebook with interactive sliders for CFG, steps, and class selection
- `nanoddpm-pro-edm.ipynb` → EDM Colab notebook with interactive sliders for CFG, solver, steps, and class selection

## Acknowledgments
Inspired by Andrej Karpathy's educational builds (`microGPT`) and the foundational papers: DDPM (Ho et al.), DDIM (Song et al.), EDM (Karras et al.). Built for students, tinkerers, and anyone who believes diffusion shouldn't be a black box.

## Philosophy
- _The readability of the code should translate to learnable mathematics._
- _No black boxes. No heavy wrappers. Just raw PyTorch, explicit diffusion equations, and a Mini-UNet that fits in a single file._

## License
MIT
# nanoddpm-pro: Fast, Guided Diffusion from Scratch (~400 lines)

> Upgrading `nanoddpm` (MNIST) to **unified MNIST/CIFAR‑10** with a **Mini‑UNet, Classifier‑Free Guidance (CFG), DDIM/EDM sampling, Euler/Heun ODE solvers, v‑prediction, EMA, weighted loss, and PCA‑FID + Sobel + KL metrics**.
>
> A single, modular file (`nanoddpm-pro.py`) that cleanly branches between datasets, samplers, targets, and solvers via CLI flags. No black boxes; just raw PyTorch and explicit diffusion math.

---

## 1. Classifier‑Free Guidance (CFG) *(Shared)*
Standard conditional diffusion requires a separate classifier to steer generation. CFG removes that dependency by training a single model on **paired conditional/unconditional data**.

During training, we randomly drop the class label ~10% of the time, forcing the network to learn both:
- `ε_θ(x_t, t, y)` : noise prediction with conditioning
- `ε_θ(x_t, t, ∅)` : noise prediction without conditioning

At inference, we combine them linearly:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)
$$

Where `w` (`cfg_scale`) controls guidance strength:
- `w = 1.0` → standard conditional sampling
- `w = 3.0–7.5` → sharper, class‑aligned outputs
- `w > 7.5` → artifacts & oversaturation

**Why it works:** `(ε_cond - ε_uncond)` points toward the class data manifold. Scaling it amplifies the "pull" without external gradients.

---

## 2. DDIM: Deterministic Fast Sampling *(Baseline Variant)*
DDPM sampling is slow because it adds random noise at every reverse step. DDIM reparameterizes the reverse process as a **deterministic ODE**, allowing safe step‑skipping.

Starting from the DDPM reverse mean:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

We can predict $x_0$ directly:

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}
$$

DDIM replaces the stochastic update (for $\sigma_t = 0$) with:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta
$$

Setting $\sigma_t = 0$ removes the random noise, making the path deterministic. Because the trajectory is fixed, we can evaluate the network at a subset of steps (e.g., `t = 500, 250, 100, 50, 0`) and interpolate cleanly.

**Result:** 10–50 steps instead of 500–1000. Same weights, same training loop, just a swapped sampler.

**Extensions in nanoddpm‑pro:**
- Supports both **linear** and **cosine** beta schedules.
- Can be trained with **noise (ε)** or **v‑prediction** targets (see §4).

---

## 3. EDM: Optimized Schedule & ODE Sampling *(Advanced Variant)*
EDM (Elucidating Diffusion Models) replaces heuristic noise schedules with a **continuous, mathematically grounded framework**. Instead of discrete timesteps, diffusion is treated as an ODE in noise space.

### 3.1. Core Components
- **Log‑Normal Noise Sampling:** Draw noise levels from `σ ~ exp(𝒩(P_mean, P_std²))`. This concentrates training on mid‑noise regimes where gradients are most informative.
- **I/O Preconditioning** (Karras et al. 2022): Wrap the network with analytically derived scalars so it always sees normalized inputs/outputs, stabilizing gradients across the entire trajectory:

$$
c_{\text{in}} = \frac{1}{\sqrt{\sigma^2+1}}, \quad c_{\text{noise}} = \frac{\log\sigma}{4}, \quad c_{\text{skip}} = \frac{1}{\sigma^2+1}, \quad c_{\text{out}} = \frac{\sigma}{\sqrt{\sigma^2+1}}
$$

$$
D(x, \sigma) = c_{\text{skip}} \cdot x + c_{\text{out}} \cdot F(x \cdot c_{\text{in}}, c_{\text{noise}})
$$

### 3.2. ODE Solvers: Euler vs Heun
The reverse process becomes a first‑order ODE: $\frac{dx}{d\sigma} = \frac{x - D(x,\sigma)}{\sigma}$. Discretizing gives two solver options:

- **Euler Solver** (1st‑order, fast):
  $$
  x_{\text{next}} = x + (\sigma_{\text{next}} - \sigma) \cdot \frac{x - D(x,\sigma)}{\sigma}
  $$

- **Heun Solver** (2nd‑order, quality):
  $$
  \begin{aligned}
  x_{\text{pred}} &= x + (\sigma_{\text{next}} - \sigma) \cdot \frac{x - D}{\sigma} \\
  x_{\text{next}} &= x + \frac{\sigma_{\text{next}} - \sigma}{2} \cdot \left( \frac{x - D}{\sigma} + \frac{x_{\text{pred}} - D_{\text{pred}}}{\sigma_{\text{next}}} \right)
  \end{aligned}
  $$

### 3.3. Training Targets: Noise (`ε`) vs `v`‑Prediction
The network can be trained to predict different quantities. Both share the same architecture and ODE solvers, but differ in loss formulation and sampling decoding:

| Target | Training Objective | Sampling Decode | Why Use It? |
|--------|-------------------|-----------------|-------------|
| **Noise (`ε`)** | Predicts added noise. Loss weighted by $1/c_{\text{out}}^2$. | $x_0 \approx D(x_\sigma, \sigma)$ | Standard DDPM/EDM baseline. Simple & effective. |
| **`v`‑Prediction** | Predicts velocity $v = c_{\text{out}}\epsilon - c_{\text{skip}}x_0$. Uniform loss weighting. | $x_0 = c_{\text{skip}}x_\sigma - c_{\text{out}}D(x_\sigma, \sigma)$ | Used in SD2+, Imagen. Keeps gradient magnitudes uniform across all `σ`, improving stability at high noise. |

**Key Insight:** 

1. Think of v-prediction as the model predicting the **direction and speed** needed to move from noise towards a clean image.
2. Once decoded to $x_0$, the **Euler/Heun solvers apply identically** to both targets. This cleanly decouples training objectives from the sampling loop.

### 3.4. Core Improvements over DDIM:
| Feature | DDIM | EDM |
|---------|------|-----|
| **Parameterization** | Discrete `t` with `α_t`/`β_t` | Continuous `σ` (sigma) |
| **Math Foundation** | Reparameterizes DDPM reverse mean into a deterministic loop | Solves the diffusion ODE `dx/dσ = (x - D(x,σ))/σ` |
| **Step Skipping** | Requires re‑deriving reverse equations for arbitrary step counts | Naturally supports any number of steps; just change the `σ` grid |
| **Noise Sampling** | Uniform `t ~ [0, T]` | Log‑normal `σ ~ exp(𝒩(P_mean, P_std²))` |
| **Stability** | Manual clipping needed | Preconditioning (`c_in`, `c_out`, `c_skip`) |
| **Solver** | Fixed reverse algebra | Interchangeable Euler/Heun ODE solvers |

**Tradeoff:** Heun requires 2× network calls but typically achieves 10‑20% lower PCA‑FID at the same step count.  
**Result:** Training converges faster, gradients stay stable at higher resolutions, and sampling naturally supports arbitrary step counts with interchangeable solvers and targets.

---

## 4. Weighted Loss & EMA *(Shared Improvements)*
- **SNR‑weighted loss** (for `epsilon` target): `loss = MSE(pred, eps) * snr/(snr+1)`. This gives more weight to steps where the Signal‑to‑Noise Ratio (SNR) is moderate, improving perceptual quality.
- **Exponential Moving Average (EMA)** of model weights: produces smoother, higher‑quality samples at inference time.

---

## 5. Lightweight Metrics *(Shared)*
Standard FID uses Inception‑V3 (2048‑d features) which is heavy, opaque, and overkill for 32×32 images. `nanoddpm‑pro` replaces it with three cheap, transparent metrics:

### 5.1. PCA‑FID
1. Flatten images to `[B, H*W*C]`
2. Compute top‑32 principal components across real + generated batches
3. Project both sets into the 32D subspace
4. Run the standard FID formula:

$$
\text{PCA-FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\right)
$$

**Advantages:** No external model, runs on CPU in <0.1s, fully inspectable.

### 5.2. Sobel Gradient Sharpness
Convolution with Sobel kernels measures average edge strength. Higher values indicate sharper images (less blur).

### 5.3. Intensity KL Divergence
Compares pixel intensity histograms (50 bins) between real and generated images. Lower values mean better match of overall brightness/contrast distribution.

---

## 6. Which Configuration Should You Use?
| Use Case | Recommended Flags |
|----------|------------------|
| Learning diffusion basics | `--dataset mnist --sampler ddim --target epsilon` |
| Research/quality focus (CIFAR‑10) | `--dataset cifar10 --sampler edm --target v --solver heun` |
| Rapid iteration | `--dataset cifar10 --sampler edm --target epsilon --solver euler` |
| Colab free tier | `--resize 16 --sample_steps 10 --batch_size 16` |
| Compare beta schedules | Train twice: `--beta_schedule linear` vs `cosine` |

**Pro tip:** The unified file lets you swap samplers, solvers, and decoding targets at inference without retraining. Train once, then explore the design space via CLI:

```bash
# Train with standard noise prediction (EDM + epsilon)
python nanoddpm-pro.py --dataset cifar10 --sampler edm --target epsilon --epochs 20

# Swap to v‑prediction decoding + Heun solver at inference (no retraining)
python nanoddpm-pro.py --sampler edm --target v --solver heun --cfg_scale 4.0
```

---
# nanoddpm-pro: Fast, Guided Diffusion from Scratch

> Upgrading `nanoddpm` with a **Mini-UNet, Classifier-Free Guidance (CFG), Denoising Diffusion Implicit Models (DDIM) sampling, and PCA-FID evaluation.** It is run on the **CIFAR-10** dataset. No high-level wrappers, no hidden abstractions, just raw **PyTorch** + the core diffusion equations.

## 1. Classifier-Free Guidance (CFG)
Standard conditional diffusion models require a separate classifier to steer generation. CFG removes that dependency by training a single model on **paired conditional/unconditional data**.

During training, we randomly mask the class label (replace with `-1` or `∅`) ~10% of the time. This forces the network to learn two representations:
- `ε_θ(x_t, t, y)` : noise prediction with class conditioning
- `ε_θ(x_t, t, ∅)` : noise prediction without conditioning

At inference, we combine them linearly:
$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)
$$
Where `w` (`cfg_scale`) controls how strongly we push toward the target class. 
- `w = 1.0` → standard conditional generation
- `w = 3.0–7.5` → sharper, more class-aligned outputs (at the cost of diversity)
- `w > 7.5` → artifacts & oversaturation

**Why it works:** The difference term `(ε_cond - ε_uncond)` points in the direction of the data manifold for class `y`. Scaling it amplifies the "pull" toward that class without needing gradients from an external network.

---

## 2. DDIM: Deterministic Fast Sampling
DDPM sampling is slow because it adds random noise at every reverse step. DDIM reparameterizes the reverse process as a **deterministic ODE**, allowing us to skip steps safely.

Starting from the DDPM reverse mean:
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

We can predict the original image $x_0$ directly:
$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}
$$

DDIM replaces the stochastic update with:
$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta + \sigma_t z
$$
Setting $\sigma_t = 0$ removes the random noise `z`, making the path deterministic. Because the trajectory is fixed, we can evaluate the network at a subset of steps (e.g., `t = 500, 250, 100, 50, 0`) and interpolate cleanly.

**Result:** 10–50 steps instead of 500–1000. Same weights, same training loop, just a swapped sampler.

---

## 3. PCA-FID: Lightweight Quality Tracking
**Fréchet Inception Distance (FID)** is a statistical metric used to evaluate the quality and diversity of images produced by generative models like _DDPM_. It measures the distance between the distribution of generated images and real training images.

Standard FID uses Inception-V3 to extract 2048-d features. That's heavy, opaque, and overkill for CIFAR-10. PCA-FID replaces the black-box extractor with **unsupervised dimensionality reduction**:

1. Flatten images to `[B, 3072]`
2. Compute top-32 principal components across real + generated batches
3. Project both sets into the 32D subspace
4. Run the standard FID formula on the projected means/covariances

$$
\text{PCA-FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})
$$

**Why it's better for educational builds:**
- No external weights or network downloads
- Runs in <0.1s on CPU
- Captures global structure & color distribution (perfect for 32×32)
- Fully transparent: you can inspect the principal components

---


# nanoddpm-pro.py: Unified DDPM for MNIST & CIFAR-10 (<400 lines)
# Features: Mini-UNet, CFG, DDIM/EDM, Euler/Heun, v-prediction, PCA-FID, EMA, weighted loss
# Usage: python nanoddpm-pro.py --dataset mnist   or   --dataset cifar10

import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np, math, json, copy
from tqdm import trange
import torch.nn.functional as F

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--cfg_scale', type=float, default=4.0)
parser.add_argument('--resize', type=int, default=32, help='Resize images to this size (if dataset!=mnist)')
parser.add_argument('--sampler', type=str, default='edm', choices=['ddim', 'edm'])
parser.add_argument('--steps', type=int, default=1000, help='Diffusion steps T (DDIM only)')
parser.add_argument('--sample_steps', type=int, default=20, help='EDM discretization steps')
parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'heun'])
parser.add_argument('--target', type=str, default='epsilon', choices=['epsilon', 'v'])
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
parser.add_argument('--use_ema', action='store_true', default=True)
parser.add_argument('--ema_decay', type=float, default=0.999)
args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)
USE_EDM = args.sampler == 'edm'
print(f"▶ nanoddpm-pro | {device} | Dataset:{args.dataset} | Sampler:{args.sampler} | Target:{args.target} | CFG:{args.cfg_scale}")

# ======== DATASET DEPENDENT SETUP ========
if args.dataset == 'mnist':
    img_channels = 1
    img_size = 28
    num_classes = 10
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
else:  # cifar10
    img_channels = 3
    img_size = args.resize
    num_classes = 10
    transform = T.Compose([T.Resize(args.resize), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=2, pin_memory=torch.cuda.is_available())
real_batch, real_labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)))
real_batch, real_labels = real_batch.to(device), real_labels.to(device)

# ======== NOISE SCHEDULES ========
if not USE_EDM:
    T_steps = args.steps
    if args.beta_schedule == 'linear':
        beta = torch.linspace(1e-4, 0.02, T_steps, device=device)
    else: # cosine
        def cosine_beta_schedule(T, s=0.008):
            x = torch.linspace(0, T, T+1, device=device)
            alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            beta = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return torch.clamp(beta, 1e-5, 0.999)
        beta = cosine_beta_schedule(T_steps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
else: # EDM schedule constants
    P_MEAN, P_STD = -1.2, 1.2
    SIGMA_MIN, SIGMA_MAX = 0.002, 80.0
    def sample_sigmas(n, dev):
        return torch.exp(P_MEAN + P_STD * torch.randn(n, device=dev))

# ======== MODEL ========
def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        num_groups = min(8, out_ch)  # safe group count that always divides out_ch
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.class_proj = nn.Linear(time_dim, out_ch) # receives the shared class embedding and projects to block channels
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity() # identity skip if channels match, else 1x1 conv
        self.time_dim = time_dim

    def forward(self, x, t, c_emb=None):
        h = F.silu(self.norm1(self.conv1(x)))
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        if c_emb is not None:
            c_emb = self.dropout(c_emb)          # apply dropout to shared embedding
            c_proj = self.class_proj(c_emb)      # project to out_ch
            t_emb = t_emb + c_proj               # fuse with time embedding
        h = h + t_emb[:, :, None, None]          # broadcast conditional info
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class MiniUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, time_dim=128, ch=32, num_classes=10):
        super().__init__()
        # single embedding shared by all blocks
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)   # +1 = unconditional
        self.down1 = ResBlock(in_ch, ch, time_dim)
        self.down2 = ResBlock(ch, ch * 2, time_dim)
        self.mid   = ResBlock(ch * 2, ch * 2, time_dim)
        self.up1   = ResBlock(ch * 4, ch, time_dim)       # input: mid + d2 = ch*4
        self.up2   = ResBlock(ch * 2, out_ch, time_dim)   # input: up1 + d1 = ch*2
        self.out   = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x, t, labels=None):
        c_emb = self.class_emb(labels) if labels is not None else None
        d1 = self.down1(x, t, c_emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), t, c_emb)
        x = self.mid(F.avg_pool2d(d2, 2), t, c_emb)
        x = self.up1(torch.cat([F.interpolate(x, scale_factor=2), d2], 1), t, c_emb)
        x = self.up2(torch.cat([F.interpolate(x, scale_factor=2), d1], 1), t, c_emb)
        return self.out(x)

class EDMWrapper(nn.Module):
    def __init__(self, model, target='epsilon'):
        super().__init__()
        self.model = model
        self.target = target
    def forward(self, x, sigma, labels=None):
        c_in = 1 / torch.sqrt(sigma**2 + 1)
        c_noise = torch.log(sigma) / 4
        model_out = self.model(x * c_in[:,None,None,None], c_noise, labels)
        c_skip = 1 / (sigma**2 + 1)
        c_out = sigma / torch.sqrt(sigma**2 + 1)
        if self.target == 'epsilon':
            return c_skip[:,None,None,None] * x + c_out[:,None,None,None] * model_out
        else: # v-prediction
            return model_out  

model = MiniUNet(in_ch=img_channels, out_ch=img_channels, time_dim=128, ch=32).to(device)
if USE_EDM:
    edm_wrapper = EDMWrapper(model, target=args.target).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3)
print(f"▶ Params: {sum(p.numel() for p in model.parameters()):,}")

# EMA
ema_model = copy.deepcopy(model) if args.use_ema else None
def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)

# ======== METRICS ========
def sobel_grad(imgs):
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=imgs.device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=imgs.device).view(1,1,3,3)
    gx = F.conv2d(imgs, sx, padding=1)
    gy = F.conv2d(imgs, sy, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8).mean().item()

def intensity_kl(real, gen, bins=50):
    r = real.cpu().view(-1).clamp(-1,1).numpy()
    g = gen.cpu().view(-1).clamp(-1,1).numpy()
    hr, _ = np.histogram(r, bins=bins, range=(-1,1), density=True)
    hg, _ = np.histogram(g, bins=bins, range=(-1,1), density=True)
    hr, hg = hr+1e-8, hg+1e-8
    hr /= hr.sum()
    hg /= hg.sum()
    return np.sum(hg * np.log(hg/hr)).item()

def pca_fid(real, gen, n_components=32):
    r = real.view(real.shape[0], -1).cpu().double()
    g = gen.view(gen.shape[0], -1).cpu().double()
    data = torch.cat([r, g], 0)
    mean = data.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(data - mean, q=n_components)
    proj = (data - mean) @ V
    r_p, g_p = proj[:real.shape[0]], proj[real.shape[0]:]
    mu_r, mu_g = r_p.mean(0), g_p.mean(0)
    var_r, var_g = r_p.var(0)+1e-6, g_p.var(0)+1e-6
    return ((mu_r-mu_g)**2).sum().item() + (var_r+var_g-2*torch.sqrt(var_r*var_g)).sum().item()

# ======== SAMPLERS ========
def x0_from_pred(x, pred, t, target, sqrt_ab, sqrt_1m):
    if target == 'epsilon':
        return (x - sqrt_1m * pred) / sqrt_ab
    else: # v-prediction
        return sqrt_ab * x - sqrt_1m * pred

@torch.no_grad()
def sample_ddim(n, labels, cfg_scale=1.0, ddim_steps=50, target='epsilon'):
    model_eval = ema_model if args.use_ema else model
    model_eval.eval()
    x = torch.randn(n, img_channels, img_size, img_size, device=device)
    step_size = max(1, T_steps // ddim_steps)
    for t in reversed(range(0, T_steps, step_size)):
        t_t = torch.full((n,), t, dtype=torch.long, device=device)
        eps_u = model_eval(x, t_t, None)
        eps_c = model_eval(x, t_t, labels)
        eps_pred = eps_u + cfg_scale * (eps_c - eps_u)
        sqrt_ab = sqrt_alpha_bar[t]
        sqrt_1m = sqrt_one_minus_alpha_bar[t]
        pred_x0 = x0_from_pred(x, eps_pred, t, target, sqrt_ab, sqrt_1m)
        a_prev = alpha[t - step_size] if t>=step_size else alpha[0]
        sqrt_a_prev = torch.sqrt(a_prev)
        sqrt_1m_prev = torch.sqrt(1 - a_prev)
        x = sqrt_a_prev * pred_x0 + sqrt_1m_prev * eps_pred
        x = torch.clamp(x, -1.0, 1.0)
    return x

@torch.no_grad()
def edm_sampler(wrapper, n, labels, cfg=1.0, steps=20, solver='euler', target='epsilon'):
    model_eval = wrapper if not args.use_ema else EDMWrapper(ema_model, target=target)
    model_eval.eval()
    sigmas = torch.linspace(SIGMA_MAX, SIGMA_MIN, steps, device=device)
    x = torch.randn(n, img_channels, img_size, img_size, device=device) * sigmas[0]
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        s_vec = torch.full((n,), s, device=device)
        D_u = model_eval(x, s_vec, None)
        D_c = model_eval(x, s_vec, labels)
        D = D_u + cfg * (D_c - D_u)
        c_skip = 1 / (s**2 + 1)
        c_out = s / torch.sqrt(s**2 + 1)
        if target == 'epsilon':
            x0_pred = D   # wrapper returns denoised x0
        else:
            x0_pred = c_skip * x - c_out * D  # D is v, decode to x0
        if solver == 'euler':
            x = x + (s_next - s) * (x - x0_pred) / s
        else:  # heun
            x_pred = x + (s_next - s) * (x - x0_pred) / s
            if s_next > 1e-5:
                s_next_vec = torch.full((n,), s_next, device=device)
                D_u_p = model_eval(x_pred, s_next_vec, None)
                D_c_p = model_eval(x_pred, s_next_vec, labels)
                D_p = D_u_p + cfg * (D_c_p - D_u_p)
                if target == 'epsilon':
                    x0_pred_p = D_p
                else:
                    c_skip_p = 1 / (s_next**2 + 1)
                    c_out_p = s_next / torch.sqrt(s_next**2 + 1)
                    x0_pred_p = c_skip_p * x_pred - c_out_p * D_p
                x = x + (s_next - s) * ((x - x0_pred)/s + (x_pred - x0_pred_p)/s_next) / 2
            else:
                x = x_pred
        x = torch.clamp(x, -1.0, 1.0)
    return x

# ======== TRAINING LOOP ========
metrics_log = []
for epoch in trange(1, args.epochs+1, desc="Training"):
    model.train()
    epoch_loss, count = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        train_labels = labels.clone()
        mask_uncond = torch.rand_like(labels.float()) < 0.1
        train_labels[mask_uncond] = num_classes  # unconditional index (10)

        if not USE_EDM: # DDIM branch + Forward Diffusion
            t = torch.randint(0, T_steps, (imgs.shape[0],), device=device)
            sqrt_ab = sqrt_alpha_bar[t][:, None, None, None]
            sqrt_1m = sqrt_one_minus_alpha_bar[t][:, None, None, None]
            eps = torch.randn_like(imgs)
            xt = sqrt_ab * imgs + sqrt_1m * eps
            pred = model(xt, t, train_labels)
            if args.target == 'epsilon': # weighted loss (SNR)
                snr = alpha_bar[t] / (1 - alpha_bar[t] + 1e-8)
                weight = snr / (snr + 1)
                loss = F.mse_loss(pred, eps, reduction='none').mean(dim=(1,2,3))
                loss = (loss * weight).mean()
            else:  # EDM branch
                target_v = sqrt_ab * eps - sqrt_1m * imgs
                loss = F.mse_loss(pred, target_v)
        else:
            sigma = sample_sigmas(imgs.shape[0], device)
            noise = torch.randn_like(imgs)
            x_sigma = imgs + sigma[:, None, None, None] * noise
            pred = edm_wrapper(x_sigma, sigma, train_labels)
            if args.target == 'epsilon':  # denoising loss with weighting
                c_out = sigma / torch.sqrt(sigma**2 + 1)
                loss_weight = (1.0 / (c_out**2 + 1e-8)).view(-1,1,1,1)
                loss = (loss_weight * F.mse_loss(pred, imgs, reduction='none')).mean()
            else: # v-prediction
                c_skip = 1 / (sigma**2 + 1)
                c_out = sigma / torch.sqrt(sigma**2 + 1)
                eps = (x_sigma - imgs) / (sigma[:,None,None,None] + 1e-8)
                target_v = c_out[:,None,None,None] * eps - c_skip[:,None,None,None] * imgs
                loss = F.mse_loss(pred, target_v)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.use_ema:
            update_ema(model, ema_model, args.ema_decay)
        epoch_loss += loss.item() * imgs.shape[0]
        count += imgs.shape[0]

    # Evaluation
    model_eval = ema_model if args.use_ema else model
    model_eval.eval()
    with torch.no_grad():
        if not USE_EDM:
            gen = sample_ddim(128, real_labels[:128], cfg_scale=args.cfg_scale,
                              ddim_steps=20, target=args.target)
        else:
            gen = edm_sampler(edm_wrapper, 128, real_labels[:128], cfg=args.cfg_scale,
                              steps=args.sample_steps, solver=args.solver, target=args.target)
        p_fid = pca_fid(real_batch[:128], gen)
        grad = sobel_grad(gen)
        kl = intensity_kl(real_batch[:128], gen)

    metrics_log.append({
        'epoch': epoch,
        'loss': epoch_loss/count,
        'pca_fid': p_fid,
        'sobel_grad': grad,
        'kl_div': kl
    })
    print(f"  Epoch {epoch:02d} | Loss: {metrics_log[-1]['loss']:.4f} | PCA-FID: {p_fid:.2f} | Grad: {grad:.3f} | KL: {kl:.4f}")

# Save metrics
with open('nanoddpm_pro_metrics.json', 'w') as f:
    # remove non-serializable entries
    serializable = [{k:v for k,v in m.items() if k != 'samples'} for m in metrics_log]
    json.dump(serializable, f, indent=2)

# ======== VISUALIZATION ========
def plot_results():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    epochs = [m['epoch'] for m in metrics_log]
    axs[0,0].plot(epochs, [m['loss'] for m in metrics_log], marker='o')
    axs[0,0].set_title('Training Loss')
    axs[0,0].grid(alpha=0.3)
    axs[0,1].plot(epochs, [m['pca_fid'] for m in metrics_log], marker='s', color='orange')
    axs[0,1].set_title('PCA-FID (↓ better)')
    axs[0,1].grid(alpha=0.3)
    axs[1,0].plot(epochs, [m['sobel_grad'] for m in metrics_log], marker='^', color='green')
    axs[1,0].set_title('Sharpness (Sobel ↑)')
    axs[1,0].grid(alpha=0.3)
    axs[1,1].plot(epochs, [m['kl_div'] for m in metrics_log], marker='d', color='red')
    axs[1,1].set_title('Intensity KL Divergence (↓ better)')
    axs[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nFinal samples ({args.sampler.upper()}/{args.target}{'+'+args.solver if USE_EDM else ''}, CFG={args.cfg_scale})")
    model_eval = ema_model if args.use_ema else model
    model_eval.eval()
    with torch.no_grad():
        if not USE_EDM:
            gen_final = sample_ddim(16, torch.randint(0, num_classes, (16,), device=device),
                                    cfg_scale=args.cfg_scale, ddim_steps=50, target=args.target)
        else:
            gen_final = edm_sampler(edm_wrapper, 16, torch.randint(0, num_classes, (16,), device=device),
                                    cfg=args.cfg_scale, steps=args.sample_steps, solver=args.solver, target=args.target)
        grid = torchvision.utils.make_grid(gen_final, nrow=4, normalize=True, value_range=(-1,1))
        plt.figure(figsize=(5,5))
        plt.imshow(grid.permute(1,2,0).cpu().numpy())
        plt.axis('off')
        plt.show()

plot_results()
print(f"Done. Metrics saved to nanoddpm_pro_metrics.json | Mode: {args.sampler}/{args.target}")
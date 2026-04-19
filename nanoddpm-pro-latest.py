# nanoddpm-pro.py: Unified DDPM for CIFAR-10 (~280 lines)
# Features: Mini-UNet, Classifier-Free Guidance, DDIM/EDM samplers, Euler/Heun solvers, v-prediction, PCA-FID
# Educational build inspired by micrograd/minbpe/nanoddpm

import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np, math, json
from tqdm import trange
import torch.nn.functional as F

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--cfg_scale', type=float, default=4.0)
parser.add_argument('--resize', type=int, default=32)
parser.add_argument('--sampler', type=str, default='edm', choices=['ddim', 'edm'], help='Sampling framework')
parser.add_argument('--steps', type=int, default=1000, help='DDIM diffusion timesteps T')
parser.add_argument('--sample_steps', type=int, default=20, help='EDM solver discretization steps')
parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'heun'], help='ODE solver (EDM only)')
parser.add_argument('--target', type=str, default='epsilon', choices=['epsilon', 'v'], help='Training target: noise (epsilon) or v-prediction')
args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)
USE_EDM = args.sampler == 'edm'
print(f"▶ nanoddpm-pro | {device} | Sampler: {args.sampler} | Target: {args.target} | CFG: {args.cfg_scale} | Resize: {args.resize}")

# === NOISE SCHEDULES ===
if not USE_EDM:
    T_steps = args.steps
    beta = torch.linspace(1e-4, 0.02, T_steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
else:
    P_MEAN, P_STD = -1.2, 1.2
    SIGMA_MIN, SIGMA_MAX = 0.002, 80.0
    def sample_sigmas(n, dev):
        return torch.exp(P_MEAN + P_STD * torch.randn(n, device=dev))

# === DATASET (CIFAR-10) ===
transform = T.Compose([T.Resize(args.resize), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
real_batch, real_labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)))
real_batch = real_batch.to(device)

# === MODEL (Mini-UNet + Time/Class Embedding) ===
def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_classes=10, dropout=0.1):
        super().__init__()
        self.num_groups_gn = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(self.num_groups_gn, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(self.num_groups_gn, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.class_emb = nn.Embedding(num_classes, time_dim)
        self.class_proj = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_dim = time_dim

    def forward(self, x, t, labels=None):
        h = F.silu(self.norm1(self.conv1(x)))
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        if labels is None:
            c_emb = torch.zeros_like(t_emb)
        else:
            c_emb = self.class_proj(self.dropout(self.class_emb(labels)))
        h = h + (t_emb + c_emb)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h))) + self.skip(x)

class MiniUNet(nn.Module):
    def __init__(self, time_dim=128, ch=32):
        super().__init__()
        self.down1 = ResBlock(3, ch, time_dim)
        self.down2 = ResBlock(ch, ch*2, time_dim)
        self.mid = ResBlock(ch*2, ch*2, time_dim)
        self.up1 = ResBlock(ch*4, ch, time_dim)
        self.up2 = ResBlock(ch*2, 3, time_dim)
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, x, t, labels=None):
        d1 = self.down1(x, t, labels)
        d2 = self.down2(F.avg_pool2d(d1, 2), t, labels)
        x = self.mid(F.avg_pool2d(d2, 2), t, labels)
        x = self.up1(torch.cat([F.interpolate(x, scale_factor=2), d2], 1), t, labels)
        x = self.up2(torch.cat([F.interpolate(x, scale_factor=2), d1], 1), t, labels)
        return self.out(x)

# === EDM PRECONDITIONING WRAPPER ===
class EDMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, sigma, labels=None):
        c_in = 1 / torch.sqrt(sigma**2 + 1)
        c_noise = torch.log(sigma) / 4
        out = self.model(x * c_in[:,None,None,None], c_noise, labels)
        c_skip, c_out = 1/(sigma**2+1), sigma/torch.sqrt(sigma**2+1)
        return c_skip[:,None,None,None]*x + c_out[:,None,None,None]*out

model = MiniUNet(time_dim=128, ch=32).to(device)
edm_wrapper = EDMWrapper(model).to(device) if USE_EDM else None
optimizer = optim.Adam(model.parameters(), lr=2e-3)
print(f"▶ Params: {sum(p.numel() for p in model.parameters()):,}")

# === PCA-FID ===
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

# === SAMPLERS ===
def forward_diffusion(x0, t):
    sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_1m = torch.sqrt(1.0 - alpha_bar[t])[:, None, None, None]
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1m * eps, eps

@torch.no_grad()
def sample_ddim(n, labels, cfg_scale=1.0, ddim_steps=50, target='epsilon'):
    model.eval()
    x = torch.randn(n, 3, args.resize, args.resize, device=device)
    step_size = max(1, T_steps // ddim_steps)
    for t in reversed(range(0, T_steps, step_size)):
        t_t = torch.full((n,), t, dtype=torch.long, device=device)
        eps_u = model(x, t_t, None)
        eps_c = model(x, t_t, labels)
        eps_pred = eps_u + cfg_scale * (eps_c - eps_u)
        a, ab = alpha[t], alpha_bar[t]
        
        if target == 'epsilon':
            # Standard: eps_pred is noise
            pred_x0 = (x - torch.sqrt(1 - ab)*eps_pred) / torch.sqrt(ab)
        else:
            # v-prediction: eps_pred is v, decode to x0
            pred_x0 = torch.sqrt(ab) * x - torch.sqrt(1.0 - ab) * eps_pred
        
        a_prev = alpha[t - step_size] if t>=step_size else alpha[0]
        x = torch.sqrt(a_prev)*pred_x0 + torch.sqrt(1 - a_prev)*eps_pred
        x = torch.clip(x, -1.0, 1.0)
    return x

@torch.no_grad()
def edm_sampler(wrapper, n, labels, cfg=1.0, steps=20, solver='euler', target='epsilon'):
    sigmas = torch.linspace(SIGMA_MAX, SIGMA_MIN, steps, device=device)
    x = torch.randn(n, 3, args.resize, args.resize, device=device) * sigmas[0]
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        s_vec = torch.full((n,), s, device=device)
        D_u = wrapper(x, s_vec, None)
        D_c = wrapper(x, s_vec, labels)
        D = D_u + cfg * (D_c - D_u)
        
        # Decode prediction to x0 based on target
        if target == 'epsilon':
            x0_pred = D  # EDM wrapper outputs x0 directly
        else:
            # v-prediction: decode v to x0: x0 = c_skip*x - c_out*v
            c_skip = 1 / (s**2 + 1)
            c_out = s / torch.sqrt(s**2 + 1)
            x0_pred = c_skip * x - c_out * D
        
        if solver == 'euler':
            x = x + (s_next - s) * (x - x0_pred) / s
        else:  # heun
            x_pred = x + (s_next - s) * (x - x0_pred) / s
            if s_next > 1e-5:
                s_next_vec = torch.full((n,), s_next, device=device)
                D_u_p = wrapper(x_pred, s_next_vec, None)
                D_c_p = wrapper(x_pred, s_next_vec, labels)
                D_p = D_u_p + cfg * (D_c_p - D_u_p)
                
                if target == 'epsilon':
                    x0_pred_p = D_p
                else:
                    c_skip_p = 1 / (s_next**2 + 1)
                    c_out_p = s_next / torch.sqrt(s_next**2 + 1)
                    x0_pred_p = c_skip_p * x_pred - c_out_p * D_p
                
                x = x + (s_next - s) * ((x - x0_pred) / s + (x_pred - x0_pred_p) / s_next) / 2
            else:
                x = x_pred
        x = torch.clip(x, -1.0, 1.0)
    return x

# === TRAINING LOOP ===
metrics_log = []
for epoch in trange(1, args.epochs+1, desc="Training"):
    model.train()
    epoch_loss, count = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        train_labels = labels.clone()
        train_labels[torch.rand_like(labels.float()) < 0.1] = -1  # 10% unconditional
        cond_mask = train_labels != -1
        uncond_mask = ~cond_mask

        if not USE_EDM:
            # DDIM branch: discrete t schedule
            t = torch.randint(0, T_steps, (imgs.shape[0],), device=device)
            xt, eps = forward_diffusion(imgs, t)
            pred = torch.zeros_like(eps if args.target == 'epsilon' else imgs)
            
            if cond_mask.any(): 
                pred[cond_mask] = model(xt[cond_mask], t[cond_mask], train_labels[cond_mask])
            if uncond_mask.any(): 
                pred[uncond_mask] = model(xt[uncond_mask], t[uncond_mask], None)
            
            if args.target == 'epsilon':
                # Standard: predict noise
                loss = F.mse_loss(pred, eps)
            else:
                # v-prediction: target = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*imgs
                sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
                sqrt_1m = torch.sqrt(1.0 - alpha_bar[t])[:, None, None, None]
                target_v = sqrt_ab * eps - sqrt_1m * imgs
                loss = F.mse_loss(pred, target_v)
        else:
            # EDM branch: continuous sigma schedule
            sigma = sample_sigmas(imgs.shape[0], device)
            x_sigma = imgs + sigma[:,None,None,None] * torch.randn_like(imgs)
            pred = torch.zeros_like(x_sigma)
            
            if cond_mask.any(): 
                pred[cond_mask] = edm_wrapper(x_sigma[cond_mask], sigma[cond_mask], train_labels[cond_mask])
            if uncond_mask.any(): 
                pred[uncond_mask] = edm_wrapper(x_sigma[uncond_mask], sigma[uncond_mask], None)
            
            if args.target == 'epsilon':
                # Standard EDM: predict x0, weight by 1/c_out^2
                c_out = sigma / torch.sqrt(sigma**2 + 1)
                loss_weight = (1.0 / (c_out ** 2)).view(-1, 1, 1, 1)
                loss = (loss_weight * F.mse_loss(pred, imgs, reduction='none')).mean()
            else:
                # v-prediction for EDM: target = c_out*eps - c_skip*x0
                c_skip = 1 / (sigma**2 + 1)
                c_out = sigma / torch.sqrt(sigma**2 + 1)
                eps = (x_sigma - imgs) / sigma[:, None, None, None]
                target_v = c_out[:, None, None, None] * eps - c_skip[:, None, None, None] * imgs
                # Uniform weighting for v-prediction (standard practice)
                loss = F.mse_loss(pred, target_v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.shape[0]
        count += imgs.shape[0]

    # Evaluation: sample with current target
    model.eval()
    with torch.no_grad():
        if not USE_EDM:
            gen = sample_ddim(128, real_labels[:128].to(device), cfg_scale=args.cfg_scale, ddim_steps=20, target=args.target)
        else:
            gen = edm_sampler(edm_wrapper, 128, real_labels[:128].to(device), cfg=args.cfg_scale, steps=args.sample_steps, solver=args.solver, target=args.target)
        fid = pca_fid(real_batch[:128], gen)
    metrics_log.append({'epoch': epoch, 'loss': epoch_loss/count, 'pca_fid': fid})
    print(f"  Epoch {epoch:02d} | Loss: {metrics_log[-1]['loss']:.4f} | PCA-FID ({args.target}): {fid:.2f}")

with open('nanoddpm_pro_metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=2)

# === VISUALIZATION ===
def plot_results():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].plot([m['epoch'] for m in metrics_log], [m['loss'] for m in metrics_log], marker='o')
    axs[0].set_title('Training Loss')
    axs[0].grid(alpha=0.3)
    axs[1].plot([m['epoch'] for m in metrics_log], [m['pca_fid'] for m in metrics_log], marker='s', color='orange')
    axs[1].set_title(f'PCA-FID ({args.sampler.upper()}/{args.target}) ↓ better')
    axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal samples ({args.sampler.upper()}/{args.target}{'+'+args.solver if USE_EDM else ''}, CFG={args.cfg_scale}):")
    model.eval()
    with torch.no_grad():
        if not USE_EDM:
            gen_final = sample_ddim(16, torch.randint(0, 10, (16,), device=device), cfg_scale=args.cfg_scale, ddim_steps=50, target=args.target)
        else:
            gen_final = edm_sampler(edm_wrapper, 16, torch.randint(0, 10, (16,), device=device), cfg=args.cfg_scale, steps=args.sample_steps, solver=args.solver, target=args.target)
        grid = torchvision.utils.make_grid(gen_final, nrow=4, normalize=True, value_range=(-1,1))
        plt.figure(figsize=(5,5))
        plt.imshow(grid.permute(1,2,0).numpy())
        plt.axis('off')
        plt.show()

plot_results()
print(f"Done. Metrics saved to nanoddpm_pro_metrics.json | Mode: {args.sampler}/{args.target}")
# nanoddpm_pro.py: From-scratch EDM for CIFAR-10 (~220 lines)
# Features: Mini-UNet, Classifier-Free Guidance, EDM framework, PCA-FID
# Educational build inspired by micrograd/minbpe/nanoddpm

import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np, math, json
from tqdm import trange
import torch.nn.functional as F

# === CONFIG (EDM-Aligned) ===
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--cfg_scale', type=float, default=4.0)
parser.add_argument('--resize', type=int, default=32)
parser.add_argument('--sample_steps', type=int, default=20, help='EDM Euler sampler steps')
args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)
print(f"▶ nanoddpm-pro (EDM) | {device} | Res: {args.resize} | CFG: {args.cfg_scale}")

# === EDM NOISE SCHEDULE ===
P_MEAN, P_STD = -1.2, 1.2          # Log-normal sampling params
SIGMA_MIN, SIGMA_MAX = 0.002, 80.0 # Noise range for sampling

def sample_sigmas(n, device):
    """Log-normal noise level sampling (EDM Eq. 5)"""
    return torch.exp(P_MEAN + P_STD * torch.randn(n, device=device))

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
        
        # Project 1D c_noise → time_dim → out_ch
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.class_emb = nn.Embedding(num_classes, time_dim)
        self.class_proj = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_dim = time_dim

    def forward(self, x, c_noise, labels=None):
        h = F.silu(self.norm1(self.conv1(x)))
        t_emb = self.time_mlp(c_noise.unsqueeze(-1)) # Unsqueeze to [B, 1] before MLP
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

    def forward(self, x, c_noise, labels=None):
        d1 = self.down1(x, c_noise, labels)
        d2 = self.down2(F.avg_pool2d(d1, 2), c_noise, labels)
        x = self.mid(F.avg_pool2d(d2, 2), c_noise, labels)
        x = self.up1(torch.cat([F.interpolate(x, scale_factor=2), d2], 1), c_noise, labels)
        x = self.up2(torch.cat([F.interpolate(x, scale_factor=2), d1], 1), c_noise, labels)
        return self.out(x)

# === EDM PRECONDITIONING WRAPPER ===
class EDMWrapper(nn.Module):
    """EDM I/O preconditioning (Karras et al. 2022)"""
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
edm_wrapper = EDMWrapper(model).to(device)
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

# === EDM EULER SAMPLER ===
@torch.no_grad()
def edm_sampler(wrapper, n, labels, cfg=1.0, steps=20):
    sigmas = torch.linspace(SIGMA_MAX, SIGMA_MIN, steps, device=device)
    x = torch.randn(n, 3, args.resize, args.resize, device=device) * sigmas[0]
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        s_vec = torch.full((n,), s, device=device)
        D_u = wrapper(x, s_vec, None)
        D_c = wrapper(x, s_vec, labels)
        D = D_u + cfg * (D_c - D_u)
        x = x + (s_next - s) * (x - D) / s  # Euler ODE step
        x = torch.clip(x, -1.0, 1.0)
    return x

# === TRAINING LOOP (EDM-Native) ===
metrics_log = []
for epoch in trange(1, args.epochs+1, desc="Training"):
    model.train()
    epoch_loss, count = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Sample continuous noise levels
        sigma = sample_sigmas(imgs.shape[0], device)
        x_sigma = imgs + sigma[:,None,None,None] * torch.randn_like(imgs)
        
        # CFG masking (~10% unconditional)
        train_labels = labels.clone()
        train_labels[torch.rand_like(labels.float()) < 0.1] = -1
        
        # Forward pass with preconditioning
        cond_mask = train_labels != -1
        uncond_mask = ~cond_mask
        pred_x0 = torch.zeros_like(x_sigma)
        if cond_mask.any():
            pred_x0[cond_mask] = edm_wrapper(x_sigma[cond_mask], sigma[cond_mask], train_labels[cond_mask])
        if uncond_mask.any():
            pred_x0[uncond_mask] = edm_wrapper(x_sigma[uncond_mask], sigma[uncond_mask], None)
        
        # EDM loss weighting: w(σ) = 1 / c_out(σ)²
        c_out = sigma / torch.sqrt(sigma**2 + 1)
        loss_weight = (1.0 / (c_out ** 2)).view(-1, 1, 1, 1)
        
        optimizer.zero_grad()
        loss = (loss_weight * F.mse_loss(pred_x0, imgs, reduction='none')).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.shape[0]
        count += imgs.shape[0]
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        gen = edm_sampler(edm_wrapper, 128, real_labels[:128].to(device), cfg=args.cfg_scale, steps=args.sample_steps)
        fid = pca_fid(real_batch[:128], gen)
    
    metrics_log.append({'epoch': epoch, 'loss': epoch_loss/count, 'pca_fid': fid})
    print(f"  Epoch {epoch:02d} | Loss: {metrics_log[-1]['loss']:.4f} | PCA-FID: {fid:.2f}")

with open('nanoddpm_pro_metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=2)

# === VISUALIZATION ===
def plot_results():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].plot([m['epoch'] for m in metrics_log], [m['loss'] for m in metrics_log], marker='o')
    axs[0].set_title('Training Loss'); axs[0].grid(alpha=0.3)
    axs[1].plot([m['epoch'] for m in metrics_log], [m['pca_fid'] for m in metrics_log], marker='s', color='orange')
    axs[1].set_title('PCA-FID (↓ better)'); axs[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    
    print("\nFinal samples (EDM sampler):")
    gen_final = edm_sampler(edm_wrapper, 16, torch.randint(0, 10, (16,), device=device), cfg=args.cfg_scale, steps=args.sample_steps)
    grid = torchvision.utils.make_grid(gen_final, nrow=4, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(5,5)); plt.imshow(grid.permute(1,2,0).numpy()); plt.axis('off'); plt.show()

plot_results()
print("Done. Metrics saved to nanoddpm_pro_metrics.json")
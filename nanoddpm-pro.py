# nanoddpm_pro.py: From-scratch DDPM for CIFAR-10 (~200 lines)
# Features: Mini-UNet, Classifier-Free Guidance, DDIM Sampling, PCA-FID
# Educational build inspired by micrograd/minbpe/nanoddpm

import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np, math, json
from tqdm import trange
import torch.nn.functional as F

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--cfg_scale', type=float, default=4.0, help="1.0 = no guidance, 3.0-7.5 = strong")
parser.add_argument('--resize', type=int, default=32)
args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)
print(f"▶ nanoddpm-pro | {device} | {args.steps} steps | CFG: {args.cfg_scale}")

# === 1. NOISE SCHEDULE & FORWARD ===
T_steps = args.steps
beta = torch.linspace(1e-4, 0.02, T_steps, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x0, t):
    sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_1m = torch.sqrt(1.0 - alpha_bar[t])[:, None, None, None]
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1m * eps, eps

# === 2. DATASET (CIFAR-10) ===
transform = T.Compose([T.Resize(args.resize), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
real_batch, real_labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)))
real_batch = real_batch.to(device)

# === 3. MODEL (Mini-UNet + Time/Class Embedding) ===
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

model = MiniUNet(time_dim=128, ch=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3)
print(f"▶ Params: {sum(p.numel() for p in model.parameters()):,}")

# === 4. PCA-FID ===
def pca_fid(real, gen, n_components=32):
    r = real.view(real.shape[0], -1).cpu().double()
    g = gen.view(gen.shape[0], -1).cpu().double()
    data = torch.cat([r, g], 0)
    mean = data.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(data - mean, q=n_components)
    proj = (data - mean) @ V
    r_p, g_p = proj[:real.shape[0]], proj[real.shape[0]:]
    mu_r, mu_g = r_p.mean(0), g_p.mean(0)
    var_r, var_g = r_p.var(0) + 1e-6, g_p.var(0) + 1e-6
    return ((mu_r - mu_g)**2).sum().item() + (var_r + var_g-2*torch.sqrt(var_r*var_g)).sum().item()

# === 5. DDIM SAMPLING ===
@torch.no_grad()
def sample_ddim(n, labels, cfg_scale=1.0, ddim_steps=50):
    model.eval()
    x = torch.randn(n, 3, args.resize, args.resize, device=device)
    step_size = max(1, T_steps // ddim_steps)
    
    for t in reversed(range(0, T_steps, step_size)):
        t_t = torch.full((n,), t, dtype=torch.long, device=device)
        eps_u = model(x, t_t, None)
        eps_c = model(x, t_t, labels)
        eps = eps_u + cfg_scale * (eps_c - eps_u)
        
        a, ab = alpha[t], alpha_bar[t]
        pred_x0 = (x - torch.sqrt(1 - ab)*eps) / torch.sqrt(ab)
        a_prev = alpha[t - step_size] if t>=step_size else alpha[0]
        x = torch.sqrt(a_prev)*pred_x0 + torch.sqrt(1 - a_prev)*eps
        x = torch.clip(x, -1.0, 1.0)
    return x

# === 6. TRAINING LOOP ===
metrics_log = []
for epoch in trange(1, args.epochs+1, desc="Training"):
    model.train()
    epoch_loss, count = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        mask = torch.rand_like(labels.float()) < 0.1
        train_labels = labels.clone()
        train_labels[mask] = -1
        
        t = torch.randint(0, T_steps, (imgs.shape[0],), device=device)
        xt, eps = forward_diffusion(imgs, t)
        optimizer.zero_grad()
        
        # Split batch for CFG
        cond_idx = (train_labels != -1).nonzero(as_tuple=True)[0]
        uncond_idx = (train_labels == -1).nonzero(as_tuple=True)[0]
        pred_eps = torch.zeros_like(eps)
        
        if len(cond_idx) > 0:
            pred_eps[cond_idx] = model(xt[cond_idx], t[cond_idx], train_labels[cond_idx])
        if len(uncond_idx) > 0:
            pred_eps[uncond_idx] = model(xt[uncond_idx], t[uncond_idx], None)
        
        loss = F.mse_loss(pred_eps, eps)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*imgs.shape[0]
        count += imgs.shape[0]
    
    # Eval
    model.eval()
    with torch.no_grad():
        gen = sample_ddim(128, real_labels[:128].to(device), cfg_scale=args.cfg_scale, ddim_steps=20)
        fid = pca_fid(real_batch[:128], gen)
    
    metrics_log.append({'epoch': epoch, 'loss': epoch_loss/count, 'pca_fid': fid})
    print(f"  Epoch {epoch:02d} | Loss: {metrics_log[-1]['loss']:.4f} | PCA-FID: {fid:.2f}")

with open('nanoddpm_pro_metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=2)

# === 7. VISUALIZATION ===
def plot_results():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].plot([m['epoch'] for m in metrics_log], [m['loss'] for m in metrics_log], marker='o')
    axs[0].set_title('Training Loss')
    axs[0].grid(alpha=0.3)
    axs[1].plot([m['epoch'] for m in metrics_log], [m['pca_fid'] for m in metrics_log], marker='s', color='orange')
    axs[1].set_title('PCA-FID (↓ better)')
    axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nFinal samples:")
    gen_final = sample_ddim(16, torch.randint(0, 10, (16,), device=device), cfg_scale=args.cfg_scale, ddim_steps=50)
    grid = torchvision.utils.make_grid(gen_final, nrow=4, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(5,5))
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.axis('off')
    plt.show()

plot_results()
print("Done. Metrics saved to nanoddpm_pro_metrics.json")
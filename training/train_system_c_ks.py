"""
System C: Kuramoto-Sivashinsky — v8-1D vs FNO-1D vs UNet-1D
===============================================================
Nature Communications generality evidence: different physics mechanism,
same architecture principle.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, h5py, sys, os, time, json
from collections import defaultdict
sys.path.insert(0, '.')

DEV = torch.device('cuda')
torch.manual_seed(42)
OUT = 'results/paper_experiments'
os.makedirs(f'{OUT}', exist_ok=True)
os.makedirs('checkpoints/system_c', exist_ok=True)

# ============================================================
# Data
# ============================================================
with h5py.File('data_generated/ks_L32_N64_T30.0.h5', 'r') as f:
    fields = f['fields'][:]  # [3000, 64]

mean = float(np.mean(fields))
std = float(np.std(fields)) + 1e-8
n_total = len(fields) - 1
n_train = int(n_total * 0.8)

ins = fields[:n_train]
tgt = fields[1:n_train+1]
ins_v = fields[n_train:-1]
tgt_v = fields[n_train+1:]

print(f'KS L=32 N=64: train={len(ins)}, val={len(ins_v)}, mean={mean:.4f}, std={std:.4f}')


# ============================================================
# Models
# ============================================================
from models.rcln_1d_ks import RCLN_UPI_v8_1D

class FNO1D(nn.Module):
    """Simple FNO-1D for KS."""
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.modes = modes; self.width = width
        self.lift = nn.Conv1d(1, width, 1)
        self.sconv_weight = nn.Parameter(
            torch.randn(4, width, width, modes, dtype=torch.cfloat) * 0.02)
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.proj = nn.Sequential(nn.Conv1d(width, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))

    def forward(self, x):
        B, _, N = x.shape
        h = self.lift(x)
        for i in range(4):
            h_ft = torch.fft.rfft(h, dim=2)
            out_ft = torch.zeros(B, self.width, N//2+1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes] = torch.einsum(
                'bim,iom->bom', h_ft[:, :, :self.modes], self.sconv_weight[i])
            h_conv = torch.fft.irfft(out_ft, n=N, dim=2)
            h = F.gelu(h_conv + self.skip[i](h))
        return self.proj(h)

class UNet1D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        w = base
        def dc(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(max(1, out_ch//4), out_ch), nn.SiLU(),
                nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(max(1, out_ch//4), out_ch), nn.SiLU())
        self.enc0 = dc(1, w)
        self.d0 = nn.MaxPool1d(2)
        self.enc1 = dc(w, w*2)
        self.d1 = nn.MaxPool1d(2)
        self.enc2 = dc(w*2, w*4)
        self.d2 = nn.MaxPool1d(2)
        self.bn = dc(w*4, w*8)
        self.up2 = nn.ConvTranspose1d(w*8, w*4, 2, stride=2)
        self.dec2 = dc(w*8, w*4)
        self.up1 = nn.ConvTranspose1d(w*4, w*2, 2, stride=2)
        self.dec1 = dc(w*4, w*2)
        self.up0 = nn.ConvTranspose1d(w*2, w, 2, stride=2)
        self.dec0 = nn.Sequential(dc(w*2, w), nn.Conv1d(w, 1, 1))

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.d0(e0))
        e2 = self.enc2(self.d1(e1))
        b = self.bn(self.d2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.dec0(torch.cat([self.up0(d1), e0], 1))


# ============================================================
# Training
# ============================================================
def train_model(model, name, epochs=30, lr=1e-3):
    model = model.to(DEV)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float('inf')

    for ep in range(epochs):
        model.train()
        # Shuffle
        idx = np.random.permutation(len(ins))
        for i in range(0, len(ins), 64):
            batch_idx = idx[i:i+64]
            u = torch.from_numpy((ins[batch_idx] - mean) / std).float().unsqueeze(1).to(DEV)
            t = torch.from_numpy((tgt[batch_idx] - mean) / std).float().unsqueeze(1).to(DEV)
            opt.zero_grad()
            p = model(u)
            if isinstance(p, tuple): p = p[0]
            loss = F.mse_loss(p, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for i in range(0, len(ins_v), 64):
                u = torch.from_numpy((ins_v[i:i+64] - mean) / std).float().unsqueeze(1).to(DEV)
                t = torch.from_numpy((tgt_v[i:i+64] - mean) / std).float().unsqueeze(1).to(DEV)
                p = model(u)
                if isinstance(p, tuple): p = p[0]
                v_loss += F.mse_loss(p, t).item() * len(u)
        v_loss /= len(ins_v)

        if v_loss < best_val:
            best_val = v_loss
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': v_loss,
                        'params': n}, f'checkpoints/system_c/{name}_best.pt')

        if ep % 5 == 0 or ep < 3:
            print(f'  {name:12s} ep {ep+1:2d}/{epochs}: val={v_loss:.6f}  best={best_val:.6f}')
    return best_val


# ============================================================
# Rollout evaluation
# ============================================================
def rollout_eval(model, data, K, n_ic=10):
    model.eval()
    rl2_vals = defaultdict(list)
    e_vals = defaultdict(list)
    crashed = 0

    spacing = max(1, (len(data) - K - 1) // n_ic)
    with torch.no_grad():
        for ic in range(min(n_ic, (len(data) - K - 1) // spacing)):
            idx = ic * spacing
            u_np = (data[idx:idx+1] - mean) / std
            u = torch.from_numpy(u_np).float().unsqueeze(0).to(DEV)
            E_init = 0.5 * (data[idx]**2).sum()
            for step in range(K):
                t_np = (data[idx+step+1] - mean) / std
                pred = model(u)
                if isinstance(pred, tuple): pred = pred[0]
                p_np = pred.cpu().numpy()
                r2 = float(np.linalg.norm(p_np.flatten() - t_np.flatten()) / (
                    np.linalg.norm(t_np.flatten()) + 1e-8))
                if step + 1 in [10, 30, 50, 100, 200]:
                    rl2_vals[step+1].append(r2)
                    e_pred = 0.5 * ((p_np.squeeze() * std + mean)**2).sum()
                    e_vals[step+1].append(e_pred / E_init)
                if r2 > 10 or not np.isfinite(r2):
                    crashed += 1
                    break
                u = pred
    return rl2_vals, e_vals, crashed


# ── Main ──
print('\nTraining System C baselines...')
results = {}

# v8-1D
print('\n[v8-1D]')
v8 = RCLN_UPI_v8_1D()
results['v8-1D'] = train_model(v8, 'v8_1d')

# FNO-1D
print('\n[FNO-1D]')
fno = FNO1D(modes=12, width=48)
results['FNO-1D'] = train_model(fno, 'fno_1d')

# UNet-1D
print('\n[UNet-1D]')
unet = UNet1D(base=32)
results['UNet-1D'] = train_model(unet, 'unet_1d')

print(f'\n{"="*60}')
print('  SYSTEM C: K-S ROLLOUT (K=200, n_ic=5)')
print(f'{"="*60}')

# Reload best checkpoints
v8.load_state_dict(torch.load('checkpoints/system_c/v8_1d_best.pt',
                    map_location=DEV, weights_only=False)['model_state_dict'])
fno.load_state_dict(torch.load('checkpoints/system_c/fno_1d_best.pt',
                     map_location=DEV, weights_only=False)['model_state_dict'])
unet.load_state_dict(torch.load('checkpoints/system_c/unet_1d_best.pt',
                      map_location=DEV, weights_only=False)['model_state_dict'])

# Use held-out test data (last 10% of trajectory at higher L)
with h5py.File('data_generated/ks_L44_N64_T40.0.h5', 'r') as f:
    test_data = f['fields'][:]  # harder KS (L=44)

K_list = [10, 30, 50, 100, 200]

for name, model in [('v8-1D', v8), ('FNO-1D', fno), ('UNet-1D', unet)]:
    rl2, e, cr = rollout_eval(model, test_data, max(K_list), n_ic=5)
    parts = '  '.join(
        f'K={k}:{np.mean(rl2[k]):.4f}' if rl2[k] else f'K={k}:crash'
        for k in K_list
    )
    e50 = np.mean(e[50]) if e.get(50) else float('nan')
    print(f'  {name:10s}  {parts}  E@50={e50:.3f}  crash={cr}/5')

print(f'\nSystem C complete.')

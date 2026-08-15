"""
NC Mechanism M1-M3 (v5 NC P0-compliant, 2026-08-01)
=====================================================
M1: FNO+PhysReg — loss-level physics (div penalty + energy constraint)
M2: Learned gate — replace geometry gate with purely-learned conv3d gate
M3: Extrapolated IC stress test — 2x velocity amplitude rollout
     Uses v5_nc_p0 checkpoint (dissipation-aware, NC P0-1 compliant).

Execution: M1 (train) → M2 (train) → M3 (eval only, uses existing checkpoints)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, h5py, sys, os, time, json
from collections import defaultdict
sys.path.insert(0, '.')
DEV = 'cuda'
OUT = 'results/paper_experiments'
os.makedirs(OUT, exist_ok=True)
os.makedirs('checkpoints/mechanism', exist_ok=True)

torch.manual_seed(42)


# ============================================================
# Data
# ============================================================
class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', train_ratio=0.8):
        self.ins = np.load('checkpoints/re1600_ins.npy', mmap_mode='r')
        self.tgt = np.load('checkpoints/re1600_tgt.npy', mmap_mode='r')
        n = len(self.ins); idx = int(n * train_ratio)
        self.start = 0 if split == 'train' else idx
        self.n_samples = idx if split == 'train' else (n - idx)
        sample = np.array(self.ins[:50])
        self.mean = float(np.mean(sample))
        self.std = float(np.std(sample)) + 1e-8
    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        idx = self.start + i
        u = torch.from_numpy(self.ins[idx].astype(np.float32).copy()).float()
        t = torch.from_numpy(self.tgt[idx].astype(np.float32).copy()).float()
        return (u - self.mean) / self.std, (t - self.mean) / self.std

ds = MemmapDataset('train'); dsv = MemmapDataset('val')
tl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
vl = torch.utils.data.DataLoader(dsv, batch_size=1, shuffle=False)
print(f'Data: train={len(ds)}, val={len(dsv)}')

def rel_l2(pred, target):
    return float(np.linalg.norm(pred.flatten()-target.flatten())/(np.linalg.norm(target.flatten())+1e-8))


# ============================================================
# M1: FNO+PhysReg — loss-level physics baseline
# ============================================================
print(f'\n{"="*60}')
print('  M1: FNO+PhysReg — Loss-Level Physics')
print(f'{"="*60}')

from neuralop.models import FNO

class PhysRegLoss(nn.Module):
    """MSE + div penalty + energy constraint — loss-level physics."""
    def __init__(self, lambda_div=0.1, lambda_energy=0.05):
        super().__init__(); self.ld = lambda_div; self.le = lambda_energy
    def forward(self, pred, target):
        loss_data = F.mse_loss(pred, target)
        du = pred[:,0:1,1:,:-1,:-1] - pred[:,0:1,:-1,:-1,:-1]
        dv = pred[:,1:2,:-1,1:,:-1] - pred[:,1:2,:-1,:-1,:-1]
        dw = pred[:,2:3,:-1,:-1,1:] - pred[:,2:3,:-1,:-1,:-1]
        loss_div = (du+dv+dw).pow(2).mean()
        Ep = 0.5 * pred.pow(2).sum(dim=(1,2,3,4))
        Et = 0.5 * target.pow(2).sum(dim=(1,2,3,4)) + 1e-8
        loss_energy = F.smooth_l1_loss(Ep/Et, torch.ones_like(Ep), beta=0.1)
        return loss_data + self.ld*loss_div + self.le*loss_energy

fno_reg = FNO(n_modes=(8,8,8), hidden_channels=72, in_channels=3, out_channels=3, n_layers=4).to(DEV)
opt_reg = torch.optim.Adam(fno_reg.parameters(), lr=1e-4)
sch_reg = torch.optim.lr_scheduler.CosineAnnealingLR(opt_reg, 30)
criterion = PhysRegLoss()
best_fno_reg = float('inf')

print(f'FNO+PhysReg: {sum(p.numel() for p in fno_reg.parameters())/1e6:.1f}M params, 30 epochs')
for ep in range(30):
    fno_reg.train()
    for u, t in tl:
        u, t = u.to(DEV), t.to(DEV); opt_reg.zero_grad()
        loss = criterion(fno_reg(u), t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fno_reg.parameters(), 1.0)
        opt_reg.step()
    sch_reg.step()
    fno_reg.eval(); vl_loss = 0
    with torch.no_grad():
        for u, t in vl:
            u, t = u.to(DEV), t.to(DEV)
            vl_loss += F.mse_loss(fno_reg(u), t).item()
    vl_loss /= max(1, len(vl))
    if vl_loss < best_fno_reg:
        best_fno_reg = vl_loss
        torch.save({'model_state_dict': fno_reg.state_dict(), 'val_loss': vl_loss},
                   'checkpoints/mechanism/fno_physreg_best.pt')
    if ep % 5 == 0 or ep < 3:
        print(f'  ep {ep+1:2d}/30: val={vl_loss:.6f}  best={best_fno_reg:.6f}')

# Reload best
sd = torch.load('checkpoints/mechanism/fno_physreg_best.pt', map_location=DEV, weights_only=False)
fno_reg.load_state_dict(sd['model_state_dict'])
fno_reg.eval()
print(f'  Best val MSE: {best_fno_reg:.6f}')


# ============================================================
# M2: Learned Gate baseline
# ============================================================
print(f'\n{"="*60}')
print('  M2: Learned Gate — Replace Geometry Gate')
print(f'{"="*60}')

from models.rcln_upi_v5 import RCLN_UPI_v8

# Quick implementation: replace ChannelArbiter with purely-learned conv3d
class LearnedGate(nn.Module):
    """Pure learned gate — no cert_8, no geometric stimulus. Same param count."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 8, 1), nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv3d(8, 1, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x), torch.zeros_like(self.net(x)), torch.zeros(1), torch.zeros(1)

# Train v8 with learned gate instead of ChannelArbiter
v8_learned = RCLN_UPI_v8(
    in_channels=3, encoder_type='transformer', attention_mode='linear').to(DEV)
v8_learned.arbiter = LearnedGate(1).to(DEV)  # learns from sigma_comp only, no geometry
n_learned = sum(p.numel() for p in v8_learned.parameters())

# Modify forward: learned gate only sees sigma_comp, not cert_8
orig_forward = v8_learned.forward
def learned_forward(self, u, target=None, return_components=False, training_progress=0.0):
    f0, f2, f3 = self.encoder(u)
    z_struct, cert_8, energy = self.observer(f3)
    if self.ablation == 'no_cert': cert_8 = 0.5 * torch.ones_like(cert_8)
    u_comp, pi, mode_entropy, load_balance = self.soft_shell(f0, z_struct)
    sigma_comp = F.softplus((u_comp - u_comp.mean(dim=1, keepdim=True)).abs().mean(dim=1, keepdim=True))
    # M2: Learned gate — concatenated [sigma_comp only, no cert]
    gate_in = sigma_comp  # pure learned — no geometric signal
    w, stim, stim_sig, risk = self.arbiter(gate_in)
    if self.ablation == 'no_arbiter': w = 0.5 * torch.ones_like(w)
    u_fused = self.fusion(u_comp, z_struct, cert_8, w) if self.ablation != 'no_fusion' else u_comp
    film_params, mem_info = self.memory(u_raw=u, z_global=z_struct, u_global=u_fused,
                                        target=target, mode='train' if self.training else 'eval',
                                        training_progress=training_progress)
    u_final = self.memory.apply_film(u_fused, film_params)
    if self.use_energy_projection and not self.training:
        u_final, energy_info = self.energy_projector(u_final, u_input=u)
    else: energy_info = {}
    if not self.training:
        from models.rcln_upi_v5 import kinetic_energy
        E_final = kinetic_energy(u_final); E_input = kinetic_energy(u)
        explosion_mask = E_final > E_input * 1.05
        if explosion_mask.any():
            scale = torch.where(explosion_mask,
                torch.sqrt(E_input*1.05/(E_final+1e-8)), torch.ones_like(E_input))
            u_final = u_final * scale.view(-1,1,1,1,1)
    if return_components:
        return u_final, {'u_fused': u_fused, 'u_comp': u_comp, 'z_struct': z_struct,
                         'cert_8': cert_8, 'w': w}
    return u_final

import types
v8_learned.forward = types.MethodType(learned_forward, v8_learned)

opt_learned = torch.optim.Adam(v8_learned.parameters(), lr=1e-4)
sch_learned = torch.optim.lr_scheduler.CosineAnnealingLR(opt_learned, 30)
best_learned = float('inf')

print(f'Learned Gate (M2): {n_learned/1e6:.1f}M params, 30 epochs')
for ep in range(30):
    v8_learned.train()
    for u, t in tl:
        u, t = u.to(DEV), t.to(DEV); opt_learned.zero_grad()
        p = v8_learned(u)
        loss = F.mse_loss(p, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v8_learned.parameters(), 1.0)
        opt_learned.step()
    sch_learned.step()
    v8_learned.eval(); vl_loss = 0; wv = []
    with torch.no_grad():
        for u, t in vl:
            u, t = u.to(DEV), t.to(DEV)
            p, c = v8_learned(u, return_components=True)
            vl_loss += F.mse_loss(p, t).item()
            wv.append(c['w'].mean().item())
    vl_loss /= max(1, len(vl)); wm = np.mean(wv)
    if vl_loss < best_learned:
        best_learned = vl_loss
        torch.save({'model_state_dict': v8_learned.state_dict(), 'val_loss': vl_loss, 'w_mean': wm},
                   'checkpoints/mechanism/learned_gate_best.pt')
    if ep % 5 == 0 or ep < 3:
        print(f'  ep {ep+1:2d}/30: val={vl_loss:.6f} w={wm:.3f} best={best_learned:.6f}')

# Reload
sd = torch.load('checkpoints/mechanism/learned_gate_best.pt', map_location=DEV, weights_only=False)
v8_learned.load_state_dict(sd['model_state_dict'])
v8_learned.eval()
print(f'  Best val MSE: {best_learned:.6f}')


# ============================================================
# M3: Extrapolated IC Stress Test (v5 NC P0-compliant)
# ============================================================
print(f'\n{"="*60}')
print('  M3: Extrapolated IC — 2x Amplitude Stress Test (v5 NC P0)')
print(f'{"="*60}')

from models.rcln_upi_v5 import RCLN_UPI_v5 as V5

# Load v5 NC P0 best checkpoint
v5_nc = V5(in_channels=3, encoder_type='cnn').to(DEV)
sd = torch.load('checkpoints/v5_nc_p0/v5_best.pt', map_location=DEV, weights_only=False)
ms = v5_nc.state_dict()
if 'model_state_dict' in sd:
    loaded = {k: v for k, v in sd['model_state_dict'].items()
              if k in ms and isinstance(v, torch.Tensor) and isinstance(ms[k], torch.Tensor) and ms[k].shape == v.shape}
else:
    loaded = {k: v for k, v in sd.items()
              if k in ms and isinstance(v, torch.Tensor) and isinstance(ms[k], torch.Tensor) and ms[k].shape == v.shape}
v5_nc.load_state_dict(loaded, strict=False); v5_nc.eval()
print(f'  Loaded v5 NC P0: {sum(p.numel() for p in v5_nc.parameters())/1e6:.1f}M params')

# Load FNO-67 baseline (from checkpoints/baselines/)
fno67 = FNO(n_modes=(8,8,8), hidden_channels=72, in_channels=3, out_channels=3, n_layers=4).to(DEV)
sd_fno = 'checkpoints/baselines/fno67_best.pt'
if os.path.exists(sd_fno):
    sd2 = torch.load(sd_fno, map_location=DEV, weights_only=False)
    ms2 = fno67.state_dict()
    loaded2 = {k: v for k, v in sd2['model_state_dict'].items()
               if k in ms2 and isinstance(v, torch.Tensor) and isinstance(ms2[k], torch.Tensor) and ms2[k].shape == v.shape}
    fno67.load_state_dict(loaded2, strict=False); fno67.eval()
    print(f'  Loaded FNO-67 baseline')
else:
    print(f'  WARNING: {sd_fno} not found — skipping FNO-67 comparison')
    fno67 = None

with h5py.File('data_generated/tgv_re6400_N64_T20.0_fluidsim.h5', 'r') as f:
    data = f['fields'][:]
mean, std = float(np.mean(data)), float(np.std(data)) + 1e-8

K = 30; n_ic = 5; amp = 2.0

def divergence_tracker(model, init_field_raw, K):
    """Track per-step rL2 and divergence accumulation."""
    init = ((init_field_raw * amp) - mean) / std
    u = torch.from_numpy(init).float().unsqueeze(0).to(DEV)
    trace = {'rl2': [], 'divergence': [], 'energy_ratio': []}
    E_init = 0.5 * (init_field_raw**2).sum()
    with torch.no_grad():
        for step in range(K):
            pred = model(u)
            if isinstance(pred, tuple): pred = pred[0]
            p_raw = pred.cpu().numpy() * std + mean
            E_pred = 0.5 * (p_raw**2).sum()
            # Divergence on denormalized field
            du = p_raw[0,0:1,1:,:-1,:-1] - p_raw[0,0:1,:-1,:-1,:-1]
            dv = p_raw[0,1:2,:-1,1:,:-1] - p_raw[0,1:2,:-1,:-1,:-1]
            dw = p_raw[0,2:3,:-1,:-1,1:] - p_raw[0,2:3,:-1,:-1,:-1]
            div_rms = float(np.sqrt((du + dv + dw) ** 2).mean() + 1e-10)
            trace['divergence'].append(div_rms)
            trace['energy_ratio'].append(float(E_pred/E_init))

            # rL2 comparison: compare against amp-scaled reference
            r2 = float(np.linalg.norm(p_raw.flatten() - data[0].flatten()*amp) / (
                np.linalg.norm(data[0].flatten()*amp) + 1e-8))
            trace['rl2'].append(r2)
            if r2 > 10: break
            u = pred
    return trace

for name, model in [('v5 NC P0', v5_nc)] + ([('FNO-67', fno67)] if fno67 is not None else []):
    traces = []
    for ic in range(n_ic):
        t = divergence_tracker(model, data[ic*10], K)
        traces.append(t)
    # Aggregate
    for key in ['rl2', 'divergence', 'energy_ratio']:
        arr = np.array([t[key] for t in traces])
        mu = arr.mean(axis=0)
        print(f'  {name:<12s} {key:<12s}: K=1 {mu[0]:.4f}  K=10 {mu[min(9,len(mu)-1)]:.4f}  '
              f'K=30 {mu[min(29,len(mu)-1)]:.4f}  trend={"stable" if mu[-1]/mu[0] < 2 else "growing"}')

# Save
m3_data = {'v8_nc_2x': [], 'fno67_2x': []}
with open(f'{OUT}/mechanism_m1_m3.json', 'w') as f:
    json.dump({'M1_fno_physreg_val': best_fno_reg,
               'M2_learned_gate_val': best_learned,
               'M3_complete': True}, f)

print(f'\nM1-M3 complete. Results: {OUT}/mechanism_m1_m3.json')

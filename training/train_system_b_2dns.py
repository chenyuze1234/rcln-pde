"""
System B: 2D Navier-Stokes (Kolmogorov forced, mid-z slice)
==============================================================
NC Generality: v8-2D vs FNO-2D vs UNet-2D + Resolution OOD.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, h5py, sys, os, time
from collections import defaultdict
sys.path.insert(0, '.')
DEV = 'cuda'
OUT = 'results/paper_experiments'
os.makedirs(OUT, exist_ok=True)
os.makedirs('checkpoints/system_b', exist_ok=True)
torch.manual_seed(42)

def rel_l2(p, t):
    return float(np.linalg.norm(p.flatten()-t.flatten())/(np.linalg.norm(t.flatten())+1e-8))

# ============================================================
# Data: Kolmogorov 3D -> 2D mid-z slice (u,v only)
# ============================================================
with h5py.File('data_generated/kolmogorov_re1000_N64_T1.5.h5', 'r') as f:
    data3d = f['fields'][:]  # [N, 3, 64, 64, 64]

# Mid-z slice, take (u, v) only — proper 2D incompressible flow
data_uv = data3d[:, :2, 32, :, :]  # [N, 2, 64, 64]
mean = float(np.mean(data_uv)); std = float(np.std(data_uv)) + 1e-8
n_total = len(data_uv) - 1
n_train = int(n_total * 0.8)
ins = data_uv[:n_train]; tgt = data_uv[1:n_train+1]
ins_v = data_uv[n_train:-1]; tgt_v = data_uv[n_train+1:]
print(f'2D NS (Kolmogorov mid-z): train={len(ins)}, val={len(ins_v)}, mean={mean:.4f}, std={std:.4f}')

# ============================================================
# Baselines
# ============================================================
from models.rcln_2d_ns import RCLN_UPI_v8_2D

class FNO2D(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__(); self.m=modes; self.w=width
        self.lift=nn.Conv2d(2,width,1)
        self.w1=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)*0.02)
        self.w2=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)*0.02)
        self.w3=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)*0.02)
        self.w4=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)*0.02)
        self.skips=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(4)])
        self.proj=nn.Sequential(nn.Conv2d(width,64,1),nn.GELU(),nn.Conv2d(64,2,1))
    def forward(self,x):
        B,_,H,W=x.shape; h=self.lift(x)
        for w_par,sk in zip([self.w1,self.w2,self.w3,self.w4],self.skips):
            hf=torch.fft.rfft2(h,dim=(2,3))
            of=torch.zeros(B,self.w,H,W//2+1,dtype=torch.cfloat,device=x.device)
            of[:,:,:self.m,:self.m]=torch.einsum('bixy,ioxy->boxy',hf[:,:,:self.m,:self.m],w_par)
            h=F.gelu(torch.fft.irfft2(of,s=(H,W),dim=(2,3))+sk(h))
        return self.proj(h)

class UNet2D(nn.Module):
    def __init__(self,base=24):
        super().__init__(); w=base
        def dc(in_ch,out_ch):
            return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1),nn.GroupNorm(4,out_ch),nn.SiLU(),
                                 nn.Conv2d(out_ch,out_ch,3,padding=1),nn.GroupNorm(4,out_ch),nn.SiLU())
        self.e0=dc(2,w); self.d0=nn.MaxPool2d(2)
        self.e1=dc(w,w*2); self.d1=nn.MaxPool2d(2)
        self.e2=dc(w*2,w*4); self.d2=nn.MaxPool2d(2)
        self.bn=dc(w*4,w*8)
        self.u2=nn.ConvTranspose2d(w*8,w*4,2,2); self.dc2=dc(w*8,w*4)
        self.u1=nn.ConvTranspose2d(w*4,w*2,2,2); self.dc1=dc(w*4,w*2)
        self.u0=nn.ConvTranspose2d(w*2,w,2,2); self.dc0=nn.Sequential(dc(w*2,w),nn.Conv2d(w,2,1))
    def forward(self,x):
        e0=self.e0(x); e1=self.e1(self.d0(e0)); e2=self.e2(self.d1(e1)); b=self.bn(self.d2(e2))
        d2=self.dc2(torch.cat([self.u2(b),e2],1)); d1=self.dc1(torch.cat([self.u1(d2),e1],1))
        return self.dc0(torch.cat([self.u0(d1),e0],1))

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
        idx = np.random.permutation(len(ins))
        bs = 64
        for i in range(0, len(ins), bs):
            bi = idx[i:i+bs]
            u = torch.from_numpy((ins[bi]-mean)/std).float().to(DEV)
            t = torch.from_numpy((tgt[bi]-mean)/std).float().to(DEV)
            opt.zero_grad()
            p = model(u)
            if isinstance(p, tuple): p = p[0]
            loss = F.mse_loss(p, t); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for i in range(0, len(ins_v), bs):
                u = torch.from_numpy((ins_v[i:i+bs]-mean)/std).float().to(DEV)
                t = torch.from_numpy((tgt_v[i:i+bs]-mean)/std).float().to(DEV)
                p = model(u)
                if isinstance(p, tuple): p = p[0]
                vl += F.mse_loss(p, t).item() * len(u)
        vl /= len(ins_v)
        if vl < best_val:
            best_val = vl
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': vl, 'params': n},
                       f'checkpoints/system_b/{name}_best.pt')
        if ep % 5 == 0 or ep < 3:
            print(f'  {name:12s} ep {ep+1:2d}/{epochs}: val={vl:.6f}  best={best_val:.6f}')
    return best_val

# ============================================================
# Rollout
# ============================================================
def rollout_2d(model, data_uv, K, n_ic, K_checkpoints=None, data_mean=None, data_std=None):
    if K_checkpoints is None: K_checkpoints = [10, 50, 100, 200]
    if data_mean is None: data_mean = mean
    if data_std is None: data_std = std
    max_K = max(K_checkpoints)
    spacing = max(1, (len(data_uv) - max_K - 1) // n_ic)
    rl2_at = {k: [] for k in K_checkpoints}
    e_at = {k: [] for k in K_checkpoints}
    crashed = 0
    with torch.no_grad():
        for ic in range(min(n_ic, (len(data_uv)-max_K-1)//spacing)):
            idx = ic * spacing; idx = min(idx, len(data_uv)-max_K-2)
            init = (data_uv[idx]-data_mean)/data_std
            u = torch.from_numpy(init).float().unsqueeze(0).to(DEV)
            E0 = 0.5*(data_uv[idx]**2).sum()
            for step in range(max_K):
                pred = model(u)
                if isinstance(pred, tuple): pred = pred[0]
                p_np = pred.cpu().numpy().squeeze()
                t_np = (data_uv[idx+step+1]-data_mean)/data_std
                r2 = rel_l2(p_np, t_np)
                if step+1 in K_checkpoints:
                    rl2_at[step+1].append(r2)
                    E_pred = 0.5*((p_np*data_std+data_mean)**2).sum()
                    e_at[step+1].append(E_pred/E0)
                if r2 > 10 or not np.isfinite(r2): crashed += 1; break
                u = pred
    return rl2_at, e_at, crashed

# ============================================================
# Main
# ============================================================
print('\nTraining System B models...')

# v8-2D
print('\n[v8-2D]')
v8_2d = RCLN_UPI_v8_2D()
r_v8 = train_model(v8_2d, 'v8_2d')

# FNO-2D
print('\n[FNO-2D]')
fno2d = FNO2D(modes=12, width=48)
r_fno = train_model(fno2d, 'fno_2d')

# UNet-2D
print('\n[UNet-2D]')
unet2d = UNet2D(base=28)
r_unet = train_model(unet2d, 'unet_2d')

# ── Reload best ──
def load_best(m, path):
    sd = torch.load(path, map_location=DEV, weights_only=False)
    m.load_state_dict(sd['model_state_dict']); m.eval()

load_best(v8_2d, 'checkpoints/system_b/v8_2d_best.pt')
load_best(fno2d, 'checkpoints/system_b/fno_2d_best.pt')
load_best(unet2d, 'checkpoints/system_b/unet_2d_best.pt')

# ── Evaluation: OOD flows ──
print(f'\n{"="*60}')
print('  SYSTEM B: 2D NS Rollout (K=10/50/100, n_ic=5)')
print(f'{"="*60}')

test_flows = {
    'Kolmogorov (in-distr)': ('data_generated/kolmogorov_re1000_N64_T1.5.h5', False),
    'Shear Layer (OOD)': ('data_generated/shear_layer_re1000_N64_T0.6.h5', True),
    'HIT (OOD)': ('data_generated/decaying_hit_re1000_N64_T0.6.h5', True),
}

K_list = [10, 50, 100]
for flow_name, (h5_path, is_ood) in test_flows.items():
    if not os.path.exists(h5_path):
        print(f'  SKIP {flow_name}: file not found')
        continue
    with h5py.File(h5_path, 'r') as f:
        flow_data = f['fields'][:, :2, 32, :, :]
    fm, fs = float(np.mean(flow_data)), float(np.std(flow_data)) + 1e-8
    max_K = max(K_list)
    n_ic5 = min(5, (len(flow_data) - max_K - 1) // 3)

    print(f'\n  {flow_name} ({"OOD" if is_ood else "in-distribution"}): {len(flow_data)} frames')
    for name, model in [('v8-2D', v8_2d), ('FNO-2D', fno2d), ('UNet-2D', unet2d)]:
        rl2, e, cr = rollout_2d(model, flow_data, max_K, n_ic5,
                                K_checkpoints=K_list, data_mean=fm, data_std=fs)
        parts = '  '.join(
            f'K={k}:{np.mean(rl2[k]):.4f}' if rl2[k] else f'K={k}:crash'
            for k in K_list
        )
        e50 = np.mean(e[50]) if e.get(50) else float('nan')
        print(f'    {name:10s}  {parts}  E@50={e50:.3f}  crash={cr}/{n_ic5}')

print(f'\nSystem B complete. All results above.')
print(f'Checkpoints: checkpoints/system_b/')
print(f'Generality coverage: System A (3D TGV) + System B (2D NS) + System C (KS 1D)')

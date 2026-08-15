"""
Decaying Taylor-Green Vortex (different flow regime)
======================================================
EXACT same full-fftn solver as generate_v5_stable.py, but:
  - NO external forcing
  - With forcing = forced organized vortices
  - Without forcing = vortices break into decaying turbulence
This is standard — decaying TGV is a well-known benchmark.
"""

import numpy as np, torch, torch.fft as fft, h5py, os

N, Re, dt = 64, 1000, 0.10
nu = 1.0 / Re
n_steps, n_warmup = 200, 50
device = 'cuda'
print(f"Decaying TGV: N={N} Re={Re}")

# Full fftn wave numbers (EXACT match to generate_v5_stable.py)
k = fft.fftfreq(N, d=1.0/N) * 2 * np.pi
kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
kx, ky, kz = kx.to(device), ky.to(device), kz.to(device)
k2 = kx**2 + ky**2 + kz**2
k2_s = k2.clone(); k2_s[0,0,0] = 1.0

# Initial: Kolmogorov spectrum, solenoidal, unit KE
torch.manual_seed(42)
k_mag = torch.sqrt(k2)
k0 = N / 4
E_k = (k_mag/k0)**4 * torch.exp(-(k_mag/k0)**2); E_k[0,0,0] = 0
amp = torch.sqrt(E_k / (k_mag**2 + 1e-10))

u_hat = amp * torch.exp(1j*torch.rand_like(amp)*2*np.pi)
v_hat = amp * torch.exp(1j*torch.rand_like(amp)*2*np.pi)
w_hat = amp * torch.exp(1j*torch.rand_like(amp)*2*np.pi)

kdotu = kx*u_hat + ky*v_hat + kz*w_hat
u_hat -= kdotu*kx/k2_s; v_hat -= kdotu*ky/k2_s; w_hat -= kdotu*kz/k2_s
uh = torch.stack([u_hat, v_hat, w_hat], dim=0)

u = fft.ifftn(uh[0]).real; v = fft.ifftn(uh[1]).real; w = fft.ifftn(uh[2]).real
scale = 1.0 / (torch.sqrt(torch.mean(u**2+v**2+w**2)) + 1e-8)
uh *= scale
print(f"Initial KE=0.5 |u|max={u.abs().max().item():.2f}")

# Pseudo-spectral step (EXACT match, NO forcing)
def step(uh):
    u = fft.ifftn(uh[0]).real; v = fft.ifftn(uh[1]).real; w = fft.ifftn(uh[2]).real
    dudx=fft.ifftn(1j*kx*uh[0]).real; dudy=fft.ifftn(1j*ky*uh[0]).real; dudz=fft.ifftn(1j*kz*uh[0]).real
    dvdx=fft.ifftn(1j*kx*uh[1]).real; dvdy=fft.ifftn(1j*ky*uh[1]).real; dvdz=fft.ifftn(1j*kz*uh[1]).real
    dwdx=fft.ifftn(1j*kx*uh[2]).real; dwdy=fft.ifftn(1j*ky*uh[2]).real; dwdz=fft.ifftn(1j*kz*uh[2]).real
    cu=fft.fftn(u*dudx+v*dudy+w*dudz)
    cv=fft.fftn(u*dvdx+v*dvdy+w*dvdz)
    cw=fft.fftn(u*dwdx+v*dwdy+w*dwdz)
    un=uh[0]-dt*(cu+nu*k2*uh[0])
    vn=uh[1]-dt*(cv+nu*k2*uh[1])
    wn=uh[2]-dt*(cw+nu*k2*uh[2])
    ns=torch.stack([un,vn,wn],dim=0)
    div=kx*ns[0]+ky*ns[1]+kz*ns[2]; phi=div/k2_s
    ns[0]-=kx*phi; ns[1]-=ky*phi; ns[2]-=kz*phi
    if not torch.isfinite(ns).all(): raise ValueError(f"blow-up")
    return ns

curr=uh; fields=[]
for s in range(n_warmup+n_steps):
    curr=step(curr)
    if s>=n_warmup:
        ut=fft.ifftn(curr[0]).real.cpu().numpy(); vt=fft.ifftn(curr[1]).real.cpu().numpy(); wt=fft.ifftn(curr[2]).real.cpu().numpy()
        fields.append(np.stack([ut,vt,wt],axis=0).astype(np.float32))
    if s%50==0:
        uu=fft.ifftn(curr[0]).real; vv=fft.ifftn(curr[1]).real; ww=fft.ifftn(curr[2]).real
        ke=0.5*(uu**2+vv**2+ww**2).mean().item(); um=uu.abs().max().item()
        print(f'  {s:3d}: KE={ke:.6f} |u|max={um:.2f}')

fields=np.stack(fields,axis=0)
print(f'\n{fields.shape} {fields.nbytes/1e9:.2f}GB [{fields.min():.3f},{fields.max():.3f}]')

os.makedirs('data_generated',exist_ok=True)
path=f'data_generated/decaying_tgv_re{Re}_N{N}_T{n_steps*dt:.1f}.h5'
with h5py.File(path,'w') as f:
    f.create_dataset('fields',data=fields,compression='gzip',compression_opts=4)
    f.create_dataset('times',data=np.arange(n_steps)*dt)
    f.attrs['flow_type']='Decaying TGV'; f.attrs['Re']=Re; f.attrs['N']=N
print(f'Saved: {path}')

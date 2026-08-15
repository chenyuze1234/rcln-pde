"""
128^3 TGV — EXACT same fftn solver as generate_v5_stable.py, just N=128.
De-aliased, proven stable at 64^3.
"""
import numpy as np, torch, torch.fft as fft, h5py, os, time

N, Re, dt, steps, warmup = 128, 1000, 0.01, 150, 30
nu = 1.0/Re; device = 'cuda'
print(f"TGV N={N} Re={Re} dt={dt}")

# Full fftn grid (EXACT pattern from generate_v5_stable.py, proven stable)
k = fft.fftfreq(N, d=1.0/N) * 2*np.pi
kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
kx, ky, kz = kx.to(device), ky.to(device), kz.to(device)
k2 = kx**2 + ky**2 + kz**2; k2_s = k2.clone(); k2_s[0,0,0] = 1.0

# Narrower spectrum to avoid aliasing at high k
torch.manual_seed(42); km = torch.sqrt(k2); k0 = N/6  # shifted to lower k for resolution
Ek = (km/k0)**4 * torch.exp(-(km/k0)**2); Ek[0,0,0] = 0
amp = torch.sqrt(Ek / (km**2 + 1e-10))

uh = amp * torch.exp(1j*torch.rand_like(amp)*2*np.pi)
vh = amp * torch.exp(1j*torch.rand_like(amp)*2*np.pi)
wh = amp * torch.exp(1j*torch.rand_like(amp)*2*np.pi)
kd = kx*uh + ky*vh + kz*wh
uh -= kd*kx/k2_s; vh -= kd*ky/k2_s; wh -= kd*kz/k2_s
uhs = torch.stack([uh, vh, wh], dim=0)

u = fft.ifftn(uhs[0]).real; v = fft.ifftn(uhs[1]).real; w = fft.ifftn(uhs[2]).real
scale = 0.8/(torch.sqrt(torch.mean(u**2+v**2+w**2)) + 1e-8); uhs *= scale
u=fft.ifftn(uhs[0]).real
ke_init = 0.5*(u**2+v**2+w**2).mean().item()
print(f"Init |u|max={u.abs().max().item():.2f} KE={ke_init:.4f}")

kmax_dealias = N // 3  # 2/3 de-aliasing rule
dealias_mask = (torch.abs(kx) < kmax_dealias) & (torch.abs(ky) < kmax_dealias) & (torch.abs(kz) < kmax_dealias)

def step(uh):
    u=fft.ifftn(uh[0]).real; v=fft.ifftn(uh[1]).real; w=fft.ifftn(uh[2]).real
    du=fft.ifftn(1j*kx*uh[0]).real; dy=fft.ifftn(1j*ky*uh[0]).real; dz=fft.ifftn(1j*kz*uh[0]).real
    dv=fft.ifftn(1j*kx*uh[1]).real; de=fft.ifftn(1j*ky*uh[1]).real; df=fft.ifftn(1j*kz*uh[1]).real
    dw=fft.ifftn(1j*kx*uh[2]).real; dg=fft.ifftn(1j*ky*uh[2]).real; dh=fft.ifftn(1j*kz*uh[2]).real
    # Convection with 2/3 de-aliasing
    cu=fft.fftn(u*du+v*dy+w*dz)*dealias_mask
    cv=fft.fftn(u*dv+v*de+w*df)*dealias_mask
    cw=fft.fftn(u*dw+v*dg+w*dh)*dealias_mask
    un=uh[0]-dt*(cu+nu*k2*uh[0]); vn=uh[1]-dt*(cv+nu*k2*uh[1]); wn=uh[2]-dt*(cw+nu*k2*uh[2])
    ns=torch.stack([un,vn,wn],dim=0)
    div=kx*ns[0]+ky*ns[1]+kz*ns[2]; phi=div/k2_s
    ns[0]-=kx*phi; ns[1]-=ky*phi; ns[2]-=kz*phi
    return ns

cur=uhs; t0=time.time()
for s in range(warmup):
    cur=step(cur)
    if s%10==0:
        uu=fft.ifftn(cur[0]).real; ke=0.5*(uu**2).mean().item(); ok=torch.isfinite(cur).all()
        print(f"  warmup {s}: KE={ke:.6f} OK={ok}")
        if not ok: exit(1)

fields=[]
for s in range(steps):
    cur=step(cur)
    ut=fft.ifftn(cur[0]).real.cpu().numpy().astype(np.float32)
    vt=fft.ifftn(cur[1]).real.cpu().numpy().astype(np.float32)
    wt=fft.ifftn(cur[2]).real.cpu().numpy().astype(np.float32)
    fields.append(np.stack([ut,vt,wt],0))
    if s%20==0:
        ke=0.5*(ut**2+vt**2+wt**2).mean(); ok=np.isfinite(ut).all()
        print(f"  step {s}: KE={ke:.6f} |u|max={np.abs(ut).max():.2f} OK={ok}")
        if not ok: break

if np.isfinite(fields[-1]).all():
    fields=np.stack(fields,0)
    print(f"\n{fields.shape} {fields.nbytes/1e9:.2f}GB [{fields.min():.3f},{fields.max():.3f}] time={(time.time()-t0)/60:.1f}min")
    os.makedirs('data_generated',exist_ok=True)
    p=f'data_generated/tgv_re{Re}_N128_T{steps*dt:.1f}.h5'
    with h5py.File(p,'w') as f:
        f.create_dataset('fields',data=fields,compression='gzip',compression_opts=4)
        f.create_dataset('times',data=np.arange(steps)*dt)
        f.attrs['flow_type']='TGV'; f.attrs['Re']=Re; f.attrs['N']=128
    print(f"Saved: {p}")
else:
    print("FAILED: NaN output")

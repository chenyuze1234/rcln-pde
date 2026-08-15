"""
Generate Kolmogorov Flow (using verified stable solver core from generate_v5_stable.py)
========================================================================================
Same rfftn-based pseudo-spectral method as Taylor-Green — just adding sin(ky*y) forcing.
"""

import numpy as np
import torch
import torch.fft as fft
import h5py, os

N, Re, dt, n_steps, n_warmup = 64, 1000, 0.10, 200, 500
nu = 1.0 / Re
device = 'cuda'

print(f"Kolmogorov flow: N={N}, Re={Re}, dt={dt}, steps={n_steps}")

# === Wavenumbers (rfftn-compatible, exactly as in generate_v5_stable.py) ===
kx_1d = fft.fftfreq(N, d=1.0/N) * 2 * np.pi
kz_1d = fft.rfftfreq(N, d=1.0/N) * 2 * np.pi
kx, ky, kz = torch.meshgrid(
    torch.tensor(kx_1d, device=device, dtype=torch.float32),
    torch.tensor(kx_1d, device=device, dtype=torch.float32),
    torch.tensor(kz_1d, device=device, dtype=torch.float32),
    indexing='ij'
)
k2 = kx**2 + ky**2 + kz**2
k2_safe = k2.clone(); k2_safe[0,0,0] = 1.0

# === Kolmogorov forcing: sin(2*y) on x-component (physical space) ===
y = torch.linspace(0, 2*np.pi, N, device=device)
forcing_phys = 0.15 * torch.sin(2.0 * y)  # [N], sinusoidal forcing along y
# Force only on x-velocity: create [3, N, N, N] forcing for the rfftn stack
forcing_stack = torch.zeros(3, N, N, N, device=device)
forcing_stack[0] = forcing_phys.view(1, N, 1).expand(N, N, N)
# RFFT of forcing (applied in spectral space each step)
forcing_hat = torch.stack([
    fft.rfftn(forcing_stack[0]),
    fft.rfftn(forcing_stack[1]),
    fft.rfftn(forcing_stack[2]),
], dim=0)

# === Initial: tiny random perturbation (let forcing build the flow from near-rest) ===
torch.manual_seed(42)
eps = 1e-6
u = eps * torch.randn(N, N, N, device=device)
v = eps * torch.randn(N, N, N, device=device)
w = eps * torch.randn(N, N, N, device=device)
u_hat_stack = torch.stack([fft.rfftn(u), fft.rfftn(v), fft.rfftn(w)], dim=0)
# div-free project
div = kx*u_hat_stack[0] + ky*u_hat_stack[1] + kz*u_hat_stack[2]
u_hat_stack[0] -= kx*div/k2_safe; u_hat_stack[1] -= ky*div/k2_safe; u_hat_stack[2] -= kz*div/k2_safe
# Normalize to small amplitude
u_tmp = fft.irfftn(u_hat_stack[0], s=(N,N,N))
scale = 0.01 / (torch.sqrt(torch.mean(u_tmp**2)) + 1e-8)
u_hat_stack *= scale
print(f"Initial KE: {0.5*scale.item()**2:.8f}")

# === Stable pseudo-spectral step with forcing ramp ===
def step(u_hat, ramp=1.0):
    u = fft.irfftn(u_hat[0], s=(N,N,N))
    v = fft.irfftn(u_hat[1], s=(N,N,N))
    w = fft.irfftn(u_hat[2], s=(N,N,N))

    dudx=fft.irfftn(1j*kx*u_hat[0], s=(N,N,N)); dudy=fft.irfftn(1j*ky*u_hat[0], s=(N,N,N)); dudz=fft.irfftn(1j*kz*u_hat[0], s=(N,N,N))
    dvdx=fft.irfftn(1j*kx*u_hat[1], s=(N,N,N)); dvdy=fft.irfftn(1j*ky*u_hat[1], s=(N,N,N)); dvdz=fft.irfftn(1j*kz*u_hat[1], s=(N,N,N))
    dwdx=fft.irfftn(1j*kx*u_hat[2], s=(N,N,N)); dwdy=fft.irfftn(1j*ky*u_hat[2], s=(N,N,N)); dwdz=fft.irfftn(1j*kz*u_hat[2], s=(N,N,N))

    conv_u = fft.rfftn(u*dudx+v*dudy+w*dudz)
    conv_v = fft.rfftn(u*dvdx+v*dvdy+w*dvdz)
    conv_w = fft.rfftn(u*dwdx+v*dwdy+w*dwdz)

    u_new = u_hat[0] - dt*(conv_u + nu*k2*u_hat[0]) + ramp*dt*forcing_hat[0]
    v_new = u_hat[1] - dt*(conv_v + nu*k2*u_hat[1])
    w_new = u_hat[2] - dt*(conv_w + nu*k2*u_hat[2])
    new = torch.stack([u_new, v_new, w_new], dim=0)

    div = kx*new[0] + ky*new[1] + kz*new[2]
    phi = div / k2_safe
    new[0] -= kx*phi; new[1] -= ky*phi; new[2] -= kz*phi
    return new

# Warmup with forcing ramp
curr = u_hat_stack
ramp_steps = 200  # ramp forcing over first 200 steps
for s in range(n_warmup):
    if not torch.isfinite(curr).all():
        print(f"  BLOW-UP at warmup {s}"); exit(1)
    ramp = min(1.0, s / ramp_steps) if ramp_steps > 0 else 1.0
    curr = step(curr, ramp=ramp)
    if s % 50 == 0:
        u_t = fft.irfftn(curr[0], s=(N,N,N)); v_t = fft.irfftn(curr[1], s=(N,N,N)); w_t = fft.irfftn(curr[2], s=(N,N,N))
        ke = 0.5*(u_t**2+v_t**2+w_t**2).mean().item()
        print(f"  Warmup {s}/{n_warmup}: KE={ke:.6f}, |u|max={u_t.abs().max().item():.3f}, ramp={ramp:.2f}")

print("Warmup PASSED")

# Production
fields = []
for s in range(n_steps):
    curr = step(curr)
    u_t = fft.irfftn(curr[0], s=(N,N,N)).cpu().numpy()
    v_t = fft.irfftn(curr[1], s=(N,N,N)).cpu().numpy()
    w_t = fft.irfftn(curr[2], s=(N,N,N)).cpu().numpy()
    fields.append(np.stack([u_t, v_t, w_t], axis=0).astype(np.float32))
    if s % 50 == 0:
        ke = 0.5*(u_t**2+v_t**2+w_t**2).mean()
        print(f"  Step {s}/{n_steps}: KE={ke:.6f}")

fields = np.stack(fields, axis=0)
print(f"\nGenerated: {fields.shape}, {fields.nbytes/1e9:.2f} GB, range=[{fields.min():.3f}, {fields.max():.3f}]")

os.makedirs('data_generated', exist_ok=True)
path = f'data_generated/kolmogorov_re{Re}_N{N}_T{n_steps*dt:.1f}.h5'
with h5py.File(path, 'w') as f:
    f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
    f.create_dataset('times', data=np.arange(n_steps)*dt)
    f.attrs['flow_type']='Kolmogorov forcing'; f.attrs['Re']=Re; f.attrs['N']=N
print(f"Saved: {path}")

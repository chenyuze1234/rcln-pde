"""
Quick OOD data generation: 20 samples of 3D Re=5000 turbulence.
Based on validated generate_data_500_v3.py method.
"""
import numpy as np
import torch
import torch.fft as fft
import h5py

N = 64
Re = 5000
nu = 1.0 / Re
n_warmup = 300
n_steps = 5  # total evolution = n_steps * dt

def pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt):
    u = fft.irfftn(u_hat[0], s=(N, N, N))
    v = fft.irfftn(u_hat[1], s=(N, N, N))
    w = fft.irfftn(u_hat[2], s=(N, N, N))
    
    dudx = fft.irfftn(1j * kx * u_hat[0], s=(N, N, N))
    dudy = fft.irfftn(1j * ky * u_hat[0], s=(N, N, N))
    dudz = fft.irfftn(1j * kz * u_hat[0], s=(N, N, N))
    dvdx = fft.irfftn(1j * kx * u_hat[1], s=(N, N, N))
    dvdy = fft.irfftn(1j * ky * u_hat[1], s=(N, N, N))
    dvdz = fft.irfftn(1j * kz * u_hat[1], s=(N, N, N))
    dwdx = fft.irfftn(1j * kx * u_hat[2], s=(N, N, N))
    dwdy = fft.irfftn(1j * ky * u_hat[2], s=(N, N, N))
    dwdz = fft.irfftn(1j * kz * u_hat[2], s=(N, N, N))
    
    conv_u = u * dudx + v * dudy + w * dudz
    conv_v = u * dvdx + v * dvdy + w * dvdz
    conv_w = u * dwdx + v * dwdy + w * dwdz
    
    conv_u_hat = fft.rfftn(conv_u)
    conv_v_hat = fft.rfftn(conv_v)
    conv_w_hat = fft.rfftn(conv_w)
    
    u_hat_new = torch.stack([
        u_hat[0] - dt * (conv_u_hat + nu * k2 * u_hat[0]),
        u_hat[1] - dt * (conv_v_hat + nu * k2 * u_hat[1]),
        u_hat[2] - dt * (conv_w_hat + nu * k2 * u_hat[2]),
    ], dim=0)
    
    div = kx * u_hat_new[0] + ky * u_hat_new[1] + kz * u_hat_new[2]
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0
    phi = div / k2_safe
    u_hat_new[0] -= kx * phi
    u_hat_new[1] -= ky * phi
    u_hat_new[2] -= kz * phi
    
    return u_hat_new

def generate_field(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    kx_1d = torch.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kz_1d = torch.fft.rfftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(kx_1d, kx_1d, kz_1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k_mag = torch.sqrt(k2)
    
    k0 = 4.0
    E_k = (k_mag/k0)**4 / (1 + (k_mag/k0)**2)**(17/6)
    E_k[0, 0, 0] = 0
    
    amplitude = torch.sqrt(E_k / (4*np.pi*k_mag**2 + 1e-10))
    amplitude[0, 0, 0] = 0
    
    phase_u = torch.rand_like(amplitude) * 2 * np.pi
    phase_v = torch.rand_like(amplitude) * 2 * np.pi
    phase_w = torch.rand_like(amplitude) * 2 * np.pi
    
    u_hat = torch.stack([
        amplitude * torch.exp(1j * phase_u),
        amplitude * torch.exp(1j * phase_v),
        amplitude * torch.exp(1j * phase_w),
    ], dim=0)
    
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0
    div = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    phi = div / k2_safe
    u_hat[0] -= kx * phi
    u_hat[1] -= ky * phi
    u_hat[2] -= kz * phi
    
    return u_hat, kx, ky, kz, k2

def generate_sample(seed, device):
    u_hat, kx, ky, kz, k2 = generate_field(seed)
    u_hat = u_hat.to(device)
    kx = kx.to(device)
    ky = ky.to(device)
    kz = kz.to(device)
    k2 = k2.to(device)
    
    # Warmup with small dt for stability
    for _ in range(n_warmup):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt=0.001)
    
    u_input = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    
    # Evolution step: dt=0.2, 5 steps = 1.0 total
    dt = 0.2
    for _ in range(n_steps):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt)
    
    u_target = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    
    return u_input.cpu().numpy(), u_target.cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples = 20
    print(f"Generating {n_samples} OOD samples (Re={Re}, 3D {N}x{N}x{N}) on {device}...")
    
    inputs = np.zeros((n_samples, 3, N, N, N), dtype=np.float32)
    targets = np.zeros((n_samples, 3, N, N, N), dtype=np.float32)
    
    for i in range(n_samples):
        inp, tgt = generate_sample(10000 + i, device)
        inputs[i] = inp
        targets[i] = tgt
        if (i+1) % 5 == 0:
            print(f"  Generated {i+1}/{n_samples}")
    
    path = "data/ood_re5000_3d_20samples.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('input', data=inputs)
        f.create_dataset('target', data=targets)
        f.attrs['Re'] = Re
        f.attrs['N'] = N
    
    print(f"Saved to {path}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Input std: {inputs.std():.4e}")
    print(f"  Any NaN: {np.isnan(inputs).any()}")

if __name__ == '__main__':
    main()

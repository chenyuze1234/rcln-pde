"""
Generate 2D OOD turbulence data using stable torch pseudo-spectral method.
Adapted from generate_data_500_v3.py (validated 3D generator).
"""
import numpy as np
import torch
import torch.fft as fft
import h5py
from tqdm import tqdm


def pseudo_spectral_step_2d(u_hat, kx, ky, k2, nu, dt):
    """One explicit pseudo-spectral step for 2D NS."""
    N = u_hat.shape[1]
    
    # Transform to physical space
    u = fft.irfft2(u_hat[0], s=(N, N))
    v = fft.irfft2(u_hat[1], s=(N, N))
    
    # Derivatives
    dudx = fft.irfft2(1j * kx * u_hat[0], s=(N, N))
    dudy = fft.irfft2(1j * ky * u_hat[0], s=(N, N))
    dvdx = fft.irfft2(1j * kx * u_hat[1], s=(N, N))
    dvdy = fft.irfft2(1j * ky * u_hat[1], s=(N, N))
    
    # Nonlinear terms
    conv_u = u * dudx + v * dudy
    conv_v = u * dvdx + v * dvdy
    
    conv_u_hat = fft.rfft2(conv_u)
    conv_v_hat = fft.rfft2(conv_v)
    
    # Explicit Euler
    u_hat_new = torch.stack([
        u_hat[0] - dt * (conv_u_hat + nu * k2 * u_hat[0]),
        u_hat[1] - dt * (conv_v_hat + nu * k2 * u_hat[1]),
    ], dim=0)
    
    # Helmholtz projection
    div = kx * u_hat_new[0] + ky * u_hat_new[1]
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0
    phi = div / k2_safe
    u_hat_new[0] -= kx * phi
    u_hat_new[1] -= ky * phi
    
    return u_hat_new


def generate_field_2d(N, kx, ky, k2):
    """Generate random initial condition with Kolmogorov spectrum."""
    k_mag = torch.sqrt(k2)
    k0 = 4.0
    E_k = (k_mag / k0) ** 4 / (1 + (k_mag / k0) ** 2) ** (17 / 6)
    E_k[0, 0] = 0
    
    amplitude = torch.sqrt(E_k / (4 * np.pi * k_mag ** 2 + 1e-10))
    amplitude[0, 0] = 0
    
    phase_u = torch.rand_like(amplitude) * 2 * np.pi
    phase_v = torch.rand_like(amplitude) * 2 * np.pi
    
    u_hat = torch.stack([
        amplitude * torch.exp(1j * phase_u),
        amplitude * torch.exp(1j * phase_v),
    ], dim=0)
    
    # Project (rfft2 compatible)
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0
    div = kx * u_hat[0] + ky * u_hat[1]
    phi = div / k2_safe
    u_hat[0] -= kx * phi
    u_hat[1] -= ky * phi
    
    # Energy normalization: ensure E = 1.0
    u = torch.stack([
        fft.irfft2(u_hat[0], s=(N, N)),
        fft.irfft2(u_hat[1], s=(N, N))
    ], dim=0)
    energy = (u[0]**2 + u[1]**2).mean()
    if energy > 0:
        u = u / torch.sqrt(energy)
        u_hat = torch.stack([
            fft.rfft2(u[0]),
            fft.rfft2(u[1])
        ], dim=0)
    
    return u_hat


def generate_2d_turbulence_data(Re, n_samples, N=64, seed_base=0):
    """Generate 2D turbulence data at specified Reynolds number."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating {n_samples} samples at Re={Re} on {device}")
    
    nu = 1.0 / Re
    # Adaptive dt based on Re for stability
    # Higher Re -> smaller dt needed
    dt_warmup = min(0.001, 1.0 / Re)
    n_warmup = max(300, int(1.5 / dt_warmup))  # Total warmup time ~1.5
    dt_evolve = min(0.01, 5.0 / Re)
    n_steps = max(1, int(0.05 / dt_evolve))  # Total ~0.05 time units
    dt_evolve = 0.05 / n_steps  # Adjust exactly
    if dt_evolve > 0.01:
        dt_evolve = 0.01
        n_steps = 5
    
    print(f"  dt_warmup={dt_warmup}, n_warmup={n_warmup}, dt_evolve={dt_evolve}, n_steps={n_steps}")
    
    # Wavenumbers for rfft2: first dim N (full), second dim N//2+1 (real)
    kx_1d = torch.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky_1d = torch.fft.rfftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = torch.meshgrid(kx_1d, ky_1d, indexing='ij')
    k2 = kx**2 + ky**2
    
    kx = kx.to(device)
    ky = ky.to(device)
    k2 = k2.to(device)
    
    inputs = []
    targets = []
    
    for i in tqdm(range(n_samples), desc=f"Re={Re}"):
        torch.manual_seed(seed_base + i)
        np.random.seed(seed_base + i)
        
        u_hat = generate_field_2d(N, kx, ky, k2).to(device)
        
        # Warmup with small dt
        for _ in range(n_warmup):
            u_hat = pseudo_spectral_step_2d(u_hat, kx, ky, k2, nu, dt_warmup)
        
        # Save input
        u_input = torch.stack([
            fft.irfft2(u_hat[0], s=(N, N)),
            fft.irfft2(u_hat[1], s=(N, N))
        ], dim=0).cpu().numpy()
        
        # Evolve for target
        for _ in range(n_steps):
            u_hat = pseudo_spectral_step_2d(u_hat, kx, ky, k2, nu, dt_evolve)
        
        u_target = torch.stack([
            fft.irfft2(u_hat[0], s=(N, N)),
            fft.irfft2(u_hat[1], s=(N, N))
        ], dim=0).cpu().numpy()
        
        inputs.append(u_input)
        targets.append(u_target)
    
    inputs = np.stack(inputs, axis=0).astype(np.float64)
    targets = np.stack(targets, axis=0).astype(np.float64)
    
    return inputs, targets


def save_ood_data(Re, n_train=100, n_val=30, n_test=50, output_dir='data'):
    """Generate and save OOD data for a given Reynolds number."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating OOD data: Re={Re}")
    print(f"{'='*60}")
    
    train_in, train_tgt = generate_2d_turbulence_data(Re, n_train, seed_base=Re * 10000)
    val_in, val_tgt = generate_2d_turbulence_data(Re, n_val, seed_base=Re * 10000 + 100000)
    test_in, test_tgt = generate_2d_turbulence_data(Re, n_test, seed_base=Re * 10000 + 200000)
    
    path = f"{output_dir}/turbulence_re{Re}_64x64_ood.h5"
    with h5py.File(path, 'w') as f:
        g_train = f.create_group('train')
        g_train.create_dataset('input', data=train_in)
        g_train.create_dataset('target', data=train_tgt)
        
        g_val = f.create_group('val')
        g_val.create_dataset('input', data=val_in)
        g_val.create_dataset('target', data=val_tgt)
        
        g_test = f.create_group('test')
        g_test.create_dataset('input', data=test_in)
        g_test.create_dataset('target', data=test_tgt)
        
        f.attrs['Re'] = Re
        f.attrs['N'] = 64
    
    print(f"\nSaved to {path}")
    print(f"  Train: {train_in.shape}")
    print(f"  Val: {val_in.shape}")
    print(f"  Test: {test_in.shape}")
    print(f"  Input range: [{train_in.min():.4f}, {train_in.max():.4f}]")
    e_ratio = (test_tgt**2).mean() / (test_in**2).mean()
    print(f"  Energy ratio (target/input): {e_ratio:.4f}")
    
    return path


if __name__ == '__main__':
    # Generate Re=500 and Re=2000 OOD data
    for Re in [500, 2000]:
        save_ood_data(Re, n_train=100, n_val=30, n_test=50)

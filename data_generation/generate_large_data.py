"""
Generate large dataset - 500 train / 100 val / 100 test
"""
import torch
import numpy as np
import h5py
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N = 64
Re = 1000.0
nu = 1.0 / Re

# Wavenumbers
k = torch.fft.fftfreq(N, 1.0/N).to(device)
kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
k_mag[0, 0, 0] = 1.0

# Dealiasing
k_max = N // 3
dealias_mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)

def project_free(u_hat):
    ux, uy, uz = u_hat[0], u_hat[1], u_hat[2]
    div = kx * ux + ky * uy + kz * uz
    return torch.stack([
        ux - div * kx / k_mag**2,
        uy - div * ky / k_mag**2,
        uz - div * kz / k_mag**2
    ])

def compute_rhs(u_hat):
    u = torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])
    
    def grad(f_hat):
        return torch.stack([
            torch.fft.ifftn(1j * kx * f_hat).real,
            torch.fft.ifftn(1j * ky * f_hat).real,
            torch.fft.ifftn(1j * kz * f_hat).real
        ])
    
    grad_u = torch.stack([grad(u_hat[i]) for i in range(3)])
    nl = torch.stack([(u * grad_u[i]).sum(dim=0) for i in range(3)])
    nl_hat = torch.stack([torch.fft.fftn(nl[i]) for i in range(3)]) * dealias_mask.unsqueeze(0)
    return -nl_hat - nu * k_mag**2 * u_hat

def rk4_step(u_hat, dt):
    k1 = compute_rhs(u_hat)
    k2 = compute_rhs(u_hat + 0.5 * dt * k1)
    k3 = compute_rhs(u_hat + 0.5 * dt * k2)
    k4 = compute_rhs(u_hat + dt * k3)
    return project_free(u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

def generate_initial(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    k0 = 8.0
    E = (k_mag**4) * torch.exp(-2*(k_mag/k0)**2)
    E[0, 0, 0] = 0
    amp = torch.sqrt(E / (4 * np.pi * k_mag**2))
    u_hat = torch.stack([amp * torch.exp(1j * torch.rand(N, N, N, device=device) * 2 * np.pi) for _ in range(3)])
    u_hat = project_free(u_hat)
    u = torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])
    return u_hat / (u**2).sum(dim=0).mean().sqrt()

def to_phys(u_hat):
    return torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])

def generate_sample(seed, dt=0.01, n_steps=10):
    u_hat = generate_initial(seed)
    for _ in range(200):  # warmup
        u_hat = rk4_step(u_hat, 0.001)
    u_in = to_phys(u_hat).cpu().numpy()
    for _ in range(n_steps):
        u_hat = rk4_step(u_hat, dt)
        if not torch.isfinite(u_hat).all():
            return None, None
    return u_in, to_phys(u_hat).cpu().numpy()

def generate_set(n, offset, dt, n_steps, name):
    print(f"\nGenerating {name}: {n} samples...")
    inputs, targets = [], []
    for i in tqdm(range(n)):
        for retry in range(3):
            u_in, u_out = generate_sample(offset + i + retry * 100000, dt, n_steps)
            if u_in is not None:
                inputs.append(u_in)
                targets.append(u_out)
                break
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

def main():
    print("="*60)
    print("Large Dataset Generation")
    print("="*60)
    
    # Generate
    train_in, train_out = generate_set(500, 0, 0.01, 10, "Train")
    val_in, val_out = generate_set(100, 1000000, 0.01, 10, "Val")
    test_in, test_out = generate_set(100, 2000000, 0.01, 10, "Test")
    
    # Stats
    mean = train_in.mean()
    std = train_in.std()
    print(f"\nMean: {mean:.6f}, Std: {std:.6f}")
    
    # Save
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/ns_3d_large.h5', 'w') as f:
        f.create_dataset('train/input', data=train_in)
        f.create_dataset('train/target', data=train_out)
        f.create_dataset('val/input', data=val_in)
        f.create_dataset('val/target', data=val_out)
        f.create_dataset('test/input', data=test_in)
        f.create_dataset('test/target', data=test_out)
        f.attrs['mean'] = float(mean)
        f.attrs['std'] = float(std)
    
    print(f"\nSaved: data/ns_3d_large.h5")
    print(f"Train: {len(train_in)}, Val: {len(val_in)}, Test: {len(test_out)}")
    
    # Verify
    import numpy as np
    corr = np.corrcoef(train_in[0].flatten(), train_out[0].flatten())[0,1]
    print(f"Sample 0 correlation: {corr:.4f}")

if __name__ == '__main__':
    main()

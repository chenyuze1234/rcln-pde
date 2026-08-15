"""
3D NS Data Generation - Clean Version
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

# Dealiasing mask
k_max = N // 3
dealias_mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)

def project_divergence_free(u_hat):
    """Project to divergence-free space"""
    ux, uy, uz = u_hat[0], u_hat[1], u_hat[2]
    div = kx * ux + ky * uy + kz * uz
    ux = ux - div * kx / k_mag**2
    uy = uy - div * ky / k_mag**2
    uz = uz - div * kz / k_mag**2
    return torch.stack([ux, uy, uz])

def compute_rhs(u_hat):
    """Compute right-hand side: -u·∇u + ν∇²u"""
    # Transform to physical space
    u = torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])
    
    # Compute gradients in Fourier space
    def grad(f_hat):
        return torch.stack([
            torch.fft.ifftn(1j * kx * f_hat).real,
            torch.fft.ifftn(1j * ky * f_hat).real,
            torch.fft.ifftn(1j * kz * f_hat).real
        ])
    
    grad_u = torch.stack([grad(u_hat[i]) for i in range(3)])  # [3, 3, N, N, N]
    
    # Nonlinear term u·∇u
    nl = torch.stack([
        (u * grad_u[0]).sum(dim=0),
        (u * grad_u[1]).sum(dim=0),
        (u * grad_u[2]).sum(dim=0)
    ])
    
    nl_hat = torch.stack([torch.fft.fftn(nl[i]) for i in range(3)])
    nl_hat = nl_hat * dealias_mask.unsqueeze(0)
    
    # Diffusion term
    diffusion = -nu * k_mag**2 * u_hat
    
    return -nl_hat + diffusion

def rk4_step(u_hat, dt):
    """RK4 integration"""
    k1 = compute_rhs(u_hat)
    k2 = compute_rhs(u_hat + 0.5 * dt * k1)
    k3 = compute_rhs(u_hat + 0.5 * dt * k2)
    k4 = compute_rhs(u_hat + dt * k3)
    
    u_new = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return project_divergence_free(u_new)

def generate_initial(seed):
    """Generate random initial condition"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Energy spectrum: k^4 * exp(-2k^2/k0^2)
    k0 = 8.0
    E = (k_mag**4) * torch.exp(-2*(k_mag/k0)**2)
    E[0, 0, 0] = 0
    
    amplitude = torch.sqrt(E / (4 * np.pi * k_mag**2))
    
    u_hat = torch.zeros(3, N, N, N, dtype=torch.complex64, device=device)
    for i in range(3):
        phase = torch.rand(N, N, N, device=device) * 2 * np.pi
        u_hat[i] = amplitude * torch.exp(1j * phase)
    
    u_hat = project_divergence_free(u_hat)
    
    # Normalize
    u = torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])
    energy = (u**2).sum(dim=0).mean().sqrt()
    u_hat = u_hat / energy
    
    return u_hat

def to_physical(u_hat):
    return torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])

def compute_correlation(u1, u2):
    u1f = u1.flatten().cpu().numpy() if torch.is_tensor(u1) else u1.flatten()
    u2f = u2.flatten().cpu().numpy() if torch.is_tensor(u2) else u2.flatten()
    return np.corrcoef(u1f, u2f)[0, 1]

def compute_divergence(u):
    if not torch.is_tensor(u):
        u = torch.tensor(u, device=device)
    u_hat = torch.stack([torch.fft.fftn(u[i]) for i in range(3)])
    div_hat = 1j * (kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2])
    div = torch.fft.ifftn(div_hat).real
    return div.abs().mean().item()

def compute_energy_ratio(u_in, u_out):
    Ein = (u_in**2).mean()
    Eout = (u_out**2).mean()
    return (Eout / (Ein + 1e-10)).item() if torch.is_tensor(Eout) else Eout / (Ein + 1e-10)

def find_dt():
    """Find optimal dt"""
    print("\nFinding optimal dt...")
    u_hat = generate_initial(42)
    
    # Warmup
    for _ in range(200):
        u_hat = rk4_step(u_hat, 0.001)
    
    u_input = to_physical(u_hat)
    
    for dt in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        u_test = u_hat.clone()
        for _ in range(max(1, int(0.1/dt))):
            u_test = rk4_step(u_test, dt)
        corr = compute_correlation(u_input, to_physical(u_test))
        marker = " <-- SELECT" if 0.88 <= corr <= 0.96 else ""
        print(f"  dt={dt:.3f}: corr={corr:.4f}{marker}")
        if 0.88 <= corr <= 0.96:
            return dt, max(1, int(0.1/dt))
    
    return 0.05, 2

def generate_one(seed, dt, n_steps):
    """Generate one sample"""
    u_hat = generate_initial(seed)
    
    # Warmup
    for _ in range(200):
        u_hat = rk4_step(u_hat, 0.001)
    
    u_input = to_physical(u_hat).cpu().numpy()
    
    # Evolve
    for _ in range(n_steps):
        u_hat = rk4_step(u_hat, dt)
        if not torch.isfinite(u_hat).all():
            return None, None
    
    u_target = to_physical(u_hat).cpu().numpy()
    return u_input, u_target

def verify(u_in, u_out):
    """Verify sample quality"""
    corr = compute_correlation(u_in, u_out)
    div = compute_divergence(u_in)
    ratio = compute_energy_ratio(u_in, u_out)
    
    ok = (0.85 <= corr <= 0.97) and (div < 1e-5) and (0.85 <= ratio <= 1.05)
    return ok, corr, div, ratio

def main():
    print("="*60)
    print("3D NS Data Generation")
    print("="*60)
    
    # Step 1: Find dt
    dt, n_steps = find_dt()
    print(f"\nUsing dt={dt:.3f}, n_steps={n_steps}")
    
    # Step 2: Verify one sample
    print("\nVerifying one sample...")
    u_in, u_out = generate_one(999, dt, n_steps)
    if u_in is None:
        print("ERROR: Sample generation failed")
        return
    
    ok, corr, div, ratio = verify(u_in, u_out)
    print(f"  Correlation: {corr:.4f}")
    print(f"  Divergence: {div:.2e}")
    print(f"  Energy ratio: {ratio:.4f}")
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    
    if not ok:
        print("Adjusting parameters...")
        dt *= 0.5
        n_steps *= 2
    
    # Step 3: Generate dataset (small for testing)
    print("\nGenerating dataset...")
    n_train, n_val, n_test = 100, 20, 20
    
    train_in, train_out = [], []
    val_in, val_out = [], []
    test_in, test_out = [], []
    
    print(f"Train: {n_train} samples...")
    for i in tqdm(range(n_train)):
        for retry in range(3):
            u_in, u_out = generate_one(i + retry*10000, dt, n_steps)
            if u_in is not None:
                train_in.append(u_in)
                train_out.append(u_out)
                break
    
    print(f"Val: {n_val} samples...")
    for i in tqdm(range(n_val)):
        for retry in range(3):
            u_in, u_out = generate_one(100000 + i + retry*10000, dt, n_steps)
            if u_in is not None:
                val_in.append(u_in)
                val_out.append(u_out)
                break
    
    print(f"Test: {n_test} samples...")
    for i in tqdm(range(n_test)):
        for retry in range(3):
            u_in, u_out = generate_one(200000 + i + retry*10000, dt, n_steps)
            if u_in is not None:
                test_in.append(u_in)
                test_out.append(u_out)
                break
    
    # Convert to arrays
    train_in = np.array(train_in, dtype=np.float32)
    train_out = np.array(train_out, dtype=np.float32)
    val_in = np.array(val_in, dtype=np.float32)
    val_out = np.array(val_out, dtype=np.float32)
    test_in = np.array(test_in, dtype=np.float32)
    test_out = np.array(test_out, dtype=np.float32)
    
    # Normalization stats
    mean = train_in.mean()
    std = train_in.std()
    print(f"\nMean: {mean:.6f}, Std: {std:.6f}")
    
    # Save
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/ns_3d_v2.h5', 'w') as f:
        f.create_dataset('train/input', data=train_in)
        f.create_dataset('train/target', data=train_out)
        f.create_dataset('val/input', data=val_in)
        f.create_dataset('val/target', data=val_out)
        f.create_dataset('test/input', data=test_in)
        f.create_dataset('test/target', data=test_out)
        f.attrs['mean'] = float(mean)
        f.attrs['std'] = float(std)
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        f.attrs['dt'] = dt
        f.attrs['n_steps'] = n_steps
    
    print(f"\nSaved: data/ns_3d_v2.h5")
    print(f"Train: {len(train_in)}, Val: {len(val_in)}, Test: {len(test_out)}")
    
    # Verify random samples
    print("\nVerification:")
    for i in [0, 10, 50]:
        c = compute_correlation(train_in[i], train_out[i])
        print(f"  Sample {i}: corr={c:.4f}")

if __name__ == '__main__':
    main()

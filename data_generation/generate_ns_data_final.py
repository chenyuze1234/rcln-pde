"""
3D NS Data Generation - Following Best Practices
"""
import torch
import numpy as np
import h5py
import os
from tqdm import tqdm

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Parameters
N = 64
Re = 1000.0
nu = 1.0 / Re

def create_wavenumbers(N):
    """Create wavenumber arrays with correct ordering for FFT"""
    k = torch.fft.fftfreq(N, 1.0/N).to(device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0, 0, 0] = 1.0  # Avoid division by zero
    return kx, ky, kz, k_mag

kx, ky, kz, k_mag = create_wavenumbers(N)

# Dealiasing mask (2/3 rule)
k_max = N // 3
dealias_mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)
dealias_mask = dealias_mask.to(device)

def project_sol(u_hat):
    """Project to divergence-free space"""
    u_x = torch.fft.fftn(u_hat[0])
    u_y = torch.fft.fftn(u_hat[1])
    u_z = torch.fft.fftn(u_hat[2])
    
    div_hat = kx * u_x + ky * u_y + kz * u_z
    
    u_x -= div_hat * kx / k_mag**2
    u_y -= div_hat * ky / k_mag**2
    u_z -= div_hat * kz / k_mag**2
    
    u_hat[0] = torch.fft.ifftn(u_x).real
    u_hat[1] = torch.fft.ifftn(u_y).real
    u_hat[2] = torch.fft.ifftn(u_z).real
    
    return u_hat

def compute_nonlinear(u_hat):
    """Compute nonlinear term in physical space, then transform"""
    u_x = torch.fft.ifftn(u_hat[0]).real
    u_y = torch.fft.ifftn(u_hat[1]).real
    u_z = torch.fft.ifftn(u_hat[2]).real
    
    # Gradients in Fourier space
    def gradient(f_hat):
        return (torch.fft.ifftn(1j * kx * f_hat).real,
                torch.fft.ifftn(1j * ky * f_hat).real,
                torch.fft.ifftn(1j * kz * f_hat).real)
    
    grad_x = gradient(u_hat[0])
    grad_y = gradient(u_hat[1])
    grad_z = gradient(u_hat[2])
    
    # Nonlinear term: u · ∇u
    nl_x = u_x * grad_x[0] + u_y * grad_x[1] + u_z * grad_x[2]
    nl_y = u_x * grad_y[0] + u_y * grad_y[1] + u_z * grad_y[2]
    nl_z = u_x * grad_z[0] + u_y * grad_z[1] + u_z * grad_z[2]
    
    return torch.stack([
        torch.fft.fftn(nl_x),
        torch.fft.fftn(nl_y),
        torch.fft.fftn(nl_z)
    ])

def rk4_step(u_hat, dt):
    """RK4 time stepping with dealiasing"""
    def rhs(v_hat):
        nl = compute_nonlinear(v_hat)
        # Dealias
        nl = nl * dealias_mask.unsqueeze(0)
        diffusion = -nu * k_mag**2 * v_hat
        return -nl + diffusion
    
    k1 = rhs(u_hat)
    k2 = rhs(u_hat + 0.5 * dt * k1)
    k3 = rhs(u_hat + 0.5 * dt * k2)
    k4 = rhs(u_hat + dt * k3)
    
    u_hat_new = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Project to divergence-free
    u_hat_new = project_sol(u_hat_new)
    
    return u_hat_new

def generate_initial_field(seed):
    """Generate random initial field"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Random Fourier coefficients
    u_hat = torch.zeros(3, N, N, N, dtype=torch.complex64, device=device)
    
    # Energy spectrum: E(k) ~ k^4 * exp(-2*k^2/k0^2)
    k0 = 8.0
    energy_spec = (k_mag**4) * torch.exp(-2*(k_mag/k0)**2)
    energy_spec[0, 0, 0] = 0
    
    # Random phases
    amplitude = torch.sqrt(energy_spec / (4 * np.pi * k_mag**2 + 1e-10))
    
    for i in range(3):
        phase = torch.rand(N, N, N, device=device) * 2 * np.pi
        u_hat[i] = amplitude * torch.exp(1j * phase)
    
    # Transform to physical and back to ensure consistency
    u = torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])
    u_hat = torch.stack([torch.fft.fftn(u[i]) for i in range(3)])
    
    # Project to divergence-free
    u_hat = project_sol(u_hat)
    
    # Normalize to unit energy
    energy = (u**2).sum(dim=0).mean().sqrt()
    u_hat = u_hat / energy
    
    return u_hat

def to_physical(u_hat):
    """Convert to physical space"""
    return torch.stack([torch.fft.ifftn(u_hat[i]).real for i in range(3)])

def compute_divergence(u):
    """Compute divergence RMS"""
    if not torch.is_tensor(u):
        u = torch.tensor(u, device=device)
    u_hat = torch.stack([torch.fft.fftn(u[i]) for i in range(3)])
    div_hat = 1j * (kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2])
    div = torch.fft.ifftn(div_hat).real
    return div.abs().mean().item()

def compute_spectral_slope(u):
    """Compute energy spectrum slope"""
    if not torch.is_tensor(u):
        u = torch.tensor(u, device=device)
    u_hat = torch.stack([torch.fft.fftn(u[i]) for i in range(3)])
    E = 0.5 * (u_hat.abs()**2).sum(dim=0)
    
    # Shell averaging
    k_bins = torch.arange(1, N//2, device=device)
    E_shell = []
    
    for k in k_bins:
        mask = (k_mag >= k-0.5) & (k_mag < k+0.5)
        E_shell.append(E[mask].mean().item())
    
    E_shell = np.array(E_shell)
    k_vals = k_bins.cpu().numpy()
    
    # Fit slope in inertial range
    valid = (E_shell > 0) & (k_vals > 3) & (k_vals < 20)
    if valid.sum() < 3:
        return -1.0
    
    log_k = np.log(k_vals[valid])
    log_E = np.log(E_shell[valid])
    slope, _ = np.polyfit(log_k, log_E, 1)
    
    return slope

def compute_correlation(u1, u2):
    """Compute correlation coefficient"""
    if torch.is_tensor(u1):
        u1 = u1.cpu().numpy()
    if torch.is_tensor(u2):
        u2 = u2.cpu().numpy()
    u1_flat = u1.flatten()
    u2_flat = u2.flatten()
    return np.corrcoef(u1_flat, u2_flat)[0, 1]

def verify_sample(u_input, u_target, verbose=True):
    """Verify one sample quality"""
    results = {}
    
    # 1. Correlation
    corr = compute_correlation(u_input, u_target)
    results['corr'] = (corr, 0.88, 0.96)
    
    # 2. Divergence
    div = compute_divergence(u_input)
    results['div'] = (div, 0, 1e-5)
    
    # 3. Energy ratio
    E_in = (u_input**2).mean().item()
    E_out = (u_target**2).mean().item()
    ratio = E_out / (E_in + 1e-10)
    results['E_ratio'] = (ratio, 0.90, 1.02)
    
    # 4. Spectral slope
    slope = compute_spectral_slope(u_input)
    results['slope'] = (slope, -2.0, -1.3)
    
    if verbose:
        print(f"  Correlation: {corr:.4f}  (target: [0.88, 0.96])  {'✓' if 0.88 <= corr <= 0.96 else '✗'}")
        print(f"  Divergence:  {div:.2e}  (target: < 1e-5)       {'✓' if div < 1e-5 else '✗'}")
        print(f"  Energy ratio: {ratio:.4f}  (target: [0.90, 1.02]) {'✓' if 0.90 <= ratio <= 1.02 else '✗'}")
        print(f"  Slope:       {slope:.3f}  (target: [-2.0, -1.3])  {'✓' if -2.0 <= slope <= -1.3 else '✗'}")
    
    all_pass = all(lo <= val <= hi for val, (lo, hi) in [(v, (l, h)) for _, (v, l, h) in results.items()])
    return all_pass, results

def find_optimal_dt(seed=42, verbose=True):
    """Find optimal dt for target correlation"""
    if verbose:
        print("\n" + "="*60)
        print("Finding optimal dt...")
        print("="*60)
    
    # Generate initial field
    u_hat = generate_initial_field(seed)
    
    # Warmup
    dt_warmup = 0.001
    for _ in range(200):
        u_hat = rk4_step(u_hat, dt_warmup)
    
    u_input = to_physical(u_hat)
    
    # Test different dt values
    dt_candidates = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    best_dt = None
    best_score = float('inf')
    
    for dt in dt_candidates:
        u_hat_test = u_hat.clone()
        n_steps = max(1, int(0.10 / dt))  # Aim for ~0.10 total evolution
        
        valid = True
        for _ in range(n_steps):
            u_hat_test = rk4_step(u_hat_test, dt)
            if not torch.isfinite(u_hat_test).all():
                valid = False
                break
        
        if not valid:
            if verbose:
                print(f"  dt={dt:.3f}: NaN (unstable)")
            continue
        
        u_target = to_physical(u_hat_test)
        corr = compute_correlation(u_input, u_target)
        
        # Score: distance from target correlation 0.92
        score = abs(corr - 0.92)
        
        if verbose:
            marker = " ← SELECT" if 0.88 <= corr <= 0.96 else ""
            print(f"  dt={dt:.3f}: corr={corr:.4f}{marker}")
        
        if score < best_score and 0.85 <= corr <= 0.97:
            best_dt = (dt, n_steps)
            best_score = score
    
    if best_dt is None:
        print("  WARNING: No suitable dt found, using default (0.01, 10)")
        return 0.01, 10
    
    if verbose:
        print(f"\nSelected: dt={best_dt[0]:.3f}, n_steps={best_dt[1]}")
    
    return best_dt

def generate_one_sample(seed, dt, n_steps, n_warmup=200):
    """Generate one sample with given parameters"""
    u_hat = generate_initial_field(seed)
    
    # Warmup
    dt_warmup = 0.001
    for _ in range(n_warmup):
        u_hat = rk4_step(u_hat, dt_warmup)
    
    # Get input
    u_input = to_physical(u_hat)
    
    # Evolve to get target
    for _ in range(n_steps):
        u_hat = rk4_step(u_hat, dt)
        if not torch.isfinite(u_hat).all():
            return None, None
    
    u_target = to_physical(u_hat)
    
    return u_input.cpu().numpy(), u_target.cpu().numpy()

def main():
    print("="*60)
    print("3D NS Data Generation - Final Version")
    print("="*60)
    print(f"Resolution: {N}^3, Re: {Re}")
    
    # Step 1: Find optimal dt
    dt, n_steps = find_optimal_dt(seed=42)
    
    # Step 2: Verify one sample
    print("\n" + "="*60)
    print("Step 2: Verifying one sample...")
    print("="*60)
    
    u_input, u_target = generate_one_sample(999, dt, n_steps)
    
    if u_input is None:
        print("ERROR: Sample generation failed (NaN)")
        return
    
    print(f"Sample shape: {u_input.shape}")
    print(f"Input range: [{u_input.min():.4f}, {u_input.max():.4f}]")
    print(f"Target range: [{u_target.min():.4f}, {u_target.max():.4f}]")
    
    passed, results = verify_sample(u_input, u_target)
    
    if not passed:
        print("\nSample verification FAILED!")
        print("Adjusting parameters...")
        # Fallback to smaller dt
        dt = dt * 0.5
        n_steps = int(n_steps * 2)
        print(f"New: dt={dt:.4f}, n_steps={n_steps}")
    else:
        print("\n✓ Sample verification PASSED!")
    
    # Step 3: Generate full dataset
    print("\n" + "="*60)
    print("Step 3: Generating full dataset...")
    print("="*60)
    
    n_train = 500
    n_val = 100
    n_test = 100
    
    train_inputs = []
    train_targets = []
    val_inputs = []
    val_targets = []
    test_inputs = []
    test_targets = []
    
    # Generate training set
    print(f"\nGenerating {n_train} training samples...")
    failed_count = 0
    
    for i in tqdm(range(n_train)):
        for retry in range(3):
            u_in, u_tgt = generate_one_sample(i + retry * 10000, dt, n_steps)
            
            if u_in is not None:
                # Quick check
                corr = compute_correlation(torch.tensor(u_in), torch.tensor(u_tgt))
                if 0.85 <= corr <= 0.97:
                    train_inputs.append(u_in)
                    train_targets.append(u_tgt)
                    break
            
            if retry == 2:
                failed_count += 1
    
    print(f"  Generated: {len(train_inputs)}/{n_train} (failed: {failed_count})")
    
    # Generate validation set
    print(f"\nGenerating {n_val} validation samples...")
    for i in tqdm(range(n_val)):
        for retry in range(3):
            u_in, u_tgt = generate_one_sample(100000 + i + retry * 10000, dt, n_steps)
            if u_in is not None:
                val_inputs.append(u_in)
                val_targets.append(u_tgt)
                break
    
    print(f"  Generated: {len(val_inputs)}/{n_val}")
    
    # Generate test set
    print(f"\nGenerating {n_test} test samples...")
    for i in tqdm(range(n_test)):
        for retry in range(3):
            u_in, u_tgt = generate_one_sample(200000 + i + retry * 10000, dt, n_steps)
            if u_in is not None:
                test_inputs.append(u_in)
                test_targets.append(u_tgt)
                break
    
    print(f"  Generated: {len(test_inputs)}/{n_test}")
    
    # Convert to arrays
    train_inputs = np.array(train_inputs, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float32)
    val_inputs = np.array(val_inputs, dtype=np.float32)
    val_targets = np.array(val_targets, dtype=np.float32)
    test_inputs = np.array(test_inputs, dtype=np.float32)
    test_targets = np.array(test_targets, dtype=np.float32)
    
    # Compute normalization stats (from train only)
    mean = train_inputs.mean()
    std = train_inputs.std()
    
    print(f"\nNormalization stats:")
    print(f"  mean: {mean:.6f}")
    print(f"  std:  {std:.6f}")
    
    # Verify normalization
    train_norm = (train_inputs - mean) / std
    print(f"  After normalization: mean={train_norm.mean():.4f}, std={train_norm.std():.4f}")
    
    # Save to HDF5
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/ns_3d_final.h5', 'w') as f:
        f.create_dataset('train/input', data=train_inputs)
        f.create_dataset('train/target', data=train_targets)
        f.create_dataset('val/input', data=val_inputs)
        f.create_dataset('val/target', data=val_targets)
        f.create_dataset('test/input', data=test_inputs)
        f.create_dataset('test/target', data=test_targets)
        
        f.attrs['mean'] = float(mean)
        f.attrs['std'] = float(std)
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        f.attrs['dt'] = dt
        f.attrs['n_steps'] = n_steps
    
    print(f"\nSaved: data/ns_3d_final.h5")
    
    # Final verification
    print("\n" + "="*60)
    print("Final Dataset Verification")
    print("="*60)
    
    # Random samples
    idx = np.random.choice(len(train_inputs), 5, replace=False)
    for i in idx:
        u_in = torch.tensor(train_inputs[i])
        u_tgt = torch.tensor(train_targets[i])
        corr = compute_correlation(u_in, u_tgt)
        print(f"  Sample {i}: corr={corr:.4f}")
    
    # Cross-sample correlation
    u0 = train_inputs[0]
    u1 = train_inputs[50]
    cross_corr = np.corrcoef(u0.flatten(), u1.flatten())[0, 1]
    print(f"\n  Cross-sample corr: {cross_corr:.4f} (should be < 0.2)")
    
    # Spectral slope
    slope = compute_spectral_slope(torch.tensor(train_inputs[0]))
    print(f"  Spectral slope: {slope:.3f} (target: ~-1.67)")
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)

if __name__ == '__main__':
    main()

"""
STEP 1: Generate correct dataset
500 INDEPENDENT fields (different seeds)
Each evolved n_steps from its own initial condition
"""
import numpy as np
import torch
import torch.fft as fft
import h5py
import os
from tqdm import tqdm

N = 64
Re = 1000
nu = 1.0 / Re
dt = 0.01  # From STEP 0
n_steps = 5  # From STEP 0
n_warmup = 200  # Warmup to reach turbulent steady state

def pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt):
    """One step of pseudo-spectral NS solver"""
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
    """Generate turbulent field"""
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
    """Generate one input-target pair"""
    u_hat, kx, ky, kz, k2 = generate_field(seed)
    u_hat = u_hat.to(device)
    kx = kx.to(device)
    ky = ky.to(device)
    kz = kz.to(device)
    k2 = k2.to(device)
    
    # Warmup with smaller dt for stability
    for _ in range(n_warmup):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt=0.001)
    
    # Save as INPUT
    u_input = torch.stack([
        fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)
    ], dim=0)
    
    # Evolve n_steps with target dt to get TARGET
    for _ in range(n_steps):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt)
    
    u_target = torch.stack([
        fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)
    ], dim=0)
    
    return u_input.cpu().numpy(), u_target.cpu().numpy()


def generate_dataset():
    """Generate full dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nGenerating dataset:")
    print(f"  N = {N}")
    print(f"  Re = {Re}")
    print(f"  dt = {dt}")
    print(f"  n_steps = {n_steps}")
    print(f"  n_warmup = {n_warmup}")
    
    n_train = 500
    n_val = 100
    
    os.makedirs('data', exist_ok=True)
    output_file = f'data/turbulence_re{Re}_n{N}_correct.h5'
    
    with h5py.File(output_file, 'w') as f:
        # Training set
        print(f"\n[1] Generating TRAINING set ({n_train} samples)...")
        train_inputs = np.zeros((n_train, 3, N, N, N), dtype=np.float32)
        train_targets = np.zeros((n_train, 3, N, N, N), dtype=np.float32)
        
        for i in tqdm(range(n_train)):
            inp, tgt = generate_sample(i, device)
            train_inputs[i] = inp
            train_targets[i] = tgt
        
        f.create_dataset('train/input', data=train_inputs)
        f.create_dataset('train/target', data=train_targets)
        
        # Validation set
        print(f"\n[2] Generating VALIDATION set ({n_val} samples)...")
        val_inputs = np.zeros((n_val, 3, N, N, N), dtype=np.float32)
        val_targets = np.zeros((n_val, 3, N, N, N), dtype=np.float32)
        
        for i in tqdm(range(n_val)):
            inp, tgt = generate_sample(n_train + i, device)
            val_inputs[i] = inp
            val_targets[i] = tgt
        
        f.create_dataset('val/input', data=val_inputs)
        f.create_dataset('val/target', data=val_targets)
        
        # Statistics (from training set only)
        mean = train_inputs.mean()
        std = train_inputs.std()
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        f.attrs['dt'] = dt
        f.attrs['n_steps'] = n_steps
        
        print(f"\n[3] Statistics (from training set):")
        print(f"  mean = {mean:.6f}")
        print(f"  std = {std:.6f}")
    
    print(f"\nDataset saved: {output_file}")
    return output_file


def verify_dataset(filepath):
    """Verify dataset quality"""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    with h5py.File(filepath, 'r') as f:
        inputs = f['train/input'][:10]
        targets = f['train/target'][:10]
        mean = f.attrs['mean']
        std = f.attrs['std']
    
    all_pass = True
    
    # Check 1: Correlation
    corrs = [np.corrcoef(inputs[i].flatten(), targets[i].flatten())[0,1] 
             for i in range(10)]
    mean_corr = np.mean(corrs)
    print(f"\n[1] Input-Target Correlation: {mean_corr:.4f}")
    if 0.85 <= mean_corr <= 0.97:
        print("    PASS ✓")
    else:
        print("    FAIL ✗")
        all_pass = False
    
    # Check 2: Sample independence
    sample_corrs = [np.corrcoef(inputs[0].flatten(), inputs[i].flatten())[0,1]
                    for i in range(1, 10)]
    mean_sample_corr = np.mean(sample_corrs)
    print(f"\n[2] Sample Independence: {mean_sample_corr:.4f}")
    if mean_sample_corr < 0.2:
        print("    PASS ✓")
    else:
        print("    FAIL ✗")
        all_pass = False
    
    # Check 3: Energy ratio
    E_in = (inputs**2).mean()
    E_out = (targets**2).mean()
    e_ratio = E_out / E_in
    print(f"\n[3] Energy Ratio: {e_ratio:.4f}")
    if 0.90 <= e_ratio <= 1.02:
        print("    PASS ✓")
    else:
        print("    FAIL ✗")
        all_pass = False
    
    # Check 4: Normalization
    inputs_norm = (inputs - mean) / std
    print(f"\n[4] Normalization: mean={inputs_norm.mean():.4f}, std={inputs_norm.std():.4f}")
    if abs(inputs_norm.mean()) < 0.1 and 0.9 < inputs_norm.std() < 1.1:
        print("    PASS ✓")
    else:
        print("    FAIL ✗")
        all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("ALL CHECKS PASSED - Proceed to training")
    else:
        print("SOME CHECKS FAILED - Fix before training")
    print("="*60)
    
    return all_pass


if __name__ == '__main__':
    # Generate dataset
    filepath = generate_dataset()
    
    # Verify
    verify_dataset(filepath)

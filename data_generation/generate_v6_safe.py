"""
Generate SAFE data - Very conservative settings
Small dt, fewer warmup steps
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
dt = 0.005       # VERY SMALL for stability
n_steps = 10     # More steps for same total time
n_warmup = 50    # Fewer warmup steps

def pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt):
    """Stable pseudo-spectral step"""
    u = fft.irfftn(u_hat[0], s=(N, N, N))
    v = fft.irfftn(u_hat[1], s=(N, N, N))
    w = fft.irfftn(u_hat[2], s=(N, N, N))
    
    # Gradients
    dudx = fft.irfftn(1j * kx * u_hat[0], s=(N, N, N))
    dudy = fft.irfftn(1j * ky * u_hat[0], s=(N, N, N))
    dudz = fft.irfftn(1j * kz * u_hat[0], s=(N, N, N))
    dvdx = fft.irfftn(1j * kx * u_hat[1], s=(N, N, N))
    dvdy = fft.irfftn(1j * ky * u_hat[1], s=(N, N, N))
    dvdz = fft.irfftn(1j * kz * u_hat[1], s=(N, N, N))
    dwdx = fft.irfftn(1j * kx * u_hat[2], s=(N, N, N))
    dwdy = fft.irfftn(1j * ky * u_hat[2], s=(N, N, N))
    dwdz = fft.irfftn(1j * kz * u_hat[2], s=(N, N, N))
    
    # Convection
    conv_u = u * dudx + v * dudy + w * dudz
    conv_v = u * dvdx + v * dvdy + w * dvdz
    conv_w = u * dwdx + v * dwdy + w * dwdz
    
    conv_u_hat = fft.rfftn(conv_u)
    conv_v_hat = fft.rfftn(conv_v)
    conv_w_hat = fft.rfftn(conv_w)
    
    # Update
    u_hat_new = torch.stack([
        u_hat[0] - dt * (conv_u_hat + nu * k2 * u_hat[0]),
        u_hat[1] - dt * (conv_v_hat + nu * k2 * u_hat[1]),
        u_hat[2] - dt * (conv_w_hat + nu * k2 * u_hat[2]),
    ], dim=0)
    
    # Project
    div = kx * u_hat_new[0] + ky * u_hat_new[1] + kz * u_hat_new[2]
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0
    phi = div / k2_safe
    u_hat_new[0] -= kx * phi
    u_hat_new[1] -= ky * phi
    u_hat_new[2] -= kz * phi
    
    return u_hat_new


def generate_field(seed):
    """Generate IC"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    kx_1d = torch.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kz_1d = torch.fft.rfftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(kx_1d, kx_1d, kz_1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k_mag = torch.sqrt(k2)
    
    # Spectrum
    k0 = 4.0
    E_k = (k_mag/k0)**4 / (1 + (k_mag/k0)**2)**(17/6)
    E_k[0, 0, 0] = 0
    
    amplitude = torch.sqrt(E_k / (4*np.pi*k_mag**2 + 1e-10))
    amplitude[0, 0, 0] = 0
    
    # Random phases
    phase_u = torch.rand_like(amplitude) * 2 * np.pi
    phase_v = torch.rand_like(amplitude) * 2 * np.pi
    phase_w = torch.rand_like(amplitude) * 2 * np.pi
    
    u_hat = torch.stack([
        amplitude * torch.exp(1j * phase_u),
        amplitude * torch.exp(1j * phase_v),
        amplitude * torch.exp(1j * phase_w),
    ], dim=0)
    
    # Project
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0
    div = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    phi = div / k2_safe
    u_hat[0] -= kx * phi
    u_hat[1] -= ky * phi
    u_hat[2] -= kz * phi
    
    # Normalize energy
    u = fft.irfftn(u_hat[0], s=(N, N, N))
    v = fft.irfftn(u_hat[1], s=(N, N, N))
    w = fft.irfftn(u_hat[2], s=(N, N, N))
    energy = 0.5 * (u**2 + v**2 + w**2).mean()
    if energy > 0:
        u_hat = u_hat / torch.sqrt(energy)
    
    return u_hat, kx, ky, kz, k2


def generate_sample(seed, device):
    """Generate one sample"""
    u_hat, kx, ky, kz, k2 = generate_field(seed)
    u_hat = u_hat.to(device)
    kx = kx.to(device)
    ky = ky.to(device)
    kz = kz.to(device)
    k2 = k2.to(device)
    
    # Warmup
    for _ in range(n_warmup):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt=0.001)
    
    # Input
    u_input = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    
    # Evolve
    for _ in range(n_steps):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt)
    
    u_target = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    
    return u_input.cpu().numpy(), u_target.cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Generating SAFE data v6...")
    print(f"dt={dt}, n_steps={n_steps}, total_time={dt*n_steps:.3f}")
    print(f"warmup={n_warmup}\n")
    
    n_train = 20
    n_val = 5
    
    os.makedirs('data', exist_ok=True)
    
    train_inputs = []
    train_targets = []
    
    print("[1] Training samples...")
    for i in tqdm(range(n_train)):
        try:
            inp, tgt = generate_sample(i, device)
            train_inputs.append(inp)
            train_targets.append(tgt)
        except Exception as e:
            print(f"  Error at {i}: {e}")
    
    if len(train_inputs) == 0:
        print("ERROR: No training data!")
        return
    
    train_inputs = np.array(train_inputs, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float32)
    
    val_inputs = []
    val_targets = []
    
    print(f"\n[2] Validation samples...")
    for i in tqdm(range(n_val)):
        try:
            inp, tgt = generate_sample(n_train + i, device)
            val_inputs.append(inp)
            val_targets.append(tgt)
        except Exception as e:
            print(f"  Error at val {i}: {e}")
    
    val_inputs = np.array(val_inputs, dtype=np.float32)
    val_targets = np.array(val_targets, dtype=np.float32)
    
    # Save
    with h5py.File('data/experiment_v6_safe.h5', 'w') as f:
        f.create_dataset('train/input', data=train_inputs)
        f.create_dataset('train/target', data=train_targets)
        f.create_dataset('val/input', data=val_inputs)
        f.create_dataset('val/target', data=val_targets)
        
        mean = train_inputs.mean()
        std = train_inputs.std()
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['N'] = N
        f.attrs['Re'] = Re
    
    print(f"\n[3] Statistics:")
    print(f"  Samples: {len(train_inputs)} train, {len(val_inputs)} val")
    print(f"  mean: {mean:.4f}")
    print(f"  std:  {std:.4f}")
    print(f"  min:  {train_inputs.min():.4f}")
    print(f"  max:  {train_inputs.max():.4f}")
    
    # Verify
    print(f"\n{'='*50}")
    print("VERIFICATION")
    print('='*50)
    
    if np.isnan(mean) or std < 0.1:
        print("ERROR: Data invalid!")
        return
    
    corrs = [np.corrcoef(train_inputs[i].flatten(), train_targets[i].flatten())[0,1]
             for i in range(min(5, len(train_inputs)))]
    print(f"Correlation: {np.mean(corrs):.4f}")
    print(f"Data range: [{train_inputs.min():.4f}, {train_inputs.max():.4f}]")
    print(f"\n✓ Data generation successful!")
    print(f"  File: data/experiment_v6_safe.h5")


if __name__ == '__main__':
    main()

"""
Experiment V3: CORRECT data with dt=0.20
Target correlation: 0.92 (tested)
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
dt = 0.20       # CORRECT value from test
n_steps = 5     # Total evolution: 1.0
n_warmup = 300

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
    
    for _ in range(n_warmup):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt=0.001)
    
    u_input = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    
    for _ in range(n_steps):
        u_hat = pseudo_spectral_step(u_hat, kx, ky, kz, k2, nu, dt)
    
    u_target = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    
    return u_input.cpu().numpy(), u_target.cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Generating CORRECT dataset...")
    print(f"dt={dt}, n_steps={n_steps}, evolution={dt*n_steps:.2f}")
    print(f"Expected correlation: ~0.92\n")
    
    n_train = 100
    n_val = 20
    
    os.makedirs('data', exist_ok=True)
    
    with h5py.File('data/experiment_v3_correct.h5', 'w') as f:
        print("[1] Training samples...")
        train_inputs = np.zeros((n_train, 3, N, N, N), dtype=np.float32)
        train_targets = np.zeros((n_train, 3, N, N, N), dtype=np.float32)
        
        for i in tqdm(range(n_train)):
            inp, tgt = generate_sample(i, device)
            train_inputs[i] = inp
            train_targets[i] = tgt
        
        f.create_dataset('train/input', data=train_inputs)
        f.create_dataset('train/target', data=train_targets)
        
        print("\n[2] Validation samples...")
        val_inputs = np.zeros((n_val, 3, N, N, N), dtype=np.float32)
        val_targets = np.zeros((n_val, 3, N, N, N), dtype=np.float32)
        
        for i in tqdm(range(n_val)):
            inp, tgt = generate_sample(n_train + i, device)
            val_inputs[i] = inp
            val_targets[i] = tgt
        
        f.create_dataset('val/input', data=val_inputs)
        f.create_dataset('val/target', data=val_targets)
        
        mean = train_inputs.mean()
        std = train_inputs.std()
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        
        print(f"\n[3] Statistics: mean={mean:.6f}, std={std:.6f}")
    
    print(f"\nSaved: data/experiment_v3_correct.h5")
    
    # Verify
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with h5py.File('data/experiment_v3_correct.h5', 'r') as f:
        inp = f['train/input'][:20]
        tgt = f['train/target'][:20]
        
        corrs = [np.corrcoef(inp[i].flatten(), tgt[i].flatten())[0,1] 
                 for i in range(20)]
        mean_corr = np.mean(corrs)
        print(f"\nCorrelation: {mean_corr:.4f}")
        
        if 0.88 <= mean_corr <= 0.97:
            print("GOOD - Ready for training!")
        elif mean_corr > 0.97:
            print("TOO HIGH")
        else:
            print("TOO LOW")


if __name__ == '__main__':
    main()

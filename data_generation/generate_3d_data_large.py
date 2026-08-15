"""Generate large 3D turbulence dataset (500-1000+ samples).

Uses SpectralTurbulenceGenerator with RK4 time integration.
Generates (u(t), u(t+dt)) pairs for training.
"""

import sys
import time
import h5py
import numpy as np
import torch

sys.path.insert(0, '.')
from spectral_turbulence_generator import SpectralTurbulenceGenerator


def generate_3d_dataset(
    n_train=800,
    n_val=100,
    n_test=100,
    n_warmup=200,
    dt=0.01,
    resolution=64,
    reynolds=1000,
    save_path='data/turbulence3d_re1000_large.h5',
    device='cuda'
):
    """Generate 3D turbulence dataset with proper time evolution.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples  
        n_test: Number of test samples
        n_warmup: RK4 warmup steps to reach statistical steady state
        dt: Time step
        resolution: Grid resolution (N for N^3 grid)
        reynolds: Reynolds number
        save_path: Output h5 file path
        device: 'cuda' or 'cpu'
    """
    print(f"Generating 3D turbulence dataset: {resolution}^3, Re={reynolds}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"  Warmup steps: {n_warmup}, dt: {dt}")
    
    gen = SpectralTurbulenceGenerator(N=resolution, Re=reynolds, device=device)
    
    def generate_split(n_samples, split_name):
        """Generate n_samples (u_t, u_t_dt) pairs."""
        print(f"\nGenerating {split_name} split ({n_samples} samples)...")
        
        u_t = np.zeros((n_samples, 3, resolution, resolution, resolution), dtype=np.float32)
        u_t_dt = np.zeros((n_samples, 3, resolution, resolution, resolution), dtype=np.float32)
        
        split_t0 = time.time()
        for i in range(n_samples):
            # Generate initial field and warmup
            u_hat, v_hat, w_hat = gen.generate_initial_field()
            
            # Warmup to reach statistical steady state
            for _ in range(n_warmup):
                u_hat, v_hat, w_hat = gen.rk4_step(u_hat, v_hat, w_hat, dt)
            
            # Save u(t)
            u = torch.fft.ifftn(u_hat).real.cpu().numpy()
            v = torch.fft.ifftn(v_hat).real.cpu().numpy()
            w = torch.fft.ifftn(w_hat).real.cpu().numpy()
            u_t[i] = np.stack([u, v, w], axis=0).astype(np.float32)
            
            # Evolve one step for u(t+dt)
            u_hat_new, v_hat_new, w_hat_new = gen.rk4_step(u_hat, v_hat, w_hat, dt)
            
            u_new = torch.fft.ifftn(u_hat_new).real.cpu().numpy()
            v_new = torch.fft.ifftn(v_hat_new).real.cpu().numpy()
            w_new = torch.fft.ifftn(w_hat_new).real.cpu().numpy()
            u_t_dt[i] = np.stack([u_new, v_new, w_new], axis=0).astype(np.float32)
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - split_t0
                rate = (i + 1) / elapsed
                remaining = (n_samples - i - 1) / rate
                print(f"  {i+1}/{n_samples} ({rate:.1f} samples/s, ETA: {remaining/60:.1f}min)")
        
        return u_t, u_t_dt
    
    # Generate all splits
    overall_t0 = time.time()
    train_input, train_target = generate_split(n_train, 'train')
    val_input, val_target = generate_split(n_val, 'val')
    test_input, test_target = generate_split(n_test, 'test')
    
    # Save to h5
    print(f"\nSaving to {save_path}...")
    with h5py.File(save_path, 'w') as f:
        # Train
        f.create_dataset('train/input', data=train_input, compression='gzip', compression_opts=4)
        f.create_dataset('train/target', data=train_target, compression='gzip', compression_opts=4)
        # Val
        f.create_dataset('val/input', data=val_input, compression='gzip', compression_opts=4)
        f.create_dataset('val/target', data=val_target, compression='gzip', compression_opts=4)
        # Test
        f.create_dataset('test/input', data=test_input, compression='gzip', compression_opts=4)
        f.create_dataset('test/target', data=test_target, compression='gzip', compression_opts=4)
        # Metadata
        f.attrs['resolution'] = resolution
        f.attrs['Re'] = reynolds
        f.attrs['dt'] = dt
        f.attrs['nu'] = 1.0 / reynolds
    
    total_time = time.time() - overall_t0
    print(f"\nDone! Total time: {total_time/60:.1f} minutes")
    print(f"File size: {save_path}")
    
    # Verify
    with h5py.File(save_path, 'r') as f:
        print(f"\nVerification:")
        for key in ['train/input', 'train/target', 'val/input', 'val/target', 'test/input', 'test/target']:
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")


if __name__ == '__main__':
    generate_3d_dataset(
        n_train=800,
        n_val=100,
        n_test=100,
        n_warmup=200,
        dt=0.01,
        resolution=64,
        reynolds=1000,
        save_path='data/turbulence3d_re1000_1000samples.h5',
        device='cuda'
    )

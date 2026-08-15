"""
Quick 3D data generation for testing
Generate small dataset to verify training pipeline
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn, fftfreq

def generate_quick_3d():
    """Generate minimal 3D dataset for quick testing"""
    N = 64
    Re = 1000
    
    print("Generating quick 3D test data...")
    
    # Setup
    k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    nu = 1.0 / Re
    
    # Dealiasing
    k_max = np.max(np.abs(k))
    dealias_mask = (np.abs(kx) < 2/3 * k_max) & (np.abs(ky) < 2/3 * k_max) & (np.abs(kz) < 2/3 * k_max)
    dealias_mask = dealias_mask.astype(float)
    
    def step(u, v, w, dt=0.001):
        u_hat = fftn(u, axes=(0,1,2)) * dealias_mask
        v_hat = fftn(v, axes=(0,1,2)) * dealias_mask
        w_hat = fftn(w, axes=(0,1,2)) * dealias_mask
        
        # Simple diffusion step
        u_hat *= np.exp(-nu * k2 * dt)
        v_hat *= np.exp(-nu * k2 * dt)
        w_hat *= np.exp(-nu * k2 * dt)
        
        u = np.real(ifftn(u_hat, axes=(0,1,2)))
        v = np.real(ifftn(v_hat, axes=(0,1,2)))
        w = np.real(ifftn(w_hat, axes=(0,1,2)))
        
        return u, v, w
    
    # Generate 100 train samples
    print("Generating 100 train samples...")
    train_input = []
    train_target = []
    
    for i in range(10):  # 10 trajectories
        # Random initial condition
        np.random.seed(i)
        u = np.random.randn(N, N, N).astype(np.float32) * 0.1
        v = np.random.randn(N, N, N).astype(np.float32) * 0.1
        w = np.random.randn(N, N, N).astype(np.float32) * 0.1
        
        # Burn-in
        for _ in range(100):
            u, v, w = step(u, v, w)
        
        # Collect 10 samples per trajectory
        for _ in range(10):
            train_input.append(np.stack([u, v, w], axis=0))
            
            # Evolve
            for _ in range(10):
                u, v, w = step(u, v, w)
            
            train_target.append(np.stack([u, v, w], axis=0))
    
    train_input = np.array(train_input, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    
    # Generate 20 val samples
    print("Generating 20 val samples...")
    val_input = []
    val_target = []
    
    for i in range(2):
        np.random.seed(100+i)
        u = np.random.randn(N, N, N).astype(np.float32) * 0.1
        v = np.random.randn(N, N, N).astype(np.float32) * 0.1
        w = np.random.randn(N, N, N).astype(np.float32) * 0.1
        
        for _ in range(100):
            u, v, w = step(u, v, w)
        
        for _ in range(10):
            val_input.append(np.stack([u, v, w], axis=0))
            
            for _ in range(10):
                u, v, w = step(u, v, w)
            
            val_target.append(np.stack([u, v, w], axis=0))
    
    val_input = np.array(val_input, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    
    # Save
    import os
    os.makedirs('data', exist_ok=True)
    
    with h5py.File('data/turbulence3d_re1000_64x64x64.h5', 'w') as f:
        f.create_dataset('train/input', data=train_input)
        f.create_dataset('train/target', data=train_target)
        f.create_dataset('val/input', data=val_input)
        f.create_dataset('val/target', data=val_target)
    
    print(f"\nSaved: data/turbulence3d_re1000_64x64x64.h5")
    print(f"  Train: {train_input.shape}")
    print(f"  Val: {val_input.shape}")
    print("\nThis is simplified data (linear diffusion only).")
    print("For full turbulence, need proper NS solver with nonlinear terms.")

if __name__ == '__main__':
    generate_quick_3d()

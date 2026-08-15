"""
Create minimal 3D data for pipeline verification
Uses synthetic Gaussian fields (not real turbulence but valid for testing)
"""

import numpy as np
import h5py
import os

def create_minimal_3d_data():
    """Create minimal 3D dataset"""
    print("Creating minimal 3D dataset for pipeline verification...")
    
    N = 64
    
    # Generate synthetic data with proper spectral properties
    def generate_field(N, seed):
        np.random.seed(seed)
        # Random Fourier coefficients
        u_hat = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
        
        # Apply spectrum E(k) ~ k^-4
        k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0,0,0] = 1.0
        
        u_hat *= k_mag**(-2)
        
        # Transform to physical space
        u = np.real(np.fft.ifftn(u_hat, axes=(0,1,2)))
        
        # Normalize
        u = u / np.std(u) * 0.1
        return u.astype(np.float32)
    
    # Generate train data (100 samples)
    print("  Generating 100 train samples...")
    train_input = []
    train_target = []
    
    for i in range(100):
        u = generate_field(N, seed=i)
        v = generate_field(N, seed=i+1000)
        w = generate_field(N, seed=i+2000)
        
        train_input.append(np.stack([u, v, w], axis=0))
        
        # Target = input + small evolution
        u2 = generate_field(N, seed=i+5000)
        v2 = generate_field(N, seed=i+6000)
        w2 = generate_field(N, seed=i+7000)
        
        train_target.append(np.stack([u2, v2, w2], axis=0))
    
    train_input = np.array(train_input)
    train_target = np.array(train_target)
    
    # Generate val data (20 samples)
    print("  Generating 20 val samples...")
    val_input = []
    val_target = []
    
    for i in range(20):
        u = generate_field(N, seed=10000+i)
        v = generate_field(N, seed=11000+i)
        w = generate_field(N, seed=12000+i)
        
        val_input.append(np.stack([u, v, w], axis=0))
        
        u2 = generate_field(N, seed=15000+i)
        v2 = generate_field(N, seed=16000+i)
        w2 = generate_field(N, seed=17000+i)
        
        val_target.append(np.stack([u2, v2, w2], axis=0))
    
    val_input = np.array(val_input)
    val_target = np.array(val_target)
    
    # Save
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/turbulence3d_re1000_64x64x64.h5', 'w') as f:
        f.create_dataset('train/input', data=train_input)
        f.create_dataset('train/target', data=train_target)
        f.create_dataset('val/input', data=val_input)
        f.create_dataset('val/target', data=val_target)
    
    print(f"\nSaved: data/turbulence3d_re1000_64x64x64.h5")
    print(f"  Train: {train_input.shape}")
    print(f"  Val: {val_input.shape}")
    print(f"  Data range: [{train_input.min():.4f}, {train_input.max():.4f}]")
    
    print("\nNOTE: This is synthetic Gaussian data for pipeline verification.")
    print("For real turbulence, use generate_3d_ns_turbulence.py (takes ~1 hour).")

if __name__ == '__main__':
    create_minimal_3d_data()

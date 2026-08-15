"""
Generate small 3D NS dataset for quick validation
Full dataset generation can run overnight
"""

import numpy as np
import h5py
import sys
import os

# Import the solver
from generate_3d_ns_turbulence import NS3DSolver

def generate_small_dataset():
    """Generate small dataset for quick testing"""
    print("="*70)
    print("Generating Small 3D NS Dataset (Quick Validation)")
    print("="*70)
    
    N = 64
    
    # Re=1000 - Small training set
    print("\nGenerating Re=1000 data (50 train, 10 val)...")
    solver = NS3DSolver(N=N, Re=1000, dt=0.001)
    
    # 1 trajectory for train
    print("  Train trajectory...")
    u, v, w = solver.generate_initial_condition(seed=0)
    
    # Burn-in
    for i in range(500):
        u, v, w = solver.step_rk4(u, v, w)
        if i % 100 == 0:
            E = solver.compute_energy(u, v, w)
            print(f"    Burn-in step {i}: Energy = {E:.4f}")
    
    # Collect 50 samples
    train_input = []
    train_target = []
    for i in range(50):
        train_input.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        
        # Evolve 10 steps
        for _ in range(10):
            u, v, w = solver.step_rk4(u, v, w)
        
        train_target.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        
        if i % 10 == 0:
            E = solver.compute_energy(u, v, w)
            div = solver.compute_divergence(
                solver.fft(train_input[-1][0]), 
                solver.fft(train_input[-1][1]), 
                solver.fft(train_input[-1][2])
            )
            print(f"    Sample {i}: Energy = {E:.4f}, Div = {div:.2e}")
    
    train_input = np.array(train_input, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    
    # 1 trajectory for val
    print("  Val trajectory...")
    u, v, w = solver.generate_initial_condition(seed=100)
    
    for _ in range(500):
        u, v, w = solver.step_rk4(u, v, w)
    
    val_input = []
    val_target = []
    for i in range(10):
        val_input.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        
        for _ in range(10):
            u, v, w = solver.step_rk4(u, v, w)
        
        val_target.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
    
    val_input = np.array(val_input, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    
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
    
    # Test sample statistics
    print("\n" + "="*70)
    print("Data Statistics")
    print("="*70)
    print(f"Train input range: [{train_input.min():.4f}, {train_input.max():.4f}]")
    print(f"Train target range: [{train_target.min():.4f}, {train_target.max():.4f}]")
    print(f"Mean velocity magnitude: {np.mean(np.abs(train_input)):.4f}")
    
    # Energy spectrum check
    u_hat = np.fft.fftn(train_input[0, 0], axes=(0,1,2))
    energy_spectrum = np.abs(u_hat)**2
    print(f"Energy at k=0: {energy_spectrum[0,0,0]:.4f}")
    print(f"Max energy at high k: {np.max(energy_spectrum[10:20, 10:20, 10:20]):.4f}")
    
    print("\n" + "="*70)
    print("SUCCESS: Small 3D NS dataset generated!")
    print("="*70)
    print("\nTo generate full dataset (1000 train, 200 val), run:")
    print("  python generate_3d_ns_turbulence.py")
    print("This will take ~30-60 minutes.")

if __name__ == '__main__':
    generate_small_dataset()

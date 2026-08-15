"""
Generate 3D Navier-Stokes turbulence data - Stable version
Using smaller time steps and dealiasing
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn, fftfreq
import sys

def compute_curl_3d(u_hat, v_hat, w_hat, kx, ky, kz):
    """Compute curl in Fourier space"""
    wx_hat = 1j * (ky * w_hat - kz * v_hat)
    wy_hat = 1j * (kz * u_hat - kx * w_hat)
    wz_hat = 1j * (kx * v_hat - ky * u_hat)
    return wx_hat, wy_hat, wz_hat

def project_divergence_free_3d(u_hat, v_hat, w_hat, kx, ky, kz, k2):
    """Project velocity field to divergence-free in 3D"""
    k_dot_u = kx * u_hat + ky * v_hat + kz * w_hat
    k2_safe = np.where(k2 == 0, 1.0, k2)
    u_hat_proj = u_hat - k_dot_u * kx / k2_safe
    v_hat_proj = v_hat - k_dot_u * ky / k2_safe
    w_hat_proj = w_hat - k_dot_u * kz / k2_safe
    return u_hat_proj, v_hat_proj, w_hat_proj

def apply_dealias_3d(u_hat, dealias_mask):
    """Apply 2/3 dealiasing rule"""
    return u_hat * dealias_mask

def step_euler_3d(u, v, w, nu, dt, kx, ky, kz, k2, dealias_mask):
    """Euler time stepping with dealiasing (more stable for data generation)"""
    u_hat = fftn(u, axes=(0,1,2))
    v_hat = fftn(v, axes=(0,1,2))
    w_hat = fftn(w, axes=(0,1,2))
    
    # Dealias
    u_hat = apply_dealias_3d(u_hat, dealias_mask)
    v_hat = apply_dealias_3d(v_hat, dealias_mask)
    w_hat = apply_dealias_3d(w_hat, dealias_mask)
    
    # Transform to physical space
    u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
    v_phys = np.real(ifftn(v_hat, axes=(0,1,2)))
    w_phys = np.real(ifftn(w_hat, axes=(0,1,2)))
    
    # Compute vorticity
    wx_hat, wy_hat, wz_hat = compute_curl_3d(u_hat, v_hat, w_hat, kx, ky, kz)
    wx = np.real(ifftn(wx_hat, axes=(0,1,2)))
    wy = np.real(ifftn(wy_hat, axes=(0,1,2)))
    wz = np.real(ifftn(wz_hat, axes=(0,1,2)))
    
    # Cross product u × ω (with clipping to prevent overflow)
    max_val = 1e3
    u_phys = np.clip(u_phys, -max_val, max_val)
    v_phys = np.clip(v_phys, -max_val, max_val)
    w_phys = np.clip(w_phys, -max_val, max_val)
    wx = np.clip(wx, -max_val, max_val)
    wy = np.clip(wy, -max_val, max_val)
    wz = np.clip(wz, -max_val, max_val)
    
    rhs_x_phys = v_phys * wz - w_phys * wy
    rhs_y_phys = w_phys * wx - u_phys * wz
    rhs_z_phys = u_phys * wy - v_phys * wx
    
    # Transform back to Fourier space
    rhs_x_hat = fftn(rhs_x_phys, axes=(0,1,2))
    rhs_y_hat = fftn(rhs_y_phys, axes=(0,1,2))
    rhs_z_hat = fftn(rhs_z_phys, axes=(0,1,2))
    
    # Dealias
    rhs_x_hat = apply_dealias_3d(rhs_x_hat, dealias_mask)
    rhs_y_hat = apply_dealias_3d(rhs_y_hat, dealias_mask)
    rhs_z_hat = apply_dealias_3d(rhs_z_hat, dealias_mask)
    
    # Add diffusion
    rhs_x_hat -= nu * k2 * u_hat
    rhs_y_hat -= nu * k2 * v_hat
    rhs_z_hat -= nu * k2 * w_hat
    
    # Update
    u_hat_new = u_hat + dt * rhs_x_hat
    v_hat_new = v_hat + dt * rhs_y_hat
    w_hat_new = w_hat + dt * rhs_z_hat
    
    # Project to divergence-free
    u_hat_new, v_hat_new, w_hat_new = project_divergence_free_3d(
        u_hat_new, v_hat_new, w_hat_new, kx, ky, kz, k2
    )
    
    # Transform back
    u_new = np.real(ifftn(u_hat_new, axes=(0,1,2)))
    v_new = np.real(ifftn(v_hat_new, axes=(0,1,2)))
    w_new = np.real(ifftn(w_hat_new, axes=(0,1,2)))
    
    return u_new, v_new, w_new

def generate_initial_condition_3d(N, seed=None):
    """Generate random divergence-free initial condition with energy spectrum"""
    if seed is not None:
        np.random.seed(seed)
    
    # Wavenumbers
    k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0,0,0] = 1.0
    
    # Random Fourier coefficients with energy spectrum E(k) ~ k^-4
    u_hat = (np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)) * k_mag**(-2)
    v_hat = (np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)) * k_mag**(-2)
    w_hat = (np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)) * k_mag**(-2)
    
    # Project to divergence-free
    k2 = k_mag**2
    k_dot_u = kx * u_hat + ky * v_hat + kz * w_hat
    u_hat -= k_dot_u * kx / k2
    v_hat -= k_dot_u * ky / k2
    w_hat -= k_dot_u * kz / k2
    
    # Transform to physical space
    u = np.real(ifftn(u_hat, axes=(0,1,2)))
    v = np.real(ifftn(v_hat, axes=(0,1,2)))
    w = np.real(ifftn(w_hat, axes=(0,1,2)))
    
    # Normalize energy to 1.0
    energy = 0.5 * np.mean(u**2 + v**2 + w**2)
    if energy > 0:
        scale = 1.0 / np.sqrt(energy)
        u *= scale
        v *= scale
        w *= scale
    
    return u, v, w

def generate_trajectory_3d(N=64, Re=1000, dt=0.001, n_steps=1000, seed=0):
    """Generate a single 3D trajectory"""
    # Setup
    k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    
    nu = 1.0 / Re
    
    # Dealiasing mask (2/3 rule)
    k_max = np.max(np.abs(k))
    dealias_mask = (np.abs(kx) < 2/3 * k_max) & (np.abs(ky) < 2/3 * k_max) & (np.abs(kz) < 2/3 * k_max)
    dealias_mask = dealias_mask.astype(float)
    
    # Initial condition
    u, v, w = generate_initial_condition_3d(N, seed=seed)
    
    # Burn-in to reach turbulence
    print(f"  Burn-in...")
    for i in range(500):
        u, v, w = step_euler_3d(u, v, w, nu, dt, kx, ky, kz, k2, dealias_mask)
        
        # Check for blow-up
        if not np.isfinite(u).all():
            print(f"  Blow-up at burn-in step {i}, restarting with smaller dt")
            return generate_trajectory_3d(N, Re, dt/2, n_steps, seed+1000)
    
    # Collect samples
    samples = []
    
    print(f"  Collecting {n_steps//10} samples...")
    for i in range(n_steps):
        if i % 10 == 0:
            samples.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        
        u, v, w = step_euler_3d(u, v, w, nu, dt, kx, ky, kz, k2, dealias_mask)
        
        if not np.isfinite(u).all():
            print(f"  Blow-up at step {i}, returning partial data")
            break
    
    return np.array(samples, dtype=np.float32)

def main():
    """Generate 3D turbulence datasets"""
    import os
    os.makedirs('data', exist_ok=True)
    
    N = 64
    
    # Re=1000 - Training data
    print("="*70)
    print("Generating Re=1000 training data")
    print("="*70)
    
    train_samples = []
    for i in range(20):  # 20 trajectories x 100 samples = 2000 samples
        print(f"Trajectory {i+1}/20")
        traj = generate_trajectory_3d(N, Re=1000, dt=0.001, n_steps=1000, seed=i)
        train_samples.append(traj)
    
    train_data = np.concatenate(train_samples, axis=0)[:1000]  # Take first 1000
    
    # Split into input/target pairs
    train_input = train_data[:-1]
    train_target = train_data[1:]
    
    # Re=1000 - Validation data
    print("\nGenerating Re=1000 validation data")
    val_samples = []
    for i in range(4):
        print(f"Trajectory {i+1}/4")
        traj = generate_trajectory_3d(N, Re=1000, dt=0.001, n_steps=1000, seed=100+i)
        val_samples.append(traj)
    
    val_data = np.concatenate(val_samples, axis=0)[:200]
    val_input = val_data[:-1]
    val_target = val_data[1:]
    
    # Save Re=1000
    with h5py.File('data/turbulence3d_re1000_64x64x64.h5', 'w') as f:
        f.create_dataset('train/input', data=train_input)
        f.create_dataset('train/target', data=train_target)
        f.create_dataset('val/input', data=val_input)
        f.create_dataset('val/target', data=val_target)
    
    print(f"\nSaved: data/turbulence3d_re1000_64x64x64.h5")
    print(f"  Train: {train_input.shape}")
    print(f"  Val: {val_input.shape}")
    
    # Re=5000 - OOD test data
    print("\n" + "="*70)
    print("Generating Re=5000 OOD data")
    print("="*70)
    
    ood_samples = []
    for i in range(4):
        print(f"Trajectory {i+1}/4")
        traj = generate_trajectory_3d(N, Re=5000, dt=0.0005, n_steps=1000, seed=200+i)
        ood_samples.append(traj)
    
    ood_data = np.concatenate(ood_samples, axis=0)[:200]
    ood_input = ood_data[:-1]
    ood_target = ood_data[1:]
    
    with h5py.File('data/turbulence3d_re5000_64x64x64.h5', 'w') as f:
        f.create_dataset('test/input', data=ood_input)
        f.create_dataset('test/target', data=ood_target)
    
    print(f"\nSaved: data/turbulence3d_re5000_64x64x64.h5")
    print(f"  Test: {ood_input.shape}")
    
    print("\n" + "="*70)
    print("3D data generation complete!")
    print("="*70)

if __name__ == '__main__':
    main()

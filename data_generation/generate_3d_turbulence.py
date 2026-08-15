"""
Generate 3D Navier-Stokes turbulence data using pseudo-spectral method
Resolution: 64^3
Reynolds numbers: 1000 (ID), 5000 (OOD)
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn, fftfreq
import sys

def compute_curl_3d(u_hat, v_hat, w_hat, kx, ky, kz):
    """Compute curl in Fourier space"""
    # ω_x = ∂w/∂y - ∂v/∂z = i*ky*w_hat - i*kz*v_hat
    wx_hat = 1j * (ky * w_hat - kz * v_hat)
    # ω_y = ∂u/∂z - ∂w/∂x = i*kz*u_hat - i*kx*w_hat  
    wy_hat = 1j * (kz * u_hat - kx * w_hat)
    # ω_z = ∂v/∂x - ∂u/∂y = i*kx*v_hat - i*ky*u_hat
    wz_hat = 1j * (kx * v_hat - ky * u_hat)
    return wx_hat, wy_hat, wz_hat

def project_divergence_free_3d(u_hat, v_hat, w_hat, kx, ky, kz, k2):
    """Project velocity field to divergence-free in 3D"""
    # k · u_hat
    k_dot_u = kx * u_hat + ky * v_hat + kz * w_hat
    
    # Remove divergence component
    k2_safe = np.where(k2 == 0, 1.0, k2)
    u_hat_proj = u_hat - k_dot_u * kx / k2_safe
    v_hat_proj = v_hat - k_dot_u * ky / k2_safe
    w_hat_proj = w_hat - k_dot_u * kz / k2_safe
    
    return u_hat_proj, v_hat_proj, w_hat_proj

def step_rk4_3d(u, v, w, nu, dt, kx, ky, kz, k2):
    """Runge-Kutta 4th order time stepping for 3D NS"""
    def rhs(u_hat, v_hat, w_hat):
        # Transform to physical space for nonlinear term
        u_phys = np.real(ifftn(u_hat, axes=(0,1,2)))
        v_phys = np.real(ifftn(v_hat, axes=(0,1,2)))
        w_phys = np.real(ifftn(w_hat, axes=(0,1,2)))
        
        # Compute vorticity (curl u)
        wx_hat, wy_hat, wz_hat = compute_curl_3d(u_hat, v_hat, w_hat, kx, ky, kz)
        wx = np.real(ifftn(wx_hat, axes=(0,1,2)))
        wy = np.real(ifftn(wy_hat, axes=(0,1,2)))
        wz = np.real(ifftn(wz_hat, axes=(0,1,2)))
        
        # Cross product u × ω in physical space
        rhs_x_phys = v_phys * wz - w_phys * wy
        rhs_y_phys = w_phys * wx - u_phys * wz
        rhs_z_phys = u_phys * wy - v_phys * wx
        
        # Transform back to Fourier space
        rhs_x_hat = fftn(rhs_x_phys, axes=(0,1,2))
        rhs_y_hat = fftn(rhs_y_phys, axes=(0,1,2))
        rhs_z_hat = fftn(rhs_z_phys, axes=(0,1,2))
        
        # Add diffusion term: -νk²u
        rhs_x_hat -= nu * k2 * u_hat
        rhs_y_hat -= nu * k2 * v_hat
        rhs_z_hat -= nu * k2 * w_hat
        
        return rhs_x_hat, rhs_y_hat, rhs_z_hat
    
    u_hat = fftn(u, axes=(0,1,2))
    v_hat = fftn(v, axes=(0,1,2))
    w_hat = fftn(w, axes=(0,1,2))
    
    # RK4
    k1_u, k1_v, k1_w = rhs(u_hat, v_hat, w_hat)
    
    u2 = u_hat + 0.5 * dt * k1_u
    v2 = v_hat + 0.5 * dt * k1_v
    w2 = w_hat + 0.5 * dt * k1_w
    u2, v2, w2 = project_divergence_free_3d(u2, v2, w2, kx, ky, kz, k2)
    k2_u, k2_v, k2_w = rhs(u2, v2, w2)
    
    u3 = u_hat + 0.5 * dt * k2_u
    v3 = v_hat + 0.5 * dt * k2_v
    w3 = w_hat + 0.5 * dt * k2_w
    u3, v3, w3 = project_divergence_free_3d(u3, v3, w3, kx, ky, kz, k2)
    k3_u, k3_v, k3_w = rhs(u3, v3, w3)
    
    u4 = u_hat + dt * k3_u
    v4 = v_hat + dt * k3_v
    w4 = w_hat + dt * k3_w
    u4, v4, w4 = project_divergence_free_3d(u4, v4, w4, kx, ky, kz, k2)
    k4_u, k4_v, k4_w = rhs(u4, v4, w4)
    
    # Update
    u_hat_new = u_hat + dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
    v_hat_new = v_hat + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    w_hat_new = w_hat + dt/6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)
    
    # Project to divergence-free
    u_hat_new, v_hat_new, w_hat_new = project_divergence_free_3d(
        u_hat_new, v_hat_new, w_hat_new, kx, ky, kz, k2
    )
    
    # Transform back to physical space
    u_new = np.real(ifftn(u_hat_new, axes=(0,1,2)))
    v_new = np.real(ifftn(v_hat_new, axes=(0,1,2)))
    w_new = np.real(ifftn(w_hat_new, axes=(0,1,2)))
    
    return u_new, v_new, w_new

def generate_initial_condition_3d(N, seed=None):
    """Generate random divergence-free initial condition"""
    if seed is not None:
        np.random.seed(seed)
    
    # Random Fourier coefficients
    u_hat = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
    v_hat = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
    w_hat = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
    
    # Scale by |k|^-2 for energy spectrum
    k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0,0,0] = 1.0  # Avoid division by zero
    
    u_hat /= k_mag**2
    v_hat /= k_mag**2
    w_hat /= k_mag**2
    
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
    
    # Normalize energy
    energy = 0.5 * np.mean(u**2 + v**2 + w**2)
    scale = 1.0 / np.sqrt(energy)
    u *= scale
    v *= scale
    w *= scale
    
    return u, v, w

def generate_3d_dataset(N=64, Re=1000, dt=0.01, n_steps=1000, n_samples=1000, split='train'):
    """Generate 3D turbulence dataset"""
    print(f"Generating 3D turbulence dataset: N={N}, Re={Re}, split={split}")
    
    # Setup wavenumbers
    k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    
    nu = 1.0 / Re
    
    inputs = []
    targets = []
    
    # Generate multiple trajectories
    n_trajectories = n_samples // 10  # Each trajectory provides ~10 samples
    
    for traj_idx in range(n_trajectories):
        print(f"  Trajectory {traj_idx+1}/{n_trajectories}")
        
        # Generate initial condition
        u, v, w = generate_initial_condition_3d(N, seed=traj_idx + (0 if split=='train' else 1000))
        
        # Burn-in to reach turbulence
        for _ in range(100):
            u, v, w = step_rk4_3d(u, v, w, nu, dt, kx, ky, kz, k2)
        
        # Collect samples
        for step in range(10):
            # Store input
            inputs.append(np.stack([u, v, w], axis=0))
            
            # Evolve to target
            for _ in range(10):  # 10 steps = dt=0.1
                u, v, w = step_rk4_3d(u, v, w, nu, dt, kx, ky, kz, k2)
            
            # Store target
            targets.append(np.stack([u, v, w], axis=0))
            
            if len(inputs) >= n_samples:
                break
        
        if len(inputs) >= n_samples:
            break
    
    inputs = np.array(inputs[:n_samples], dtype=np.float32)
    targets = np.array(targets[:n_samples], dtype=np.float32)
    
    return inputs, targets

def main():
    """Generate and save 3D turbulence datasets"""
    import os
    os.makedirs('data', exist_ok=True)
    
    # Re=1000 (ID)
    print("="*70)
    print("Generating Re=1000 dataset")
    print("="*70)
    
    train_input, train_target = generate_3d_dataset(
        N=64, Re=1000, n_samples=1000, split='train'
    )
    val_input, val_target = generate_3d_dataset(
        N=64, Re=1000, n_samples=200, split='val'
    )
    test_input, test_target = generate_3d_dataset(
        N=64, Re=1000, n_samples=200, split='test'
    )
    
    with h5py.File('data/turbulence3d_re1000_64x64x64.h5', 'w') as f:
        f.create_dataset('train/input', data=train_input)
        f.create_dataset('train/target', data=train_target)
        f.create_dataset('val/input', data=val_input)
        f.create_dataset('val/target', data=val_target)
        f.create_dataset('test/input', data=test_input)
        f.create_dataset('test/target', data=test_target)
    
    print("\nSaved: data/turbulence3d_re1000_64x64x64.h5")
    print(f"  Train: {train_input.shape}")
    print(f"  Val: {val_input.shape}")
    print(f"  Test: {test_input.shape}")
    
    # Re=5000 (OOD)
    print("\n" + "="*70)
    print("Generating Re=5000 dataset")
    print("="*70)
    
    test_input_ood, test_target_ood = generate_3d_dataset(
        N=64, Re=5000, n_samples=200, split='test'
    )
    
    with h5py.File('data/turbulence3d_re5000_64x64x64.h5', 'w') as f:
        f.create_dataset('test/input', data=test_input_ood)
        f.create_dataset('test/target', data=test_target_ood)
    
    print("\nSaved: data/turbulence3d_re5000_64x64x64.h5")
    print(f"  Test: {test_input_ood.shape}")
    
    print("\n" + "="*70)
    print("3D data generation complete!")
    print("="*70)

if __name__ == '__main__':
    main()

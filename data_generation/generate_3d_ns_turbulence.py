"""
Generate 3D Navier-Stokes turbulence using pseudo-spectral method
Full nonlinear terms with 2/3 dealiasing
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn, fftfreq
import sys
import os

class NS3DSolver:
    """3D Navier-Stokes solver using pseudo-spectral method"""
    
    def __init__(self, N=64, L=2*np.pi, Re=1000, dt=0.001):
        self.N = N
        self.L = L
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        
        # Wavenumbers
        k = np.fft.fftfreq(N, L/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0  # Avoid division by zero
        
        # Dealiasing mask (2/3 rule)
        k_max = np.max(np.abs(k))
        self.dealias_mask = (
            (np.abs(self.kx) < 2/3 * k_max) & 
            (np.abs(self.ky) < 2/3 * k_max) & 
            (np.abs(self.kz) < 2/3 * k_max)
        ).astype(float)
    
    def fft(self, u):
        """3D FFT with dealiasing"""
        return fftn(u, axes=(0,1,2)) * self.dealias_mask
    
    def ifft(self, u_hat):
        """3D inverse FFT"""
        return np.real(ifftn(u_hat, axes=(0,1,2)))
    
    def divergence_free_projection(self, u_hat, v_hat, w_hat):
        """Project to divergence-free space"""
        k_dot_u = self.kx * u_hat + self.ky * v_hat + self.kz * w_hat
        u_hat_proj = u_hat - k_dot_u * self.kx / self.k2
        v_hat_proj = v_hat - k_dot_u * self.ky / self.k2
        w_hat_proj = w_hat - k_dot_u * self.kz / self.k2
        return u_hat_proj, v_hat_proj, w_hat_proj
    
    def compute_nonlinear(self, u_hat, v_hat, w_hat):
        """Compute nonlinear term (u·∇)u using pseudospectral method"""
        # Transform to physical space
        u = self.ifft(u_hat)
        v = self.ifft(v_hat)
        w = self.ifft(w_hat)
        
        # Compute vorticity ω = ∇ × u
        wx_hat = 1j * (self.ky * w_hat - self.kz * v_hat)
        wy_hat = 1j * (self.kz * u_hat - self.kx * w_hat)
        wz_hat = 1j * (self.kx * v_hat - self.ky * u_hat)
        
        wx = self.ifft(wx_hat)
        wy = self.ifft(wy_hat)
        wz = self.ifft(wz_hat)
        
        # u × ω (Lamb vector)
        nl_x = v * wz - w * wy
        nl_y = w * wx - u * wz
        nl_z = u * wy - v * wx
        
        # Transform back
        nl_x_hat = self.fft(nl_x)
        nl_y_hat = self.fft(nl_y)
        nl_z_hat = self.fft(nl_z)
        
        return nl_x_hat, nl_y_hat, nl_z_hat
    
    def rhs(self, u_hat, v_hat, w_hat):
        """Right-hand side of NS equations"""
        # Nonlinear term
        nl_x_hat, nl_y_hat, nl_z_hat = self.compute_nonlinear(u_hat, v_hat, w_hat)
        
        # Diffusion term
        diff_x = -self.nu * self.k2 * u_hat
        diff_y = -self.nu * self.k2 * v_hat
        diff_z = -self.nu * self.k2 * w_hat
        
        # Total
        rhs_x = nl_x_hat + diff_x
        rhs_y = nl_y_hat + diff_y
        rhs_z = nl_z_hat + diff_z
        
        return rhs_x, rhs_y, rhs_z
    
    def step_rk4(self, u, v, w):
        """RK4 time stepping"""
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)
        
        # k1
        k1_u, k1_v, k1_w = self.rhs(u_hat, v_hat, w_hat)
        
        # k2
        u2 = u_hat + 0.5 * self.dt * k1_u
        v2 = v_hat + 0.5 * self.dt * k1_v
        w2 = w_hat + 0.5 * self.dt * k1_w
        u2, v2, w2 = self.divergence_free_projection(u2, v2, w2)
        k2_u, k2_v, k2_w = self.rhs(u2, v2, w2)
        
        # k3
        u3 = u_hat + 0.5 * self.dt * k2_u
        v3 = v_hat + 0.5 * self.dt * k2_v
        w3 = w_hat + 0.5 * self.dt * k2_w
        u3, v3, w3 = self.divergence_free_projection(u3, v3, w3)
        k3_u, k3_v, k3_w = self.rhs(u3, v3, w3)
        
        # k4
        u4 = u_hat + self.dt * k3_u
        v4 = v_hat + self.dt * k3_v
        w4 = w_hat + self.dt * k3_w
        u4, v4, w4 = self.divergence_free_projection(u4, v4, w4)
        k4_u, k4_v, k4_w = self.rhs(u4, v4, w4)
        
        # Update
        u_hat_new = u_hat + self.dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_hat_new = v_hat + self.dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        w_hat_new = w_hat + self.dt/6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        
        # Project to divergence-free
        u_hat_new, v_hat_new, w_hat_new = self.divergence_free_projection(
            u_hat_new, v_hat_new, w_hat_new
        )
        
        # Transform back
        u_new = self.ifft(u_hat_new)
        v_new = self.ifft(v_hat_new)
        w_new = self.ifft(w_hat_new)
        
        return u_new, v_new, w_new
    
    def generate_initial_condition(self, seed=None):
        """Generate random divergence-free initial condition"""
        if seed is not None:
            np.random.seed(seed)
        
        # Random Fourier coefficients with energy spectrum E(k) ~ k^-4
        u_hat = (np.random.randn(self.N, self.N, self.N) + 
                 1j * np.random.randn(self.N, self.N, self.N))
        v_hat = (np.random.randn(self.N, self.N, self.N) + 
                 1j * np.random.randn(self.N, self.N, self.N))
        w_hat = (np.random.randn(self.N, self.N, self.N) + 
                 1j * np.random.randn(self.N, self.N, self.N))
        
        # Apply energy spectrum
        k_mag = np.sqrt(self.k2)
        k_mag[0,0,0] = 1.0
        u_hat *= k_mag**(-2) * self.dealias_mask
        v_hat *= k_mag**(-2) * self.dealias_mask
        w_hat *= k_mag**(-2) * self.dealias_mask
        
        # Project to divergence-free
        u_hat, v_hat, w_hat = self.divergence_free_projection(u_hat, v_hat, w_hat)
        
        # Transform to physical space
        u = self.ifft(u_hat)
        v = self.ifft(v_hat)
        w = self.ifft(w_hat)
        
        # Normalize energy to 1.0
        energy = 0.5 * np.mean(u**2 + v**2 + w**2)
        if energy > 0:
            scale = 1.0 / np.sqrt(energy)
            u *= scale
            v *= scale
            w *= scale
        
        return u, v, w
    
    def compute_energy(self, u, v, w):
        """Compute kinetic energy"""
        return 0.5 * np.mean(u**2 + v**2 + w**2)
    
    def compute_divergence(self, u_hat, v_hat, w_hat):
        """Compute divergence (should be near zero)"""
        div_hat = 1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat)
        div = self.ifft(div_hat)
        return np.mean(div**2)


def generate_trajectory(solver, n_steps=1000, n_burnin=500, seed=0):
    """Generate a single trajectory"""
    print(f"    Generating trajectory (seed={seed})...")
    
    # Initial condition
    u, v, w = solver.generate_initial_condition(seed=seed)
    
    # Burn-in to reach turbulence
    print(f"    Burn-in ({n_burnin} steps)...")
    for i in range(n_burnin):
        u, v, w = solver.step_rk4(u, v, w)
        
        # Check for blow-up
        if not np.isfinite(u).all():
            print(f"    WARNING: Blow-up at burn-in step {i}, restarting...")
            return generate_trajectory(solver, n_steps, n_burnin, seed+1000)
    
    # Collect samples
    samples = []
    print(f"    Collecting {n_steps//10} samples...")
    
    for i in range(n_steps):
        if i % 10 == 0:
            samples.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        
        u, v, w = solver.step_rk4(u, v, w)
        
        if not np.isfinite(u).all():
            print(f"    WARNING: Blow-up at step {i}, returning partial data")
            break
    
    return np.array(samples, dtype=np.float32)


def main():
    """Generate 3D NS turbulence datasets"""
    os.makedirs('data', exist_ok=True)
    
    N = 64
    
    # Re=1000 (ID)
    print("="*70)
    print("Generating Re=1000 dataset")
    print("="*70)
    
    solver = NS3DSolver(N=N, Re=1000, dt=0.001)
    
    # Training data: 10 trajectories x 100 samples = 1000 samples
    train_samples = []
    for i in range(10):
        print(f"  Trajectory {i+1}/10")
        traj = generate_trajectory(solver, n_steps=1000, n_burnin=500, seed=i)
        train_samples.append(traj)
        print(f"    Collected {len(traj)} samples")
    
    train_data = np.concatenate(train_samples, axis=0)
    # Split into input/target pairs (t -> t+dt)
    train_input = train_data[:-1]
    train_target = train_data[1:]
    
    # Validation data: 2 trajectories x 100 samples = 200 samples
    print("\nGenerating validation data...")
    val_samples = []
    for i in range(2):
        print(f"  Trajectory {i+1}/2")
        traj = generate_trajectory(solver, n_steps=1000, n_burnin=500, seed=100+i)
        val_samples.append(traj)
    
    val_data = np.concatenate(val_samples, axis=0)
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
    
    # Re=5000 (OOD)
    print("\n" + "="*70)
    print("Generating Re=5000 OOD dataset")
    print("="*70)
    
    solver_ood = NS3DSolver(N=N, Re=5000, dt=0.0005)  # Smaller dt for stability
    
    ood_samples = []
    for i in range(2):
        print(f"  Trajectory {i+1}/2")
        traj = generate_trajectory(solver_ood, n_steps=1000, n_burnin=800, seed=200+i)
        ood_samples.append(traj)
    
    ood_data = np.concatenate(ood_samples, axis=0)
    ood_input = ood_data[:-1]
    ood_target = ood_data[1:]
    
    with h5py.File('data/turbulence3d_re5000_64x64x64.h5', 'w') as f:
        f.create_dataset('test/input', data=ood_input)
        f.create_dataset('test/target', data=ood_target)
    
    print(f"\nSaved: data/turbulence3d_re5000_64x64x64.h5")
    print(f"  Test: {ood_input.shape}")
    
    print("\n" + "="*70)
    print("3D NS turbulence data generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()

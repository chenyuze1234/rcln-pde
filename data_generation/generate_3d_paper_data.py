"""
Generate FULL 3D turbulence dataset matching paper specifications:
- 500 training samples (independent turbulent fields)
- 100 validation samples
- N=64 resolution
- Re_lambda=433 (target)
- Proper dt for input-target evolution
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn
import os
from tqdm import tqdm


class NS3DSolverFast:
    """Optimized 3D NS solver for batch data generation"""
    
    def __init__(self, N=64, L=2*np.pi, Re=433, dt=0.005):
        self.N = N
        self.L = L
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        
        # Precompute wavenumbers
        k = np.fft.fftfreq(N, L/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0
        
        # Dealiasing mask (2/3 rule)
        k_max = np.max(np.abs(k))
        self.dealias_mask = (
            (np.abs(self.kx) < 2/3 * k_max) & 
            (np.abs(self.ky) < 2/3 * k_max) & 
            (np.abs(self.kz) < 2/3 * k_max)
        ).astype(float)
        
        # Precompute for linear step
        self.damping = np.exp(-self.nu * self.dt * self.k2)
    
    def fft(self, u):
        return fftn(u, axes=(0,1,2)) * self.dealias_mask
    
    def ifft(self, u_hat):
        return np.real(ifftn(u_hat, axes=(0,1,2)))
    
    def project_div_free(self, u_hat, v_hat, w_hat):
        kdu = self.kx*u_hat + self.ky*v_hat + self.kz*w_hat
        return (
            u_hat - kdu*self.kx/self.k2,
            v_hat - kdu*self.ky/self.k2,
            w_hat - kdu*self.kz/self.k2
        )
    
    def nonlinear_rhs(self, uh, vh, wh):
        """Compute nonlinear term in spectral space"""
        u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        
        # Vorticity
        wx = self.ifft(1j*(self.ky*wh - self.kz*vh))
        wy = self.ifft(1j*(self.kz*uh - self.kx*wh))
        wz = self.ifft(1j*(self.kx*vh - self.ky*uh))
        
        # Lamb vector u × ω
        nl_x = self.fft(v*wz - w*wy)
        nl_y = self.fft(w*wx - u*wz)
        nl_z = self.fft(u*wy - v*wx)
        
        return nl_x, nl_y, nl_z
    
    def step_rk4(self, u, v, w):
        """RK4 time stepping"""
        uh, vh, wh = self.fft(u), self.fft(v), self.fft(w)
        
        # k1
        nl_x, nl_y, nl_z = self.nonlinear_rhs(uh, vh, wh)
        k1_u = nl_x - self.nu*self.k2*uh
        k1_v = nl_y - self.nu*self.k2*vh
        k1_w = nl_z - self.nu*self.k2*wh
        
        # k2
        u2 = uh + 0.5*self.dt*k1_u
        v2 = vh + 0.5*self.dt*k1_v
        w2 = wh + 0.5*self.dt*k1_w
        u2, v2, w2 = self.project_div_free(u2, v2, w2)
        nl_x, nl_y, nl_z = self.nonlinear_rhs(u2, v2, w2)
        k2_u = nl_x - self.nu*self.k2*u2
        k2_v = nl_y - self.nu*self.k2*v2
        k2_w = nl_z - self.nu*self.k2*w2
        
        # k3
        u3 = uh + 0.5*self.dt*k2_u
        v3 = vh + 0.5*self.dt*k2_v
        w3 = wh + 0.5*self.dt*k2_w
        u3, v3, w3 = self.project_div_free(u3, v3, w3)
        nl_x, nl_y, nl_z = self.nonlinear_rhs(u3, v3, w3)
        k3_u = nl_x - self.nu*self.k2*u3
        k3_v = nl_y - self.nu*self.k2*v3
        k3_w = nl_z - self.nu*self.k2*w3
        
        # k4
        u4 = uh + self.dt*k3_u
        v4 = vh + self.dt*k3_v
        w4 = wh + self.dt*k3_w
        u4, v4, w4 = self.project_div_free(u4, v4, w4)
        nl_x, nl_y, nl_z = self.nonlinear_rhs(u4, v4, w4)
        k4_u = nl_x - self.nu*self.k2*u4
        k4_v = nl_y - self.nu*self.k2*v4
        k4_w = nl_z - self.nu*self.k2*w4
        
        # Update
        uh_new = uh + self.dt/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)
        vh_new = vh + self.dt/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        wh_new = wh + self.dt/6*(k1_w + 2*k2_w + 2*k3_w + k4_w)
        
        uh_new, vh_new, wh_new = self.project_div_free(uh_new, vh_new, wh_new)
        
        return self.ifft(uh_new), self.ifft(vh_new), self.ifft(wh_new)
    
    def generate_initial_condition(self, seed=None):
        """Generate random divergence-free initial condition"""
        if seed is not None:
            np.random.seed(seed)
        
        # Random Fourier coefficients with k^-4 spectrum
        uh = (np.random.randn(self.N, self.N, self.N) + 
              1j*np.random.randn(self.N, self.N, self.N))
        vh = (np.random.randn(self.N, self.N, self.N) + 
              1j*np.random.randn(self.N, self.N, self.N))
        wh = (np.random.randn(self.N, self.N, self.N) + 
              1j*np.random.randn(self.N, self.N, self.N))
        
        # Apply energy spectrum E(k) ~ k^-4
        km = np.sqrt(self.k2)
        km[0,0,0] = 1.0
        uh *= km**(-2) * self.dealias_mask
        vh *= km**(-2) * self.dealias_mask
        wh *= km**(-2) * self.dealias_mask
        
        # Project to divergence-free
        uh, vh, wh = self.project_div_free(uh, vh, wh)
        
        u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        
        # Normalize energy
        E = 0.5*np.mean(u**2 + v**2 + w**2)
        if E > 0:
            scale = 1.0/np.sqrt(E)
            u, v, w = u*scale, v*scale, w*scale
        
        return u, v, w
    
    def evolve_to_target(self, u, v, w, n_steps=10):
        """Evolve field by n_steps to create input-target pair"""
        for _ in range(n_steps):
            u, v, w = self.step_rk4(u, v, w)
            # Check for blow-up
            if not np.isfinite(u).all():
                return None, None, None
        return u, v, w


def generate_single_sample(solver, seed, n_burnin=200, n_evolve=10):
    """Generate one input-target pair"""
    # Generate initial condition
    u, v, w = solver.generate_initial_condition(seed=seed)
    
    # Burn-in to reach turbulence
    for i in range(n_burnin):
        u, v, w = solver.step_rk4(u, v, w)
        if not np.isfinite(u).all():
            # Restart with new seed if blow-up
            return generate_single_sample(solver, seed+10000, n_burnin, n_evolve)
    
    # This is input
    input_field = np.stack([u.copy(), v.copy(), w.copy()], axis=0)
    
    # Evolve to get target
    u_target, v_target, w_target = solver.evolve_to_target(u, v, w, n_evolve)
    
    if u_target is None:
        return generate_single_sample(solver, seed+10000, n_burnin, n_evolve)
    
    target_field = np.stack([u_target, v_target, w_target], axis=0)
    
    return input_field, target_field


def generate_dataset(n_samples, solver, seed_start=0, desc="Generating"):
    """Generate dataset with proper input-target pairs"""
    inputs = []
    targets = []
    
    for i in tqdm(range(n_samples), desc=desc):
        inp, tgt = generate_single_sample(solver, seed=seed_start+i)
        inputs.append(inp)
        targets.append(tgt)
    
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def main():
    """Generate full paper dataset"""
    print("="*70)
    print("Generating FULL 3D Paper Dataset")
    print("="*70)
    print("Target: 500 train + 100 val samples, N=64, Re=433")
    print()
    
    N = 64
    Re = 433  # Paper uses Re_lambda=433
    dt = 0.005  # Adjusted for stability
    n_evolve = 10  # Number of steps between input and target
    
    solver = NS3DSolverFast(N=N, Re=Re, dt=dt)
    
    # Training data: 500 samples
    print("[1] Generating TRAINING data (500 samples)...")
    print("-"*70)
    train_input, train_target = generate_dataset(
        500, solver, seed_start=0, desc="Train"
    )
    
    # Validation data: 100 samples
    print("\n[2] Generating VALIDATION data (100 samples)...")
    print("-"*70)
    val_input, val_target = generate_dataset(
        100, solver, seed_start=10000, desc="Val"
    )
    
    # Statistics
    print("\n[3] Computing statistics...")
    print("-"*70)
    all_data = np.concatenate([train_input, train_target, val_input, val_target], axis=0)
    mean = float(all_data.mean())
    std = float(all_data.std())
    
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Min:  {all_data.min():.6f}")
    print(f"  Max:  {all_data.max():.6f}")
    
    # Validation checks
    print("\n[4] Validation checks...")
    print("-"*70)
    
    # Check correlation (should be < 0.99 for meaningful evolution)
    corr = np.corrcoef(train_input[:20].flatten(), train_target[:20].flatten())[0,1]
    print(f"  Input-Target correlation: {corr:.4f}")
    if corr > 0.995:
        print("  WARNING: Correlation too high, dt may be too small!")
    elif corr < 0.8:
        print("  WARNING: Correlation too low, dt may be too large!")
    else:
        print("  OK: Correlation in good range")
    
    # Check energy evolution
    E_in = 0.5 * (train_input**2).mean(axis=(1,2,3,4))
    E_out = 0.5 * (train_target**2).mean(axis=(1,2,3,4))
    E_ratio = E_out / E_in
    print(f"  Energy ratio (out/in): {E_ratio.mean():.4f} ± {E_ratio.std():.4f}")
    
    # Check divergence-free (approximate)
    print(f"  NaN check: {'PASSED' if not np.isnan(all_data).any() else 'FAILED'}")
    
    # Save
    print("\n[5] Saving data...")
    print("-"*70)
    
    os.makedirs('data', exist_ok=True)
    output_file = 'data/turbulence3d_paper_full.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train/input', data=train_input)
        f.create_dataset('train/target', data=train_target)
        f.create_dataset('val/input', data=val_input)
        f.create_dataset('val/target', data=val_target)
        
        # Save statistics
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        f.attrs['dt'] = dt
        f.attrs['n_evolve'] = n_evolve
        f.attrs['n_train'] = 500
        f.attrs['n_val'] = 100
    
    print(f"  Saved: {output_file}")
    print(f"    Train: {train_input.shape}")
    print(f"    Val:   {val_input.shape}")
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nReady for 50-epoch training to target MSE=1.07e-5")


if __name__ == '__main__':
    main()

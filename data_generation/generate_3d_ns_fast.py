"""
Generate CORRECT 3D NS data - OPTIMIZED VERSION
- Reduced samples for faster generation
- Still uses proper pseudo-spectral method with time evolution
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn
import os


class NS3DSolver:
    """Optimized 3D NS solver"""
    
    def __init__(self, N=64, L=2*np.pi, Re=1000, dt=0.001):
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
        
        # Dealiasing mask
        k_max = np.max(np.abs(k))
        self.dealias_mask = (
            (np.abs(self.kx) < 2/3 * k_max) & 
            (np.abs(self.ky) < 2/3 * k_max) & 
            (np.abs(self.kz) < 2/3 * k_max)
        ).astype(float)
    
    def fft(self, u):
        return fftn(u, axes=(0,1,2)) * self.dealias_mask
    
    def ifft(self, u_hat):
        return np.real(ifftn(u_hat, axes=(0,1,2)))
    
    def project_div_free(self, u_hat, v_hat, w_hat):
        k_dot_u = self.kx * u_hat + self.ky * v_hat + self.kz * w_hat
        return (
            u_hat - k_dot_u * self.kx / self.k2,
            v_hat - k_dot_u * self.ky / self.k2,
            w_hat - k_dot_u * self.kz / self.k2
        )
    
    def compute_rhs(self, u_hat, v_hat, w_hat):
        # Nonlinear term (rotational form)
        u = self.ifft(u_hat)
        v = self.ifft(v_hat)
        w = self.ifft(w_hat)
        
        # Vorticity
        wx = self.ifft(1j * (self.ky * w_hat - self.kz * v_hat))
        wy = self.ifft(1j * (self.kz * u_hat - self.kx * w_hat))
        wz = self.ifft(1j * (self.kx * v_hat - self.ky * u_hat))
        
        # u × ω
        nl_x = self.fft(v * wz - w * wy)
        nl_y = self.fft(w * wx - u * wz)
        nl_z = self.fft(u * wy - v * wx)
        
        # Diffusion
        diff_x = -self.nu * self.k2 * u_hat
        diff_y = -self.nu * self.k2 * v_hat
        diff_z = -self.nu * self.k2 * w_hat
        
        return nl_x + diff_x, nl_y + diff_y, nl_z + diff_z
    
    def step(self, u, v, w):
        """Single RK4 step"""
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)
        
        # RK4
        k1_u, k1_v, k1_w = self.compute_rhs(u_hat, v_hat, w_hat)
        
        u2 = u_hat + 0.5*self.dt*k1_u
        v2 = v_hat + 0.5*self.dt*k1_v
        w2 = w_hat + 0.5*self.dt*k1_w
        u2, v2, w2 = self.project_div_free(u2, v2, w2)
        k2_u, k2_v, k2_w = self.compute_rhs(u2, v2, w2)
        
        u3 = u_hat + 0.5*self.dt*k2_u
        v3 = v_hat + 0.5*self.dt*k2_v
        w3 = w_hat + 0.5*self.dt*k2_w
        u3, v3, w3 = self.project_div_free(u3, v3, w3)
        k3_u, k3_v, k3_w = self.compute_rhs(u3, v3, w3)
        
        u4 = u_hat + self.dt*k3_u
        v4 = v_hat + self.dt*k3_v
        w4 = w_hat + self.dt*k3_w
        u4, v4, w4 = self.project_div_free(u4, v4, w4)
        k4_u, k4_v, k4_w = self.compute_rhs(u4, v4, w4)
        
        # Update
        u_hat_new = u_hat + self.dt/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_hat_new = v_hat + self.dt/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        w_hat_new = w_hat + self.dt/6*(k1_w + 2*k2_w + 2*k3_w + k4_w)
        
        u_hat_new, v_hat_new, w_hat_new = self.project_div_free(u_hat_new, v_hat_new, w_hat_new)
        
        return self.ifft(u_hat_new), self.ifft(v_hat_new), self.ifft(w_hat_new)
    
    def init_cond(self, seed=None):
        """Generate initial condition"""
        if seed is not None:
            np.random.seed(seed)
        
        u_hat = (np.random.randn(self.N, self.N, self.N) + 
                 1j*np.random.randn(self.N, self.N, self.N))
        v_hat = (np.random.randn(self.N, self.N, self.N) + 
                 1j*np.random.randn(self.N, self.N, self.N))
        w_hat = (np.random.randn(self.N, self.N, self.N) + 
                 1j*np.random.randn(self.N, self.N, self.N))
        
        k_mag = np.sqrt(self.k2)
        k_mag[0,0,0] = 1.0
        u_hat *= k_mag**(-2) * self.dealias_mask
        v_hat *= k_mag**(-2) * self.dealias_mask
        w_hat *= k_mag**(-2) * self.dealias_mask
        
        u_hat, v_hat, w_hat = self.project_div_free(u_hat, v_hat, w_hat)
        
        u, v, w = self.ifft(u_hat), self.ifft(v_hat), self.ifft(w_hat)
        
        # Normalize
        E = 0.5*np.mean(u**2 + v**2 + w**2)
        if E > 0:
            scale = 1.0/np.sqrt(E)
            u, v, w = u*scale, v*scale, w*scale
        
        return u, v, w


def generate_data():
    """Generate optimized dataset"""
    print("Generating FAST 3D NS data...")
    
    N = 64
    Re = 1000
    solver = NS3DSolver(N=N, Re=Re, dt=0.001)
    
    # Small dataset for quick iteration
    # But still proper time evolution!
    n_train = 120  # 120 samples
    n_val = 30     # 30 samples
    
    print(f"\nGenerating {n_train} train + {n_val} val samples...")
    
    # Generate single long trajectory for training
    print("\n[1] Train trajectory...")
    u, v, w = solver.init_cond(seed=42)
    
    # Short burn-in
    print("    Burn-in...")
    for _ in range(100):
        u, v, w = solver.step(u, v, w)
    
    # Collect samples
    train_samples = []
    print("    Collecting...")
    for i in range(n_train + 1):  # +1 for target
        train_samples.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        u, v, w = solver.step(u, v, w)
        if i % 30 == 0:
            print(f"      Step {i}/{n_train+1}")
    
    train_samples = np.array(train_samples, dtype=np.float32)
    train_input = train_samples[:-1]
    train_target = train_samples[1:]
    
    # Generate validation trajectory (different seed)
    print("\n[2] Val trajectory...")
    u, v, w = solver.init_cond(seed=123)
    
    for _ in range(150):
        u, v, w = solver.step(u, v, w)
    
    val_samples = []
    for i in range(n_val + 1):
        val_samples.append(np.stack([u.copy(), v.copy(), w.copy()], axis=0))
        u, v, w = solver.step(u, v, w)
    
    val_samples = np.array(val_samples, dtype=np.float32)
    val_input = val_samples[:-1]
    val_target = val_samples[1:]
    
    # Statistics
    print("\n[3] Computing statistics...")
    all_data = np.concatenate([train_input, train_target, val_input, val_target], axis=0)
    mean = all_data.mean()
    std = all_data.std()
    
    print(f"    Mean: {mean:.4f}")
    print(f"    Std:  {std:.4f}")
    
    # Validation
    print("\n[4] Validation...")
    corr = np.corrcoef(train_input[:10].flatten(), train_target[:10].flatten())[0,1]
    print(f"    Correlation: {corr:.4f} (should be > 0.9)")
    
    E_in = 0.5 * (train_input**2).mean(axis=(1,2,3,4))
    E_out = 0.5 * (train_target**2).mean(axis=(1,2,3,4))
    print(f"    Energy ratio: {(E_out/E_in).mean():.4f} (should be ~1.0)")
    
    # Save
    print("\n[5] Saving...")
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/turbulence3d_re1000_correct.h5', 'w') as f:
        f.create_dataset('train/input', data=train_input)
        f.create_dataset('train/target', data=train_target)
        f.create_dataset('val/input', data=val_input)
        f.create_dataset('val/target', data=val_target)
        f.attrs['mean'] = mean
        f.attrs['std'] = std
    
    print(f"\nSaved: data/turbulence3d_re1000_correct.h5")
    print(f"  Train: {train_input.shape}")
    print(f"  Val:   {val_input.shape}")
    print("\n✓ Data is ready for training with standard normalization!")


if __name__ == '__main__':
    generate_data()

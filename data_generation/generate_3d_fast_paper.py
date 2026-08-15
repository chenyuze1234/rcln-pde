"""
FAST version for validation - 100 train + 20 val samples
Reduced burn-in (100 steps vs 200) for quicker generation
Still N=64, proper pseudo-spectral method
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn
import os
from tqdm import tqdm


class NS3DSolverFast:
    def __init__(self, N=64, L=2*np.pi, Re=433, dt=0.005):
        self.N = N
        self.L = L
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        
        k = np.fft.fftfreq(N, L/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0
        
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
        kdu = self.kx*u_hat + self.ky*v_hat + self.kz*w_hat
        return (
            u_hat - kdu*self.kx/self.k2,
            v_hat - kdu*self.ky/self.k2,
            w_hat - kdu*self.kz/self.k2
        )
    
    def step_rk4(self, u, v, w):
        uh, vh, wh = self.fft(u), self.fft(v), self.fft(w)
        
        # Nonlinear
        u_p, v_p, w_p = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        wx = self.ifft(1j*(self.ky*wh - self.kz*vh))
        wy = self.ifft(1j*(self.kz*uh - self.kx*wh))
        wz = self.ifft(1j*(self.kx*vh - self.ky*uh))
        
        nl_x = self.fft(v_p*wz - w_p*wy)
        nl_y = self.fft(w_p*wx - u_p*wz)
        nl_z = self.fft(u_p*wy - v_p*wx)
        
        k1_u = nl_x - self.nu*self.k2*uh
        k1_v = nl_y - self.nu*self.k2*vh
        k1_w = nl_z - self.nu*self.k2*wh
        
        u2 = uh + 0.5*self.dt*k1_u
        v2 = vh + 0.5*self.dt*k1_v
        w2 = wh + 0.5*self.dt*k1_w
        u2, v2, w2 = self.project_div_free(u2, v2, w2)
        
        u_p, v_p, w_p = self.ifft(u2), self.ifft(v2), self.ifft(w2)
        wx = self.ifft(1j*(self.ky*w2 - self.kz*v2))
        wy = self.ifft(1j*(self.kz*u2 - self.kx*w2))
        wz = self.ifft(1j*(self.kx*v2 - self.ky*u2))
        nl_x = self.fft(v_p*wz - w_p*wy)
        nl_y = self.fft(w_p*wx - u_p*wz)
        nl_z = self.fft(u_p*wy - v_p*wx)
        k2_u = nl_x - self.nu*self.k2*u2
        k2_v = nl_y - self.nu*self.k2*v2
        k2_w = nl_z - self.nu*self.k2*w2
        
        u3 = uh + 0.5*self.dt*k2_u
        v3 = vh + 0.5*self.dt*k2_v
        w3 = wh + 0.5*self.dt*k2_w
        u3, v3, w3 = self.project_div_free(u3, v3, w3)
        
        u_p, v_p, w_p = self.ifft(u3), self.ifft(v3), self.ifft(w3)
        wx = self.ifft(1j*(self.ky*w3 - self.kz*v3))
        wy = self.ifft(1j*(self.kz*u3 - self.kx*w3))
        wz = self.ifft(1j*(self.kx*v3 - self.ky*u3))
        nl_x = self.fft(v_p*wz - w_p*wy)
        nl_y = self.fft(w_p*wx - u_p*wz)
        nl_z = self.fft(u_p*wy - v_p*wx)
        k3_u = nl_x - self.nu*self.k2*u3
        k3_v = nl_y - self.nu*self.k2*v3
        k3_w = nl_z - self.nu*self.k2*w3
        
        u4 = uh + self.dt*k3_u
        v4 = vh + self.dt*k3_v
        w4 = wh + self.dt*k3_w
        u4, v4, w4 = self.project_div_free(u4, v4, w4)
        
        u_p, v_p, w_p = self.ifft(u4), self.ifft(v4), self.ifft(w4)
        wx = self.ifft(1j*(self.ky*w4 - self.kz*v4))
        wy = self.ifft(1j*(self.kz*u4 - self.kx*w4))
        wz = self.ifft(1j*(self.kx*v4 - self.ky*u4))
        nl_x = self.fft(v_p*wz - w_p*wy)
        nl_y = self.fft(w_p*wx - u_p*wz)
        nl_z = self.fft(u_p*wy - v_p*wx)
        k4_u = nl_x - self.nu*self.k2*u4
        k4_v = nl_y - self.nu*self.k2*v4
        k4_w = nl_z - self.nu*self.k2*w4
        
        uh_new = uh + self.dt/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)
        vh_new = vh + self.dt/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        wh_new = wh + self.dt/6*(k1_w + 2*k2_w + 2*k3_w + k4_w)
        
        uh_new, vh_new, wh_new = self.project_div_free(uh_new, vh_new, wh_new)
        return self.ifft(uh_new), self.ifft(vh_new), self.ifft(wh_new)
    
    def generate_ic(self, seed=None):
        if seed: np.random.seed(seed)
        uh = (np.random.randn(self.N,self.N,self.N) + 
              1j*np.random.randn(self.N,self.N,self.N))
        vh = (np.random.randn(self.N,self.N,self.N) + 
              1j*np.random.randn(self.N,self.N,self.N))
        wh = (np.random.randn(self.N,self.N,self.N) + 
              1j*np.random.randn(self.N,self.N,self.N))
        
        km = np.sqrt(self.k2)
        km[0,0,0] = 1.0
        uh *= km**(-2) * self.dealias_mask
        vh *= km**(-2) * self.dealias_mask
        wh *= km**(-2) * self.dealias_mask
        
        uh, vh, wh = self.project_div_free(uh, vh, wh)
        u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        
        E = 0.5*np.mean(u**2+v**2+w**2)
        if E > 0:
            s = 1.0/np.sqrt(E)
            u, v, w = u*s, v*s, w*s
        return u, v, w


def generate_sample(solver, seed, n_burnin=100, n_evolve=10):
    """Generate one input-target pair"""
    u, v, w = solver.generate_ic(seed=seed)
    
    # Burn-in
    for _ in range(n_burnin):
        u, v, w = solver.step_rk4(u, v, w)
        if not np.isfinite(u).all():
            return generate_sample(solver, seed+10000, n_burnin, n_evolve)
    
    input_field = np.stack([u.copy(), v.copy(), w.copy()], axis=0)
    
    # Evolve
    for _ in range(n_evolve):
        u, v, w = solver.step_rk4(u, v, w)
        if not np.isfinite(u).all():
            return generate_sample(solver, seed+10000, n_burnin, n_evolve)
    
    target_field = np.stack([u, v, w], axis=0)
    return input_field, target_field


def main():
    print("="*70)
    print("FAST Paper Data Generation (100 train + 20 val)")
    print("="*70)
    
    N, Re, dt = 64, 433, 0.005
    solver = NS3DSolverFast(N=N, Re=Re, dt=dt)
    
    # 100 train samples
    print("\n[1] Training data (100 samples)...")
    train_in, train_tgt = [], []
    for i in tqdm(range(100)):
        inp, tgt = generate_sample(solver, seed=i, n_burnin=100, n_evolve=10)
        train_in.append(inp)
        train_tgt.append(tgt)
    train_in = np.array(train_in, dtype=np.float32)
    train_tgt = np.array(train_tgt, dtype=np.float32)
    
    # 20 val samples
    print("\n[2] Validation data (20 samples)...")
    val_in, val_tgt = [], []
    for i in tqdm(range(20)):
        inp, tgt = generate_sample(solver, seed=10000+i, n_burnin=100, n_evolve=10)
        val_in.append(inp)
        val_tgt.append(tgt)
    val_in = np.array(val_in, dtype=np.float32)
    val_tgt = np.array(val_tgt, dtype=np.float32)
    
    # Stats
    all_data = np.concatenate([train_in, train_tgt, val_in, val_tgt], axis=0)
    mean, std = all_data.mean(), all_data.std()
    
    print(f"\n[3] Statistics: mean={mean:.4f}, std={std:.4f}")
    
    # Validation
    corr = np.corrcoef(train_in[:20].flatten(), train_tgt[:20].flatten())[0,1]
    print(f"    Correlation: {corr:.4f}")
    
    # Save
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/turbulence3d_paper_fast.h5', 'w') as f:
        f.create_dataset('train/input', data=train_in)
        f.create_dataset('train/target', data=train_tgt)
        f.create_dataset('val/input', data=val_in)
        f.create_dataset('val/target', data=val_tgt)
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['N'] = N
        f.attrs['Re'] = Re
    
    print(f"\nSaved: data/turbulence3d_paper_fast.h5")
    print(f"  Train: {train_in.shape}")
    print(f"  Val: {val_in.shape}")

if __name__ == '__main__':
    main()

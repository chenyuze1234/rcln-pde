"""
Generate minimal 3D NS data for quick testing
Uses small N=32 to speed up FFT
Still maintains correct physics
"""

import numpy as np
import h5py
from scipy.fft import fftn, ifftn
import os


class NS3DSolver:
    def __init__(self, N=32, L=2*np.pi, Re=1000, dt=0.001):
        self.N = N
        self.nu = 1.0 / Re
        self.dt = dt
        
        k = np.fft.fftfreq(N, L/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0
        
        k_max = np.max(np.abs(k))
        mask = (
            (np.abs(self.kx) < 2/3 * k_max) & 
            (np.abs(self.ky) < 2/3 * k_max) & 
            (np.abs(self.kz) < 2/3 * k_max)
        )
        self.dealias_mask = mask.astype(float)
    
    def fft(self, u):
        return fftn(u, axes=(0,1,2)) * self.dealias_mask
    
    def ifft(self, u_hat):
        return np.real(ifftn(u_hat, axes=(0,1,2)))
    
    def project(self, u_hat, v_hat, w_hat):
        kdu = self.kx*u_hat + self.ky*v_hat + self.kz*w_hat
        return (
            u_hat - kdu*self.kx/self.k2,
            v_hat - kdu*self.ky/self.k2,
            w_hat - kdu*self.kz/self.k2
        )
    
    def rhs(self, uh, vh, wh):
        u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        wx = self.ifft(1j*(self.ky*wh - self.kz*vh))
        wy = self.ifft(1j*(self.kz*uh - self.kx*wh))
        wz = self.ifft(1j*(self.kx*vh - self.ky*uh))
        
        nl_x = self.fft(v*wz - w*wy)
        nl_y = self.fft(w*wx - u*wz)
        nl_z = self.fft(u*wy - v*wx)
        
        return (
            nl_x - self.nu*self.k2*uh,
            nl_y - self.nu*self.k2*vh,
            nl_z - self.nu*self.k2*wh
        )
    
    def step(self, u, v, w):
        uh, vh, wh = self.fft(u), self.fft(v), self.fft(w)
        
        k1 = self.rhs(uh, vh, wh)
        
        u2 = uh + 0.5*self.dt*k1[0]
        v2 = vh + 0.5*self.dt*k1[1]
        w2 = wh + 0.5*self.dt*k1[2]
        u2, v2, w2 = self.project(u2, v2, w2)
        k2 = self.rhs(u2, v2, w2)
        
        u3 = uh + 0.5*self.dt*k2[0]
        v3 = vh + 0.5*self.dt*k2[1]
        w3 = wh + 0.5*self.dt*k2[2]
        u3, v3, w3 = self.project(u3, v3, w3)
        k3 = self.rhs(u3, v3, w3)
        
        u4 = uh + self.dt*k3[0]
        v4 = vh + self.dt*k3[1]
        w4 = wh + self.dt*k3[2]
        u4, v4, w4 = self.project(u4, v4, w4)
        k4 = self.rhs(u4, v4, w4)
        
        uh_new = uh + self.dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        vh_new = vh + self.dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        wh_new = wh + self.dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        
        uh_new, vh_new, wh_new = self.project(uh_new, vh_new, wh_new)
        return self.ifft(uh_new), self.ifft(vh_new), self.ifft(wh_new)
    
    def init(self, seed=None):
        if seed: np.random.seed(seed)
        uh = np.random.randn(self.N,self.N,self.N) + 1j*np.random.randn(self.N,self.N,self.N)
        vh = np.random.randn(self.N,self.N,self.N) + 1j*np.random.randn(self.N,self.N,self.N)
        wh = np.random.randn(self.N,self.N,self.N) + 1j*np.random.randn(self.N,self.N,self.N)
        
        km = np.sqrt(self.k2)
        km[0,0,0] = 1.0
        uh *= km**(-2) * self.dealias_mask
        vh *= km**(-2) * self.dealias_mask
        wh *= km**(-2) * self.dealias_mask
        
        uh, vh, wh = self.project(uh, vh, wh)
        u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        
        E = 0.5*np.mean(u**2+v**2+w**2)
        if E > 0:
            s = 1.0/np.sqrt(E)
            u, v, w = u*s, v*s, w*s
        return u, v, w


def main():
    print("Generating MINIMAL 3D NS data (N=32)...")
    
    N = 32
    solver = NS3DSolver(N=N, Re=1000, dt=0.001)
    
    # Minimal dataset
    n_train, n_val = 50, 10
    
    # Train
    print("Train...")
    u, v, w = solver.init(seed=42)
    for _ in range(50): u, v, w = solver.step(u, v, w)
    
    train = []
    for _ in range(n_train + 1):
        train.append(np.stack([u.copy(), v.copy(), w.copy()], 0))
        u, v, w = solver.step(u, v, w)
    train = np.array(train, dtype=np.float32)
    
    # Val
    print("Val...")
    u, v, w = solver.init(seed=123)
    for _ in range(60): u, v, w = solver.step(u, v, w)
    
    val = []
    for _ in range(n_val + 1):
        val.append(np.stack([u.copy(), v.copy(), w.copy()], 0))
        u, v, w = solver.step(u, v, w)
    val = np.array(val, dtype=np.float32)
    
    # Stats
    all_data = np.concatenate([train, val], 0)
    mean, std = all_data.mean(), all_data.std()
    
    # Save
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/turbulence3d_re1000_correct.h5', 'w') as f:
        f.create_dataset('train/input', data=train[:-1])
        f.create_dataset('train/target', data=train[1:])
        f.create_dataset('val/input', data=val[:-1])
        f.create_dataset('val/target', data=val[1:])
        f.attrs['mean'] = mean
        f.attrs['std'] = std
    
    print(f"\nSaved: {train[:-1].shape}, {val[:-1].shape}")
    print(f"Stats: mean={mean:.4f}, std={std:.4f}")
    
    corr = np.corrcoef(train[:-1][:5].flatten(), train[1:][:5].flatten())[0,1]
    print(f"Correlation: {corr:.4f}")

if __name__ == '__main__':
    main()

"""
MINI version - 20 train + 5 val for quick validation
"""
import numpy as np
import h5py
from scipy.fft import fftn, ifftn
import os

class Solver:
    def __init__(self, N=64, Re=433, dt=0.005):
        self.N = N
        self.nu = 1.0/Re
        self.dt = dt
        k = np.fft.fftfreq(N, 2*np.pi/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0
        kmax = np.max(np.abs(k))
        self.mask = ((np.abs(self.kx) < 2/3*kmax) & 
                     (np.abs(self.ky) < 2/3*kmax) & 
                     (np.abs(self.kz) < 2/3*kmax)).astype(float)
    
    def fft(self, u):
        return fftn(u, axes=(0,1,2)) * self.mask
    def ifft(self, uh):
        return np.real(ifftn(uh, axes=(0,1,2)))
    def proj(self, uh, vh, wh):
        kdu = self.kx*uh + self.ky*vh + self.kz*wh
        return uh-kdu*self.kx/self.k2, vh-kdu*self.ky/self.k2, wh-kdu*self.kz/self.k2
    
    def step(self, u, v, w):
        uh, vh, wh = self.fft(u), self.fft(v), self.fft(w)
        u1, v1, w1 = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        wx = self.ifft(1j*(self.ky*wh - self.kz*vh))
        wy = self.ifft(1j*(self.kz*uh - self.kx*wh))
        wz = self.ifft(1j*(self.kx*vh - self.ky*uh))
        nl_x = self.fft(v1*wz - w1*wy)
        nl_y = self.fft(w1*wx - u1*wz)
        nl_z = self.fft(u1*wy - v1*wx)
        k1_u = nl_x - self.nu*self.k2*uh
        k1_v = nl_y - self.nu*self.k2*vh
        k1_w = nl_z - self.nu*self.k2*wh
        
        u2, v2, w2 = self.proj(uh+0.5*self.dt*k1_u, vh+0.5*self.dt*k1_v, wh+0.5*self.dt*k1_w)
        u1, v1, w1 = self.ifft(u2), self.ifft(v2), self.ifft(w2)
        wx = self.ifft(1j*(self.ky*w2 - self.kz*v2))
        wy = self.ifft(1j*(self.kz*u2 - self.kx*w2))
        wz = self.ifft(1j*(self.kx*v2 - self.ky*u2))
        nl_x = self.fft(v1*wz - w1*wy)
        nl_y = self.fft(w1*wx - u1*wz)
        nl_z = self.fft(u1*wy - v1*wx)
        k2_u = nl_x - self.nu*self.k2*u2
        k2_v = nl_y - self.nu*self.k2*v2
        k2_w = nl_z - self.nu*self.k2*w2
        
        u3, v3, w3 = self.proj(uh+0.5*self.dt*k2_u, vh+0.5*self.dt*k2_v, wh+0.5*self.dt*k2_w)
        u1, v1, w1 = self.ifft(u3), self.ifft(v3), self.ifft(w3)
        wx = self.ifft(1j*(self.ky*w3 - self.kz*v3))
        wy = self.ifft(1j*(self.kz*u3 - self.kx*w3))
        wz = self.ifft(1j*(self.kx*v3 - self.ky*u3))
        nl_x = self.fft(v1*wz - w1*wy)
        nl_y = self.fft(w1*wx - u1*wz)
        nl_z = self.fft(u1*wy - v1*wx)
        k3_u = nl_x - self.nu*self.k2*u3
        k3_v = nl_y - self.nu*self.k2*v3
        k3_w = nl_z - self.nu*self.k2*w3
        
        u4, v4, w4 = self.proj(uh+self.dt*k3_u, vh+self.dt*k3_v, wh+self.dt*k3_w)
        u1, v1, w1 = self.ifft(u4), self.ifft(v4), self.ifft(w4)
        wx = self.ifft(1j*(self.ky*w4 - self.kz*v4))
        wy = self.ifft(1j*(self.kz*u4 - self.kx*w4))
        wz = self.ifft(1j*(self.kx*v4 - self.ky*u4))
        nl_x = self.fft(v1*wz - w1*wy)
        nl_y = self.fft(w1*wx - u1*wz)
        nl_z = self.fft(u1*wy - v1*wx)
        k4_u = nl_x - self.nu*self.k2*u4
        k4_v = nl_y - self.nu*self.k2*v4
        k4_w = nl_z - self.nu*self.k2*w4
        
        un = uh + self.dt/6*(k1_u+2*k2_u+2*k3_u+k4_u)
        vn = vh + self.dt/6*(k1_v+2*k2_v+2*k3_v+k4_v)
        wn = wh + self.dt/6*(k1_w+2*k2_w+2*k3_w+k4_w)
        un, vn, wn = self.proj(un, vn, wn)
        return self.ifft(un), self.ifft(vn), self.ifft(wn)
    
    def ic(self, seed=None):
        if seed: np.random.seed(seed)
        uh = np.random.randn(self.N,self.N,self.N) + 1j*np.random.randn(self.N,self.N,self.N)
        vh = np.random.randn(self.N,self.N,self.N) + 1j*np.random.randn(self.N,self.N,self.N)
        wh = np.random.randn(self.N,self.N,self.N) + 1j*np.random.randn(self.N,self.N,self.N)
        km = np.sqrt(self.k2)
        km[0,0,0] = 1.0
        uh *= km**(-2) * self.mask
        vh *= km**(-2) * self.mask
        wh *= km**(-2) * self.mask
        uh, vh, wh = self.proj(uh, vh, wh)
        u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
        E = 0.5*np.mean(u**2+v**2+w**2)
        if E > 0:
            s = 1.0/np.sqrt(E)
            u, v, w = u*s, v*s, w*s
        return u, v, w

def gen(solver, seed, burn=50, evolve=10):
    u, v, w = solver.ic(seed=seed)
    for _ in range(burn):
        u, v, w = solver.step(u, v, w)
        if not np.isfinite(u).all():
            return gen(solver, seed+10000, burn, evolve)
    inp = np.stack([u.copy(), v.copy(), w.copy()], 0)
    for _ in range(evolve):
        u, v, w = solver.step(u, v, w)
        if not np.isfinite(u).all():
            return gen(solver, seed+10000, burn, evolve)
    tgt = np.stack([u, v, w], 0)
    return inp, tgt

def main():
    print("Generating MINI dataset (20 train + 5 val)...")
    solver = Solver(N=64, Re=433, dt=0.005)
    
    train_in, train_tgt = [], []
    for i in range(20):
        if i % 5 == 0: print(f"  Train {i}/20")
        inp, tgt = gen(solver, seed=i, burn=50, evolve=10)
        train_in.append(inp)
        train_tgt.append(tgt)
    
    val_in, val_tgt = [], []
    for i in range(5):
        print(f"  Val {i}/5")
        inp, tgt = gen(solver, seed=10000+i, burn=50, evolve=10)
        val_in.append(inp)
        val_tgt.append(tgt)
    
    train_in = np.array(train_in, dtype=np.float32)
    train_tgt = np.array(train_tgt, dtype=np.float32)
    val_in = np.array(val_in, dtype=np.float32)
    val_tgt = np.array(val_tgt, dtype=np.float32)
    
    all_data = np.concatenate([train_in, train_tgt, val_in, val_tgt], 0)
    mean, std = all_data.mean(), all_data.std()
    
    print(f"\nStats: mean={mean:.4f}, std={std:.4f}")
    corr = np.corrcoef(train_in.flatten(), train_tgt.flatten())[0,1]
    print(f"Correlation: {corr:.4f}")
    
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/turbulence3d_paper_mini.h5', 'w') as f:
        f.create_dataset('train/input', data=train_in)
        f.create_dataset('train/target', data=train_tgt)
        f.create_dataset('val/input', data=val_in)
        f.create_dataset('val/target', data=val_tgt)
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['N'] = 64
    
    print(f"Saved: data/turbulence3d_paper_mini.h5")

if __name__ == '__main__':
    main()

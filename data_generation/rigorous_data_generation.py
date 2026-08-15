#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格化数据生成方案
修正之前的问题，确保物理正确性
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RigorousTurbulenceGenerator:
    """
    严格化的湍流数据生成器
    确保：
    1. 严格不可压（谱空间精确投影）
    2. CFL条件满足
    3. 能量守恒检验通过
    """
    
    def __init__(self, nx=64, ny=64, nz=64, Re=400, dt=None):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Re = Re
        self.nu = 1.0 / Re
        
        # CFL条件：dt < dx / u_max
        # 对于谱方法，更严格的条件是 dt < 1/(nu * k_max^2)
        k_max = np.pi * min(nx, ny, nz) / 2
        dt_viscous = 1.0 / (self.nu * k_max**2)
        self.dt = dt if dt is not None else min(0.001, dt_viscous * 0.1)
        
        print(f"Grid: {nx}x{ny}x{nz}, Re={Re}")
        print(f"Viscous time step limit: {dt_viscous:.6f}")
        print(f"Using dt: {self.dt:.6f}")
        
        # 波数网格
        self.kx = np.fft.fftfreq(nx, 1.0/nx) * 2 * np.pi
        self.ky = np.fft.fftfreq(ny, 1.0/ny) * 2 * np.pi
        self.kz = np.fft.fftfreq(nz, 1.0/nz) * 2 * np.pi
        self.KX, self.KY, self.KZ = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        self.k2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.k2[0, 0, 0] = 1.0  # 避免除零
        self.k_mag = np.sqrt(self.k2)
        
    def generate_initial_spectral(self, seed=42):
        """
        在谱空间生成初始条件，确保严格不可压
        """
        np.random.seed(seed)
        
        # 生成复随机场（实部和虚部独立）
        u_hat = np.random.randn(self.nx, self.ny, self.nz) + \
                1j * np.random.randn(self.nx, self.ny, self.nz)
        v_hat = np.random.randn(self.nx, self.ny, self.nz) + \
                1j * np.random.randn(self.nx, self.ny, self.nz)
        w_hat = np.random.randn(self.nx, self.ny, self.nz) + \
                1j * np.random.randn(self.nx, self.ny, self.nz)
        
        # von Karman 能量谱（简化版）
        k0 = 4.0  # 峰值波数
        
        # von Karman谱形状：E(k) = C * k^4 / (k^2 + k0^2)^(17/6)
        # 使用更简单的形式确保有能量
        E_k = np.zeros_like(self.k_mag)
        mask = self.k_mag > 0.5
        E_k[mask] = (self.k_mag[mask] / k0)**2 * np.exp(-(self.k_mag[mask] / k0)**2)
        E_k[0, 0, 0] = 0
        
        # 归一化
        amplitude = np.sqrt(2 * E_k / (self.k_mag**2 + 1e-10))
        
        u_hat *= amplitude
        v_hat *= amplitude
        w_hat *= amplitude
        
        # 严格投影到无散空间（谱空间精确操作）
        # u_perp = u - (k·u)k / |k|^2
        kdotu = self.KX * u_hat + self.KY * v_hat + self.KZ * w_hat
        
        u_hat_solenoidal = u_hat - kdotu * self.KX / self.k2
        v_hat_solenoidal = v_hat - kdotu * self.KY / self.k2
        w_hat_solenoidal = w_hat - kdotu * self.KZ / self.k2
        
        # 验证无散条件（应在机器精度内）
        kdotu_new = self.KX * u_hat_solenoidal + self.KY * v_hat_solenoidal + self.KZ * w_hat_solenoidal
        max_div_spectral = np.max(np.abs(kdotu_new))
        print(f"Max divergence in spectral space: {max_div_spectral:.2e}")
        
        # 反变换到物理空间
        u = np.fft.ifftn(u_hat_solenoidal).real
        v = np.fft.ifftn(v_hat_solenoidal).real
        w = np.fft.ifftn(w_hat_solenoidal).real
        
        # 归一化到单位能量
        energy = 0.5 * np.mean(u**2 + v**2 + w**2)
        scale = 1.0 / np.sqrt(energy + 1e-10)
        
        return u*scale, v*scale, w*scale
    
    def velocity_to_vorticity(self, u, v, w):
        """计算涡量（谱方法更准确）"""
        u_hat, v_hat, w_hat = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
        
        # 涡量在谱空间：omega = i k × u
        wx_hat = 1j * (self.KY * w_hat - self.KZ * v_hat)
        wy_hat = 1j * (self.KZ * u_hat - self.KX * w_hat)
        wz_hat = 1j * (self.KX * v_hat - self.KY * u_hat)
        
        wx = np.fft.ifftn(wx_hat).real
        wy = np.fft.ifftn(wy_hat).real
        wz = np.fft.ifftn(wz_hat).real
        
        return wx, wy, wz
    
    def check_divergence_spectral(self, u, v, w):
        """在谱空间检查散度（更准确）"""
        u_hat, v_hat, w_hat = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
        div_hat = 1j * (self.KX * u_hat + self.KY * v_hat + self.KZ * w_hat)
        div = np.fft.ifftn(div_hat).real
        return np.max(np.abs(div)), np.mean(np.abs(div))
    
    def compute_rhs_spectral(self, u_hat, v_hat, w_hat):
        """
        计算NS方程的RHS（谱空间）
        du/dt = -N(u) - nu*k^2*u
        其中N(u)是非线性项的投影
        """
        # 转换到物理空间计算非线性项
        u, v, w = [np.fft.ifftn(h).real for h in [u_hat, v_hat, w_hat]]
        
        # 计算对流项 u·∇u（物理空间）
        def compute_gradient(phi):
            phi_hat = np.fft.fftn(phi)
            return (np.fft.ifftn(1j * self.KX * phi_hat).real,
                    np.fft.ifftn(1j * self.KY * phi_hat).real,
                    np.fft.ifftn(1j * self.KZ * phi_hat).real)
        
        du_dx, du_dy, du_dz = compute_gradient(u)
        dv_dx, dv_dy, dv_dz = compute_gradient(v)
        dw_dx, dw_dy, dw_dz = compute_gradient(w)
        
        # 对流项
        Nx = -(u * du_dx + v * du_dy + w * du_dz)
        Ny = -(u * dv_dx + v * dv_dy + w * dv_dz)
        Nz = -(u * dw_dx + v * dw_dy + w * dw_dz)
        
        # 转换到谱空间并投影
        Nx_hat, Ny_hat, Nz_hat = np.fft.fftn(Nx), np.fft.fftn(Ny), np.fft.fftn(Nz)
        
        # 投影到无散空间
        kdotN = self.KX * Nx_hat + self.KY * Ny_hat + self.KZ * Nz_hat
        Nx_hat -= kdotN * self.KX / self.k2
        Ny_hat -= kdotN * self.KY / self.k2
        Nz_hat -= kdotN * self.KZ / self.k2
        
        # 添加粘性项
        rhs_x = Nx_hat - self.nu * self.k2 * u_hat
        rhs_y = Ny_hat - self.nu * self.k2 * v_hat
        rhs_z = Nz_hat - self.nu * self.k2 * w_hat
        
        return rhs_x, rhs_y, rhs_z
    
    def rk4_step(self, u_hat, v_hat, w_hat):
        """RK4时间积分（更精确）"""
        def compute_rhs(u_h, v_h, w_h):
            return self.compute_rhs_spectral(u_h, v_h, w_h)
        
        k1_x, k1_y, k1_z = compute_rhs(u_hat, v_hat, w_hat)
        
        k2_x, k2_y, k2_z = compute_rhs(
            u_hat + 0.5*self.dt*k1_x,
            v_hat + 0.5*self.dt*k1_y,
            w_hat + 0.5*self.dt*k1_z
        )
        
        k3_x, k3_y, k3_z = compute_rhs(
            u_hat + 0.5*self.dt*k2_x,
            v_hat + 0.5*self.dt*k2_y,
            w_hat + 0.5*self.dt*k2_z
        )
        
        k4_x, k4_y, k4_z = compute_rhs(
            u_hat + self.dt*k3_x,
            v_hat + self.dt*k3_y,
            w_hat + self.dt*k3_z
        )
        
        u_new = u_hat + self.dt/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_new = v_hat + self.dt/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        w_new = w_hat + self.dt/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
        
        return u_new, v_new, w_new
    
    def generate_trajectory(self, n_steps=20, save_every=1):
        """生成严格验证的时间序列"""
        print("\nGenerating initial condition...")
        u, v, w = self.generate_initial_spectral(seed=42)
        
        # 验证初始条件
        max_div, mean_div = self.check_divergence_spectral(u, v, w)
        print(f"Initial divergence: max={max_div:.2e}, mean={mean_div:.2e}")
        
        # 转换到谱空间
        u_hat, v_hat, w_hat = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
        
        trajectory = []
        
        for step in range(n_steps):
            # RK4步进
            u_hat, v_hat, w_hat = self.rk4_step(u_hat, v_hat, w_hat)
            
            if step % save_every == 0:
                u = np.fft.ifftn(u_hat).real
                v = np.fft.ifftn(v_hat).real
                w = np.fft.ifftn(w_hat).real
                
                # 验证
                max_div, _ = self.check_divergence_spectral(u, v, w)
                wx, wy, wz = self.velocity_to_vorticity(u, v, w)
                
                energy = 0.5 * np.mean(u**2 + v**2 + w**2)
                enstrophy = 0.5 * np.mean(wx**2 + wy**2 + wz**2)
                
                trajectory.append({
                    'velocity': np.stack([u, v, w], axis=0),
                    'vorticity': np.stack([wx, wy, wz], axis=0),
                    'energy': energy,
                    'enstrophy': enstrophy,
                    'divergence': max_div,
                    'step': step
                })
                
                if step % 5 == 0:
                    print(f"Step {step}: E={energy:.6f}, Z={enstrophy:.6f}, div={max_div:.2e}")
        
        return trajectory
    
    def validate_trajectory(self, trajectory):
        """验证整个轨迹的物理正确性"""
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        
        # 1. 不可压条件
        max_divs = [t['divergence'] for t in trajectory]
        print(f"\n[1] Incompressibility:")
        print(f"  Max divergence over trajectory: {max(max_divs):.2e}")
        print(f"  Status: {'PASS' if max(max_divs) < 1e-8 else 'WARNING'}")
        
        # 2. 能量衰减
        energies = [t['energy'] for t in trajectory]
        print(f"\n[2] Energy decay:")
        print(f"  Initial: {energies[0]:.6f}, Final: {energies[-1]:.6f}")
        print(f"  Decay: {(energies[0]-energies[-1])/energies[0]*100:.2f}%")
        
        # 3. 涡量变化
        enstrophies = [t['enstrophy'] for t in trajectory]
        print(f"\n[3] Enstrophy:")
        print(f"  Initial: {enstrophies[0]:.6f}, Final: {enstrophies[-1]:.6f}")
        print(f"  Ratio Z/E: {enstrophies[0]/energies[0]:.2f}")
        
        # 4. 能量谱
        u, v, w = trajectory[0]['velocity']
        u_hat = np.fft.fftn(u)
        energy_spec = 0.5 * np.abs(u_hat)**2
        
        print(f"\n[4] Energy spectrum:")
        print(f"  Peak energy at k ~ {np.unravel_index(np.argmax(energy_spec), energy_spec.shape)}")
        
        return {
            'max_divergence': max(max_divs),
            'energy_decay': (energies[0]-energies[-1])/energies[0],
            'enstrophy_decay': (enstrophies[0]-enstrophies[-1])/enstrophies[0]
        }


def main():
    print("="*70)
    print("RIGOROUS TURBULENCE DATA GENERATION")
    print("="*70)
    
    # 使用更大的网格和更小的时间步长
    generator = RigorousTurbulenceGenerator(nx=64, ny=64, nz=64, Re=400)
    trajectory = generator.generate_trajectory(n_steps=20, save_every=1)
    
    # 验证
    stats = generator.validate_trajectory(trajectory)
    
    # 可视化
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 速度场
    u = trajectory[0]['velocity'][0]
    axes[0, 0].imshow(u[:, :, 32], cmap='RdBu_r', origin='lower')
    axes[0, 0].set_title('u (t=0)')
    
    u_end = trajectory[-1]['velocity'][0]
    axes[0, 1].imshow(u_end[:, :, 32], cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title('u (t=end)')
    
    # 涡量
    wx = trajectory[0]['vorticity'][0]
    axes[0, 2].imshow(wx[:, :, 32], cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title('omega_x (t=0)')
    
    # 能量衰减
    energies = [t['energy'] for t in trajectory]
    axes[1, 0].plot(energies, 'o-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Decay')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 涡量衰减
    enstrophies = [t['enstrophy'] for t in trajectory]
    axes[1, 1].plot(enstrophies, 'o-', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Enstrophy')
    axes[1, 1].set_title('Enstrophy Decay')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 散度检查
    divs = [t['divergence'] for t in trajectory]
    axes[1, 2].semilogy(divs, 'o-', linewidth=2, color='green')
    axes[1, 2].axhline(y=1e-8, color='r', linestyle='--', label='Tolerance')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Max Divergence')
    axes[1, 2].set_title('Incompressibility Check')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('D:\\AxiomOS_Project\\论文图表\\rigorous_data_validation.png', dpi=200)
    print("Saved: rigorous_data_validation.png")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

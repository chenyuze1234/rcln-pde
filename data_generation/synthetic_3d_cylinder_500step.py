"""
Synthetic 3D Cylinder Wake Data Generation with Strong Noise
Strict Physics: Spectral NS Solver + Helmholtz Projection
Domain: 3D Cylinder Wake (x,y,z) = (flow, cross-span, spanwise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Using device: {device}")

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    # Grid
    nx, ny, nz = 128, 64, 32  # x: streamwise, y: cross-stream, z: spanwise
    lx, ly, lz = 16.0, 8.0, 4.0  # Domain size
    
    # Cylinder
    cylinder_x, cylinder_y = 2.0, 4.0  # Cylinder center
    cylinder_d = 1.0  # Diameter
    
    # Physics
    reynolds = 5000
    nu = 1.0 / reynolds
    dt = 0.01
    
    # Noise
    noise_level = 0.15  # 15% strong noise
    noise_correlation_length = 0.5  # Spatial correlation
    
    # Simulation
    n_steps = 500
    save_interval = 50
    
    # Wake parameters (empirical fit to real wakes)
    wake_velocity_deficit = 0.3
    wake_spread_rate = 0.15
    vortex_shedding_freq = 0.2  # Strouhal-based

config = Config()

# ==============================================================================
# 3D Spectral Navier-Stokes Solver with Cylinder
# ==============================================================================
class CylinderWake3DSolver:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.setup_grid()
        self.setup_cylinder()
        self.setup_spectral_operators()
        
    def setup_grid(self):
        cfg = self.cfg
        # Physical grid
        self.x = torch.linspace(0, cfg.lx, cfg.nx, device=device)
        self.y = torch.linspace(0, cfg.ly, cfg.ny, device=device)
        self.z = torch.linspace(0, cfg.lz, cfg.nz, device=device)
        self.X, self.Y, self.Z = torch.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Wavenumbers for FFT
        self.kx = 2 * np.pi * torch.fft.fftfreq(cfg.nx, cfg.lx/cfg.nx, device=device)
        self.ky = 2 * np.pi * torch.fft.fftfreq(cfg.ny, cfg.ly/cfg.ny, device=device)
        self.kz = 2 * np.pi * torch.fft.fftfreq(cfg.nz, cfg.lz/cfg.nz, device=device)
        
        self.Kx, self.Ky, self.Kz = torch.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        self.k_sq = self.Kx**2 + self.Ky**2 + self.Kz**2
        self.k_sq[0, 0, 0] = 1.0  # Avoid division by zero
        
        print(f"[GRID] Domain: {cfg.nx}x{cfg.ny}x{cfg.nz}")
        print(f"[GRID] Physical: {cfg.lx:.1f}x{cfg.ly:.1f}x{cfg.lz:.1f}")
        
    def setup_cylinder(self):
        cfg = self.cfg
        # Cylinder mask (infinite in z-direction)
        r = torch.sqrt((self.X - cfg.cylinder_x)**2 + (self.Y - cfg.cylinder_y)**2)
        self.cylinder_mask = (r <= cfg.cylinder_d/2).float()
        
        # Cylinder surface normal
        self.nx_cyl = torch.where(r > 0, (self.X - cfg.cylinder_x) / r, 0)
        self.ny_cyl = torch.where(r > 0, (self.Y - cfg.cylinder_y) / r, 0)
        
        print(f"[CYLINDER] D={cfg.cylinder_d}, center=({cfg.cylinder_x},{cfg.cylinder_y})")
        print(f"[CYLINDER] Masked points: {self.cylinder_mask.sum().item()}")
        
    def setup_spectral_operators(self):
        # Dealiasing mask (2/3 rule)
        cfg = self.cfg
        kx_max = cfg.nx // 3
        ky_max = cfg.ny // 3
        kz_max = cfg.nz // 3
        
        self.dealias = (
            (torch.abs(torch.arange(cfg.nx, device=device)[:,None,None]) < kx_max) &
            (torch.abs(torch.arange(cfg.ny, device=device)[None,:,None]) < ky_max) &
            (torch.abs(torch.arange(cfg.nz, device=device)[None,None,:]) < kz_max)
        ).float()
        
    def apply_cylinder_bc(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply no-slip boundary condition on cylinder surface"""
        mask = self.cylinder_mask
        
        # No-slip: velocity = 0 on surface
        u = u * (1 - mask)
        v = v * (1 - mask)
        w = w * (1 - mask)
        
        return u, v, w
    
    def inflow_bc(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform inflow with optional perturbation"""
        cfg = self.cfg
        # Uniform flow at inlet (x=0)
        u[0, :, :] = 1.0
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0
        return u, v, w
    
    def outflow_bc(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convective outflow boundary condition"""
        cfg = self.cfg
        # Zero gradient (Neumann) with convection velocity
        u_out = u[-2, :, :].clone()
        v_out = v[-2, :, :].clone()
        w_out = w[-2, :, :].clone()
        
        u[-1, :, :] = u_out
        v[-1, :, :] = v_out
        w[-1, :, :] = w_out
        
        return u, v, w
    
    def helmholtz_projection(self, u_hat: torch.Tensor, v_hat: torch.Tensor, 
                            w_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project velocity to divergence-free space"""
        # Divergence in Fourier space
        div_hat = self.Kx * u_hat + self.Ky * v_hat + self.Kz * w_hat
        
        # Projection: u = u - k*(k·u)/|k|²
        u_hat_proj = u_hat - div_hat * self.Kx / self.k_sq
        v_hat_proj = v_hat - div_hat * self.Ky / self.k_sq
        w_hat_proj = w_hat - div_hat * self.Kz / self.k_sq
        
        return u_hat_proj, v_hat_proj, w_hat_proj
    
    def compute_nonlinear(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute nonlinear term -∇·(uu) in spectral space"""
        # Transform to physical space
        u_x = torch.fft.ifftn(u, dim=(0,1,2)).real
        u_y = torch.fft.ifftn(v, dim=(0,1,2)).real
        u_z = torch.fft.ifftn(w, dim=(0,1,2)).real
        
        # Compute products in physical space
        uu_x = torch.fft.fftn(u_x * u_x, dim=(0,1,2)) * self.dealias
        uv_y = torch.fft.fftn(u_x * u_y, dim=(0,1,2)) * self.dealias
        uw_z = torch.fft.fftn(u_x * u_z, dim=(0,1,2)) * self.dealias
        
        vu_x = torch.fft.fftn(u_y * u_x, dim=(0,1,2)) * self.dealias
        vv_y = torch.fft.fftn(u_y * u_y, dim=(0,1,2)) * self.dealias
        vw_z = torch.fft.fftn(u_y * u_z, dim=(0,1,2)) * self.dealias
        
        wu_x = torch.fft.fftn(u_z * u_x, dim=(0,1,2)) * self.dealias
        wv_y = torch.fft.fftn(u_z * u_y, dim=(0,1,2)) * self.dealias
        ww_z = torch.fft.fftn(u_z * u_z, dim=(0,1,2)) * self.dealias
        
        # Divergence in spectral space
        Nx = 1j * (self.Kx * uu_x + self.Ky * uv_y + self.Kz * uw_z)
        Ny = 1j * (self.Kx * vu_x + self.Ky * vv_y + self.Kz * vw_z)
        Nz = 1j * (self.Kx * wu_x + self.Ky * wv_y + self.Kz * ww_z)
        
        return Nx, Ny, Nz
    
    def compute_diffusion(self, u_hat: torch.Tensor, v_hat: torch.Tensor, 
                         w_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diffusion term: ν∇²u"""
        laplacian = -self.k_sq
        Du = self.cfg.nu * laplacian * u_hat
        Dv = self.cfg.nu * laplacian * v_hat
        Dw = self.cfg.nu * laplacian * w_hat
        return Du, Dv, Dw
    
    def step(self, u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single RK3 step with strict projection"""
        cfg = self.cfg
        dt = cfg.dt
        nu = cfg.nu
        
        # RK3 coefficients
        a = [8/15, 5/12, 3/4]
        b = [0, -17/60, -5/12]
        g = [8/15, 2/15, 1/3]
        
        u_hat_k = u_hat.clone()
        v_hat_k = v_hat.clone()
        w_hat_k = w_hat.clone()
        
        for stage in range(3):
            # Nonlinear and diffusion
            Nx, Ny, Nz = self.compute_nonlinear(u_hat_k, v_hat_k, w_hat_k)
            Dx, Dy, Dz = self.compute_diffusion(u_hat_k, v_hat_k, w_hat_k)
            
            if stage == 0:
                u_hat_temp = u_hat + dt * (a[0]*Nx + g[0]*Dx)
                v_hat_temp = v_hat + dt * (a[0]*Ny + g[0]*Dy)
                w_hat_temp = w_hat + dt * (a[0]*Nz + g[0]*Dz)
            else:
                u_hat_temp = u_hat + dt * (a[stage]*Nx + b[stage]*Nx_prev + g[stage]*Dx)
                v_hat_temp = v_hat + dt * (a[stage]*Ny + b[stage]*Ny_prev + g[stage]*Dy)
                w_hat_temp = w_hat + dt * (a[stage]*Nz + b[stage]*Nz_prev + g[stage]*Dz)
            
            Nx_prev, Ny_prev, Nz_prev = Nx, Ny, Nz
            
            # Helmholtz projection
            u_hat_k, v_hat_k, w_hat_k = self.helmholtz_projection(
                u_hat_temp, v_hat_temp, w_hat_temp)
        
        # Transform to physical, apply BCs
        u = torch.fft.ifftn(u_hat_k, dim=(0,1,2)).real
        v = torch.fft.ifftn(v_hat_k, dim=(0,1,2)).real
        w = torch.fft.ifftn(w_hat_k, dim=(0,1,2)).real
        
        # Apply cylinder BC
        u, v, w = self.apply_cylinder_bc(u, v, w)
        
        # Apply inflow/outflow
        u, v, w = self.inflow_bc(u, v, w)
        u, v, w = self.outflow_bc(u, v, w)
        
        # Transform back
        u_hat_new = torch.fft.fftn(u, dim=(0,1,2))
        v_hat_new = torch.fft.fftn(v, dim=(0,1,2))
        w_hat_new = torch.fft.fftn(w, dim=(0,1,2))
        
        return u_hat_new, v_hat_new, w_hat_new
    
    def add_correlated_noise(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add spatially correlated noise (smooth, physical-like perturbations)"""
        cfg = self.cfg
        
        # Generate white noise
        noise_u = torch.randn_like(u)
        noise_v = torch.randn_like(v)
        noise_w = torch.randn_like(w)
        
        # Filter in Fourier space for correlation
        noise_u_hat = torch.fft.fftn(noise_u, dim=(0,1,2))
        noise_v_hat = torch.fft.fftn(noise_v, dim=(0,1,2))
        noise_w_hat = torch.fft.fftn(noise_w, dim=(0,1,2))
        
        # Gaussian filter: exp(-k²L²/2)
        k_filter = torch.exp(-self.k_sq * cfg.noise_correlation_length**2 / 2)
        
        noise_u_hat *= k_filter
        noise_v_hat *= k_filter
        noise_w_hat *= k_filter
        
        # Back to physical
        noise_u = torch.fft.ifftn(noise_u_hat, dim=(0,1,2)).real
        noise_v = torch.fft.ifftn(noise_v_hat, dim=(0,1,2)).real
        noise_w = torch.fft.ifftn(noise_w_hat, dim=(0,1,2)).real
        
        # Normalize and scale
        u_rms = torch.sqrt(torch.mean(u**2))
        scale = cfg.noise_level * u_rms
        
        noise_u = scale * noise_u / (torch.std(noise_u) + 1e-10)
        noise_v = scale * noise_v / (torch.std(noise_v) + 1e-10)
        noise_w = scale * noise_w / (torch.std(noise_w) + 1e-10)
        
        # Don't add noise inside cylinder
        mask = 1 - self.cylinder_mask
        
        return u + noise_u * mask, v + noise_v * mask, w + noise_w * mask
    
    def initialize_wake(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize with realistic wake profile + vortex shedding"""
        cfg = self.cfg
        
        # Base flow: uniform + wake deficit behind cylinder
        u = torch.ones_like(self.X)
        v = torch.zeros_like(self.X)
        w = torch.zeros_like(self.X)
        
        # Wake deficit: Gaussian profile downstream of cylinder
        x_rel = self.X - cfg.cylinder_x - cfg.cylinder_d
        y_rel = self.Y - cfg.cylinder_y
        
        # Wake spreads with sqrt(x)
        wake_width = cfg.cylinder_d * (1 + cfg.wake_spread_rate * torch.clamp(x_rel, 0, 10) / cfg.cylinder_d)
        
        # Deficit magnitude decays with x
        deficit = cfg.wake_velocity_deficit * torch.exp(-y_rel**2 / (2*wake_width**2))
        deficit = deficit * torch.sigmoid(x_rel * 2)  # Only downstream
        
        u = u - deficit
        
        # Add vortex shedding (Karman vortex street)
        theta = torch.atan2(y_rel, torch.clamp(x_rel, 0.1, 10))
        r = torch.sqrt(x_rel**2 + y_rel**2)
        
        # Oscillating perturbation
        omega = 2 * np.pi * cfg.vortex_shedding_freq
        phase = omega * 0  # Initial phase
        
        vortex_strength = 0.1 * torch.exp(-r / (2*cfg.cylinder_d))
        v += vortex_strength * torch.sin(2*theta + phase) * torch.exp(-x_rel**2/(2*cfg.cylinder_d**2))
        
        # 3D instability: streamwise vortices
        z_pert = 0.05 * torch.sin(2*np.pi*self.Z/cfg.lz) * torch.exp(-(x_rel**2 + y_rel**2)/(cfg.cylinder_d**2))
        w += z_pert
        
        # Apply cylinder BC
        u, v, w = self.apply_cylinder_bc(u, v, w)
        
        # Transform to spectral
        u_hat = torch.fft.fftn(u, dim=(0,1,2))
        v_hat = torch.fft.fftn(v, dim=(0,1,2))
        w_hat = torch.fft.fftn(w, dim=(0,1,2))
        
        # Project to divergence-free
        u_hat, v_hat, w_hat = self.helmholtz_projection(u_hat, v_hat, w_hat)
        
        return u_hat, v_hat, w_hat

# ==============================================================================
# Coherent Structure Detection
# ==============================================================================
class CoherentStructureAnalyzer:
    """Detect coherent structures using Q-criterion, Lambda2, and Delta-criterion"""
    
    @staticmethod
    def compute_velocity_gradient(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                   dx: float, dy: float, dz: float) -> Tuple[np.ndarray, ...]:
        """Compute velocity gradient tensor components"""
        # Central differences
        dudx = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2*dx)
        dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2*dy)
        dudz = (np.roll(u, -1, 2) - np.roll(u, 1, 2)) / (2*dz)
        
        dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2*dx)
        dvdy = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2*dy)
        dvdz = (np.roll(v, -1, 2) - np.roll(v, 1, 2)) / (2*dz)
        
        dwdx = (np.roll(w, -1, 0) - np.roll(w, 1, 0)) / (2*dx)
        dwdy = (np.roll(w, -1, 1) - np.roll(w, 1, 1)) / (2*dy)
        dwdz = (np.roll(w, -1, 2) - np.roll(w, 1, 2)) / (2*dz)
        
        return dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
    
    @staticmethod
    def q_criterion(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                    dx: float, dy: float, dz: float) -> np.ndarray:
        """
        Q-criterion: Q = 1/2(||Ω||² - ||S||²) > 0 indicates vortex
        where Ω is vorticity tensor, S is strain rate tensor
        """
        grad = CoherentStructureAnalyzer.compute_velocity_gradient(u, v, w, dx, dy, dz)
        dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz = grad
        
        # Strain rate S_ij = 1/2(∂u_i/∂x_j + ∂u_j/∂x_i)
        S11 = dudx
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S22 = dvdy
        S23 = 0.5 * (dvdz + dwdy)
        S33 = dwdz
        
        # Vorticity Ω_ij = 1/2(∂u_i/∂x_j - ∂u_j/∂x_i)
        O12 = 0.5 * (dudy - dvdx)
        O13 = 0.5 * (dudz - dwdx)
        O23 = 0.5 * (dvdz - dwdy)
        
        # Norms squared
        norm_S_sq = S11**2 + 2*S12**2 + 2*S13**2 + S22**2 + 2*S23**2 + S33**2
        norm_O_sq = 2*O12**2 + 2*O13**2 + 2*O23**2
        
        Q = 0.5 * (norm_O_sq - norm_S_sq)
        return Q
    
    @staticmethod
    def lambda2_criterion(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                          dx: float, dy: float, dz: float) -> np.ndarray:
        """
        Lambda2-criterion: λ2 < 0 indicates vortex
        λ2 is the middle eigenvalue of S² + Ω²
        """
        grad = CoherentStructureAnalyzer.compute_velocity_gradient(u, v, w, dx, dy, dz)
        dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz = grad
        
        # Construct S² + Ω² at each point
        nx, ny, nz = u.shape
        lambda2 = np.zeros_like(u)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    J = np.array([
                        [dudx[i,j,k], dudy[i,j,k], dudz[i,j,k]],
                        [dvdx[i,j,k], dvdy[i,j,k], dvdz[i,j,k]],
                        [dwdx[i,j,k], dwdy[i,j,k], dwdz[i,j,k]]
                    ])
                    
                    S = 0.5 * (J + J.T)
                    O = 0.5 * (J - J.T)
                    
                    A = S @ S + O @ O
                    eigenvalues = np.linalg.eigvals(A)
                    eigenvalues = np.sort(eigenvalues)
                    lambda2[i,j,k] = eigenvalues[1]  # Middle eigenvalue
        
        return lambda2
    
    @staticmethod
    def vorticity_magnitude(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            dx: float, dy: float, dz: float) -> np.ndarray:
        """Compute vorticity magnitude |ω|"""
        grad = CoherentStructureAnalyzer.compute_velocity_gradient(u, v, w, dx, dy, dz)
        dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz = grad
        
        wx = dwdy - dvdz
        wy = dudz - dwdx
        wz = dvdx - dudy
        
        return np.sqrt(wx**2 + wy**2 + wz**2)

# ==============================================================================
# Visualization
# ==============================================================================
class Visualizer:
    def __init__(self, cfg: Config, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.dx = cfg.lx / cfg.nx
        self.dy = cfg.ly / cfg.ny
        self.dz = cfg.lz / cfg.nz
        
    def save_contour_plots(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                           step: int, prefix: str = ""):
        """Save 2D contour plots at mid-planes"""
        cfg = self.cfg
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # XY plane (z = nz//2)
        z_idx = cfg.nz // 2
        u_xy = u[:, :, z_idx]
        v_xy = v[:, :, z_idx]
        w_xy = w[:, :, z_idx]
        
        # XZ plane (y = ny//2)
        y_idx = cfg.ny // 2
        u_xz = u[:, y_idx, :]
        w_xz = w[:, y_idx, :]
        
        # Custom colormap
        colors = ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        # XY plane plots
        im1 = axes[0,0].contourf(u_xy.T, levels=20, cmap=cmap, vmin=-0.5, vmax=1.2)
        axes[0,0].set_title(f'U-velocity (XY plane, z={z_idx})')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(v_xy.T, levels=20, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        axes[0,1].set_title(f'V-velocity (XY plane, z={z_idx})')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Vorticity in XY plane
        dvdx = (np.roll(v_xy, -1, 0) - np.roll(v_xy, 1, 0)) / (2*self.dx)
        dudy = (np.roll(u_xy, -1, 1) - np.roll(u_xy, 1, 1)) / (2*self.dy)
        vort_z = dvdx - dudy
        
        im3 = axes[0,2].contourf(vort_z.T, levels=20, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[0,2].set_title(f'Vorticity Z (XY plane)')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[0,2])
        
        # XZ plane plots
        im4 = axes[1,0].contourf(u_xz.T, levels=20, cmap=cmap, vmin=-0.5, vmax=1.2)
        axes[1,0].set_title(f'U-velocity (XZ plane, y={y_idx})')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('z')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].contourf(w_xz.T, levels=20, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        axes[1,1].set_title(f'W-velocity (XZ plane, y={y_idx})')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('z')
        plt.colorbar(im5, ax=axes[1,1])
        
        # Velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        im6 = axes[1,2].contourf(vel_mag[:, :, z_idx].T, levels=20, cmap='viridis')
        axes[1,2].set_title(f'|U| magnitude (XY plane)')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('y')
        plt.colorbar(im6, ax=axes[1,2])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{prefix}contours_step{step:04d}.png", dpi=150)
        plt.close()
        
    def save_coherent_structures(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                  step: int, prefix: str = ""):
        """Save coherent structure visualizations"""
        cfg = self.cfg
        
        analyzer = CoherentStructureAnalyzer()
        
        # Compute criteria
        Q = analyzer.q_criterion(u, v, w, self.dx, self.dy, self.dz)
        vort_mag = analyzer.vorticity_magnitude(u, v, w, self.dx, self.dy, self.dz)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Q-criterion
        z_idx = cfg.nz // 2
        Q_thresh = 0.1 * np.max(Q)
        Q_binary = (Q > Q_thresh).astype(float)
        
        im1 = axes[0,0].contourf(Q[:, :, z_idx].T, levels=20, cmap='hot')
        axes[0,0].set_title(f'Q-criterion (z={z_idx})')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(Q_binary[:, :, z_idx].T, levels=[0, 0.5, 1], cmap='Reds')
        axes[0,1].set_title(f'Q > {Q_thresh:.3f} (Vortex regions)')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        
        # Vorticity magnitude
        im3 = axes[0,2].contourf(vort_mag[:, :, z_idx].T, levels=20, cmap='magma')
        axes[0,2].set_title(f'|ω| magnitude (z={z_idx})')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[0,2])
        
        # 3D isosurface projection (showing Q>threshold locations)
        Q_mask = Q > Q_thresh
        x_proj = np.sum(Q_mask, axis=1)  # Project onto YZ plane
        y_proj = np.sum(Q_mask, axis=0)  # Project onto XZ plane
        z_proj = np.sum(Q_mask, axis=2)  # Project onto XY plane
        
        im4 = axes[1,0].contourf(x_proj.T, levels=10, cmap='hot')
        axes[1,0].set_title('Vortex projection (YZ)')
        axes[1,0].set_xlabel('y')
        axes[1,0].set_ylabel('z')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].contourf(y_proj.T, levels=10, cmap='hot')
        axes[1,1].set_title('Vortex projection (XZ)')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('z')
        plt.colorbar(im5, ax=axes[1,1])
        
        im6 = axes[1,2].contourf(z_proj.T, levels=10, cmap='hot')
        axes[1,2].set_title('Vortex projection (XY)')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('y')
        plt.colorbar(im6, ax=axes[1,2])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{prefix}coherent_step{step:04d}.png", dpi=150)
        plt.close()
        
        return Q, vort_mag
    
    def save_isosurface_data(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                             Q: np.ndarray, step: int, prefix: str = ""):
        """Save data for 3D isosurface plotting (external)"""
        cfg = self.cfg
        
        # Save numpy arrays for external visualization
        np.savez_compressed(
            f"{self.output_dir}/{prefix}data_step{step:04d}.npz",
            u=u, v=v, w=w,
            Q=Q,
            x=np.linspace(0, cfg.lx, cfg.nx),
            y=np.linspace(0, cfg.ly, cfg.ny),
            z=np.linspace(0, cfg.lz, cfg.nz),
            cylinder_x=cfg.cylinder_x,
            cylinder_y=cfg.cylinder_y,
            cylinder_d=cfg.cylinder_d
        )

# ==============================================================================
# Main Simulation
# ==============================================================================
def run_synthetic_3d_cylinder_simulation():
    """Run complete 500-step synthetic 3D cylinder wake simulation"""
    
    print("=" * 70)
    print("SYNTHETIC 3D CYLINDER WAKE - 500 STEP SIMULATION")
    print("Strict Physics: Spectral NS + Helmholtz Projection")
    print(f"Re = {config.reynolds}, Noise Level = {config.noise_level*100}%")
    print("=" * 70)
    
    # Initialize
    solver = CylinderWake3DSolver(config)
    visualizer = Visualizer(config, "paper/synthetic_3d_cylinder")
    
    # Initial condition
    print("\n[INIT] Creating initial wake profile...")
    u_hat, v_hat, w_hat = solver.initialize_wake()
    
    # Storage for analysis
    history = {
        'time': [],
        'energy': [],
        'divergence': [],
        'enstrophy': [],
        'u_rms': [],
        'v_rms': [],
        'w_rms': []
    }
    
    # Run simulation
    print(f"\n[SIMULATION] Running {config.n_steps} steps...")
    print(f"{'Step':<10}{'Time':<12}{'Energy':<15}{'Divergence':<15}{'Enstrophy':<15}")
    print("-" * 70)
    
    for step in range(config.n_steps + 1):
        # Transform to physical
        u = torch.fft.ifftn(u_hat, dim=(0,1,2)).real
        v = torch.fft.ifftn(v_hat, dim=(0,1,2)).real
        w = torch.fft.ifftn(w_hat, dim=(0,1,2)).real
        
        # Add strong noise every 10 steps
        if step > 0 and step % 10 == 0:
            u, v, w = solver.add_correlated_noise(u, v, w)
            # Re-project after noise
            u_hat = torch.fft.fftn(u, dim=(0,1,2))
            v_hat = torch.fft.fftn(v, dim=(0,1,2))
            w_hat = torch.fft.fftn(w, dim=(0,1,2))
            u_hat, v_hat, w_hat = solver.helmholtz_projection(u_hat, v_hat, w_hat)
            u = torch.fft.ifftn(u_hat, dim=(0,1,2)).real
            v = torch.fft.ifftn(v_hat, dim=(0,1,2)).real
            w = torch.fft.ifftn(w_hat, dim=(0,1,2)).real
        
        # Compute diagnostics
        with torch.no_grad():
            energy = 0.5 * torch.mean(u**2 + v**2 + w**2).item()
            
            # Divergence
            dudx = (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2*visualizer.dx)
            dvdy = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2*visualizer.dy)
            dwdz = (torch.roll(w, -1, 2) - torch.roll(w, 1, 2)) / (2*visualizer.dz)
            div = torch.mean((dudx + dvdy + dwdz)**2).sqrt().item()
            
            # Enstrophy (vorticity squared)
            dvdx = (torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2*visualizer.dx)
            dudy = (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2*visualizer.dy)
            dwdx = (torch.roll(w, -1, 0) - torch.roll(w, 1, 0)) / (2*visualizer.dx)
            dudz = (torch.roll(u, -1, 2) - torch.roll(u, 1, 2)) / (2*visualizer.dz)
            dvdz = (torch.roll(v, -1, 2) - torch.roll(v, 1, 2)) / (2*visualizer.dz)
            dwdy = (torch.roll(w, -1, 1) - torch.roll(w, 1, 1)) / (2*visualizer.dy)
            
            wx = dwdy - dvdz
            wy = dudz - dwdx
            wz = dvdx - dudy
            enstrophy = torch.mean(wx**2 + wy**2 + wz**2).item()
            
            u_rms = torch.sqrt(torch.mean(u**2)).item()
            v_rms = torch.sqrt(torch.mean(v**2)).item()
            w_rms = torch.sqrt(torch.mean(w**2)).item()
        
        # Store history
        t = step * config.dt
        history['time'].append(t)
        history['energy'].append(energy)
        history['divergence'].append(div)
        history['enstrophy'].append(enstrophy)
        history['u_rms'].append(u_rms)
        history['v_rms'].append(v_rms)
        history['w_rms'].append(w_rms)
        
        # Print progress
        if step % 50 == 0:
            print(f"{step:<10}{t:<12.3f}{energy:<15.6f}{div:<15.2e}{enstrophy:<15.4f}")
            
            # Save visualizations
            u_np = u.cpu().numpy()
            v_np = v.cpu().numpy()
            w_np = w.cpu().numpy()
            
            visualizer.save_contour_plots(u_np, v_np, w_np, step, "gt_")
            Q, vort_mag = visualizer.save_coherent_structures(u_np, v_np, w_np, step, "gt_")
            visualizer.save_isosurface_data(u_np, v_np, w_np, Q, step, "gt_")
            
            # Save checkpoint
            np.savez_compressed(
                f"paper/synthetic_3d_cylinder/checkpoint_step{step:04d}.npz",
                u=u_np, v=v_np, w=w_np,
                Q=Q, vort_mag=vort_mag,
                step=step, time=t
            )
        
        # Time step
        if step < config.n_steps:
            u_hat, v_hat, w_hat = solver.step(u_hat, v_hat, w_hat)
    
    # Save final history
    np.savez("paper/synthetic_3d_cylinder/simulation_history.npz", **history)
    
    # Plot history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(history['time'], history['energy'])
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Kinetic Energy')
    axes[0,0].set_title('Energy Evolution')
    axes[0,0].grid(True)
    
    axes[0,1].semilogy(history['time'], history['divergence'])
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('RMS Divergence')
    axes[0,1].set_title('Divergence (should be ~machine precision)')
    axes[0,1].grid(True)
    
    axes[1,0].plot(history['time'], history['enstrophy'])
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Enstrophy')
    axes[1,0].set_title('Enstrophy Evolution')
    axes[1,0].grid(True)
    
    axes[1,1].plot(history['time'], history['u_rms'], label='u_rms')
    axes[1,1].plot(history['time'], history['v_rms'], label='v_rms')
    axes[1,1].plot(history['time'], history['w_rms'], label='w_rms')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('RMS Velocity')
    axes[1,1].set_title('Velocity Components')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig("paper/synthetic_3d_cylinder/simulation_history.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: paper/synthetic_3d_cylinder/")
    print(f"Files generated:")
    print(f"  - gt_contours_step*.png: Velocity contour plots")
    print(f"  - gt_coherent_step*.png: Coherent structure detection")
    print(f"  - gt_data_step*.npz: Raw data for 3D visualization")
    print(f"  - checkpoint_step*.npz: Full state checkpoints")
    print(f"  - simulation_history.npz: Time series data")
    print(f"  - simulation_history.png: Evolution plots")
    
    return history

# ==============================================================================
# Neural Network Prediction on Synthetic Data
# ==============================================================================
class SimplePredictor(nn.Module):
    """Simple ConvLSTM-based predictor for testing"""
    def __init__(self, channels=3, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden, channels, 3, padding=1),
        )
        
    def forward(self, x):
        return x + self.decoder(self.encoder(x))

def train_and_predict_500_steps():
    """Train predictor on synthetic data and run 500-step prediction"""
    
    print("\n" + "=" * 70)
    print("TRAINING PREDICTOR ON SYNTHETIC 3D CYLINDER DATA")
    print("=" * 70)
    
    # Create visualizer
    visualizer = Visualizer(config, "paper/synthetic_3d_cylinder")
    
    # Load ground truth data
    checkpoints = []
    for step in range(0, config.n_steps + 1, config.save_interval):
        data = np.load(f"paper/synthetic_3d_cylinder/checkpoint_step{step:04d}.npz")
        u = torch.tensor(data['u'], dtype=torch.float32, device=device)
        v = torch.tensor(data['v'], dtype=torch.float32, device=device)
        w = torch.tensor(data['w'], dtype=torch.float32, device=device)
        checkpoints.append(torch.stack([u, v, w], dim=0))  # [3, nx, ny, nz]
    
    checkpoints = torch.stack(checkpoints)  # [n_checkpoints, 3, nx, ny, nz]
    print(f"Loaded {len(checkpoints)} checkpoints")
    
    # Create training pairs (t -> t+1)
    train_data = []
    for i in range(len(checkpoints) - 1):
        train_data.append((checkpoints[i], checkpoints[i+1]))
    
    # Simple model
    model = SimplePredictor(channels=3, hidden=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    print("\n[TRAINING]...")
    model.train()
    for epoch in range(100):
        total_loss = 0
        for x, y in train_data:
            x_batch = x.unsqueeze(0)  # [1, 3, nx, ny, nz]
            y_batch = y.unsqueeze(0)
            
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_data):.6f}")
    
    # 500-step prediction
    print("\n[500-STEP PREDICTION]...")
    model.eval()
    
    # Start from initial condition
    u_current = checkpoints[0].clone()
    
    pred_history = {
        'mse': [],
        'energy_error': [],
        'divergence': []
    }
    
    with torch.no_grad():
        for step in range(config.n_steps + 1):
            # Predict
            u_pred = model(u_current.unsqueeze(0))[0]
            u_current = u_pred
            
            # Compare with ground truth at checkpoints
            if step % config.save_interval == 0:
                gt_idx = step // config.save_interval
                if gt_idx < len(checkpoints):
                    gt = checkpoints[gt_idx]
                    
                    mse = F.mse_loss(u_current, gt).item()
                    
                    # Energy
                    energy_pred = 0.5 * torch.mean(u_current[0]**2 + u_current[1]**2 + u_current[2]**2).item()
                    energy_gt = 0.5 * torch.mean(gt[0]**2 + gt[1]**2 + gt[2]**2).item()
                    energy_err = abs(energy_pred - energy_gt) / (energy_gt + 1e-10)
                    
                    # Divergence
                    u, v, w = u_current[0], u_current[1], u_current[2]
                    dudx = (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2*visualizer.dx)
                    dvdy = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2*visualizer.dy)
                    dwdz = (torch.roll(w, -1, 2) - torch.roll(w, 1, 2)) / (2*visualizer.dz)
                    div = torch.mean((dudx + dvdy + dwdz)**2).sqrt().item()
                    
                    pred_history['mse'].append(mse)
                    pred_history['energy_error'].append(energy_err)
                    pred_history['divergence'].append(div)
                    
                    print(f"Step {step:4d}: MSE={mse:.6e}, Energy_err={energy_err:.4f}, Div={div:.2e}")
                    
                    # Save visualization
                    if step % (config.save_interval * 2) == 0:
                        u_np = u_current[0].cpu().numpy()
                        v_np = u_current[1].cpu().numpy()
                        w_np = u_current[2].cpu().numpy()
                        visualizer.save_contour_plots(u_np, v_np, w_np, step, "pred_")
    
    # Save prediction results
    np.savez("paper/synthetic_3d_cylinder/prediction_results.npz", **pred_history)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = np.arange(0, config.n_steps + 1, config.save_interval)
    
    axes[0].semilogy(steps, pred_history['mse'])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Prediction MSE')
    axes[0].grid(True)
    
    axes[1].plot(steps, np.array(pred_history['energy_error']) * 100)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Energy Error (%)')
    axes[1].set_title('Energy Drift')
    axes[1].grid(True)
    
    axes[2].semilogy(steps, pred_history['divergence'])
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Divergence')
    axes[2].set_title('Divergence Evolution')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("paper/synthetic_3d_cylinder/prediction_metrics.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    # Create output directory
    os.makedirs("paper/synthetic_3d_cylinder", exist_ok=True)
    
    # Run simulation
    history = run_synthetic_3d_cylinder_simulation()
    
    # Train and predict
    train_and_predict_500_steps()

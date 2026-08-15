"""
Taylor-Green Vortex Validation v2 - Full NS Evolution
Proper nonlinear evolution with convection term
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 10

class TaylorGreenSolver:
    """Full Taylor-Green vortex solver with nonlinear NS evolution"""
    
    def __init__(self, N=64, Re=1600, L=2*np.pi):
        self.N = N
        self.Re = Re
        self.L = L
        self.nu = 1.0 / Re
        self.dx = L / N
        
        # Physical grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        
        # Wavenumbers for spectral derivatives
        k = np.fft.fftfreq(N, L/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0  # Avoid division by zero
        
    def initial_condition(self):
        """Taylor-Green initial condition"""
        u = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        v = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        w = np.zeros_like(u)
        return u, v, w
    
    def fft(self, u):
        return np.fft.fftn(u)
    
    def ifft(self, u_hat):
        return np.fft.ifftn(u_hat).real
    
    def spectral_derivative(self, u_hat, direction):
        """Compute spectral derivative"""
        if direction == 'x':
            return self.ifft(1j * self.kx * u_hat)
        elif direction == 'y':
            return self.ifft(1j * self.ky * u_hat)
        elif direction == 'z':
            return self.ifft(1j * self.kz * u_hat)
    
    def compute_rhs(self, u, v, w):
        """
        Compute RHS of NS equations: du/dt = -u·grad(u) - grad(p) + nu*laplacian(u)
        Using pressure projection for incompressibility
        """
        # Forward FFT
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)
        
        # Physical space derivatives for nonlinear term
        ux = self.spectral_derivative(u_hat, 'x')
        uy = self.spectral_derivative(u_hat, 'y')
        uz = self.spectral_derivative(u_hat, 'z')
        vx = self.spectral_derivative(v_hat, 'x')
        vy = self.spectral_derivative(v_hat, 'y')
        vz = self.spectral_derivative(v_hat, 'z')
        wx = self.spectral_derivative(w_hat, 'x')
        wy = self.spectral_derivative(w_hat, 'y')
        wz = self.spectral_derivative(w_hat, 'z')
        
        # Nonlinear convection in physical space
        conv_u = u * ux + v * uy + w * uz
        conv_v = u * vx + v * vy + w * vz
        conv_w = u * wx + v * wy + w * wz
        
        # FFT of nonlinear terms
        conv_u_hat = self.fft(conv_u)
        conv_v_hat = self.fft(conv_v)
        conv_w_hat = self.fft(conv_w)
        
        # Viscous term in spectral space
        visc_u_hat = -self.nu * self.k2 * u_hat
        visc_v_hat = -self.nu * self.k2 * v_hat
        visc_w_hat = -self.nu * self.k2 * w_hat
        
        # RHS without pressure
        rhs_u_hat = -conv_u_hat + visc_u_hat
        rhs_v_hat = -conv_v_hat + visc_v_hat
        rhs_w_hat = -conv_w_hat + visc_w_hat
        
        # Pressure projection (incompressibility constraint)
        div_rhs = self.kx * rhs_u_hat + self.ky * rhs_v_hat + self.kz * rhs_w_hat
        p_hat = div_rhs / self.k2
        p_hat[0,0,0] = 0  # Zero mean pressure
        
        # Subtract pressure gradient
        rhs_u_hat = rhs_u_hat - self.kx * p_hat * 1j
        rhs_v_hat = rhs_v_hat - self.ky * p_hat * 1j
        rhs_w_hat = rhs_w_hat - self.kz * p_hat * 1j
        
        return rhs_u_hat, rhs_v_hat, rhs_w_hat
    
    def step_rk4(self, u, v, w, dt):
        """RK4 time integration"""
        # k1
        rhs_u1, rhs_v1, rhs_w1 = self.compute_rhs(u, v, w)
        
        # k2
        u2 = u + 0.5 * dt * self.ifft(rhs_u1)
        v2 = v + 0.5 * dt * self.ifft(rhs_v1)
        w2 = w + 0.5 * dt * self.ifft(rhs_w1)
        rhs_u2, rhs_v2, rhs_w2 = self.compute_rhs(u2, v2, w2)
        
        # k3
        u3 = u + 0.5 * dt * self.ifft(rhs_u2)
        v3 = v + 0.5 * dt * self.ifft(rhs_v2)
        w3 = w + 0.5 * dt * self.ifft(rhs_w2)
        rhs_u3, rhs_v3, rhs_w3 = self.compute_rhs(u3, v3, w3)
        
        # k4
        u4 = u + dt * self.ifft(rhs_u3)
        v4 = v + dt * self.ifft(rhs_v3)
        w4 = w + dt * self.ifft(rhs_w3)
        rhs_u4, rhs_v4, rhs_w4 = self.compute_rhs(u4, v4, w4)
        
        # Combine
        u_new = u + dt/6 * (self.ifft(rhs_u1) + 2*self.ifft(rhs_u2) + 
                            2*self.ifft(rhs_u3) + self.ifft(rhs_u4))
        v_new = v + dt/6 * (self.ifft(rhs_v1) + 2*self.ifft(rhs_v2) + 
                            2*self.ifft(rhs_v3) + self.ifft(rhs_v4))
        w_new = w + dt/6 * (self.ifft(rhs_w1) + 2*self.ifft(rhs_w2) + 
                            2*self.ifft(rhs_w3) + self.ifft(rhs_w4))
        
        return u_new, v_new, w_new
    
    def compute_enstrophy(self, u, v, w):
        """Compute mean enstrophy"""
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)
        
        # Vorticity in spectral space
        wx_hat = 1j * (self.ky * w_hat - self.kz * v_hat)
        wy_hat = 1j * (self.kz * u_hat - self.kx * w_hat)
        wz_hat = 1j * (self.kx * v_hat - self.ky * u_hat)
        
        # Enstrophy = 0.5 * <omega^2>
        enstrophy = 0.5 * np.mean(np.abs(wx_hat)**2 + np.abs(wy_hat)**2 + np.abs(wz_hat)**2)
        
        return enstrophy
    
    def compute_energy(self, u, v, w):
        """Compute kinetic energy"""
        return 0.5 * np.mean(u**2 + v**2 + w**2)
    
    def laplacian_rms(self, u, v, w):
        """Compute RMS of Laplacian (sigma2_HC)"""
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)
        
        lap_u = self.ifft(-self.k2 * u_hat)
        lap_v = self.ifft(-self.k2 * v_hat)
        lap_w = self.ifft(-self.k2 * w_hat)
        
        return np.sqrt(np.mean(lap_u**2 + lap_v**2 + lap_w**2))
    
    def run_simulation(self, t_end=20, dt=0.01):
        """Run full simulation"""
        print(f"Running Taylor-Green vortex simulation:")
        print(f"  Resolution: {self.N}^3")
        print(f"  Re = {self.Re}")
        print(f"  dt = {dt}, t_end = {t_end}")
        print(f"  Total steps: {int(t_end/dt)}")
        print()
        
        u, v, w = self.initial_condition()
        
        # Storage
        times = []
        energies = []
        enstrophies = []
        sigma2_hc_values = []
        snapshots = {}
        
        n_steps = int(t_end / dt)
        save_times = [0, 3, 6, 9, 12, 15, 18]
        
        t = 0
        for step in range(n_steps + 1):
            # Record data
            if step % int(1.0/dt) == 0:  # Every 1 time unit
                E = self.compute_energy(u, v, w)
                ens = self.compute_enstrophy(u, v, w)
                sigma2 = self.laplacian_rms(u, v, w)
                
                times.append(t)
                energies.append(E)
                enstrophies.append(ens)
                sigma2_hc_values.append(sigma2)
                
                marker = ""
                if abs(t - 9) < 0.5:
                    marker = " <-- expected peak"
                
                print(f"t = {t:5.1f}: E = {E:.4f}, Enstrophy = {ens:.4f}, sigma2_HC = {sigma2:.4f}{marker}")
            
            # Save snapshots
            if round(t) in save_times and round(t) not in [round(k) for k in snapshots.keys()]:
                snapshots[round(t)] = (u.copy(), v.copy(), w.copy())
            
            # Time step
            if step < n_steps:
                u, v, w = self.step_rk4(u, v, w, dt)
                t += dt
        
        return (np.array(times), np.array(energies), np.array(enstrophies), 
                np.array(sigma2_hc_values), snapshots)


def visualize_results(times, energies, enstrophies, sigma2_hc, snapshots, solver):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Row 1: Time evolution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, sigma2_hc, 'o-', linewidth=2.5, markersize=10, 
             color='darkred', label=r'$\sigma^2_{HC}$ (Laplacian RMS)')
    ax1.axvline(9, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Expected peak (t=9)')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel(r'$\sigma^2_{HC}$', fontsize=12)
    ax1.set_title(r'$\sigma^2_{HC}$ Time Evolution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Enstrophy evolution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.semilogy(times, enstrophies, 's-', linewidth=2.5, markersize=8,
                 color='darkblue', label='Enstrophy')
    ax2.axvline(9, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Enstrophy (log scale)', fontsize=12)
    ax2.set_title('Enstrophy Evolution\n(Should peak near t=9)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    # Energy decay
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(times, energies, '^-', linewidth=2.5, markersize=8,
             color='darkgreen', label='Kinetic Energy')
    ax3.axvline(9, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Energy', fontsize=12)
    ax3.set_title('Energy Decay', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    
    # Row 2 & 3: Snapshots at t=0, 9, 18
    key_times = [0, 9, 18]
    for idx, t in enumerate(key_times):
        if t not in snapshots:
            continue
            
        u, v, w = snapshots[t]
        
        # Velocity magnitude (mid-plane slice)
        mid = solver.N // 2
        vel_mag = np.sqrt(u[:, :, mid]**2 + v[:, :, mid]**2 + w[:, :, mid]**2)
        
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(vel_mag.T, origin='lower', cmap='hot', extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_title(f't={t}: |u| (z=π slice)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Vorticity magnitude
        u_hat = solver.fft(u)
        v_hat = solver.fft(v)
        w_hat = solver.fft(w)
        
        wx = solver.ifft(1j * (solver.ky * w_hat - solver.kz * v_hat))
        wy = solver.ifft(1j * (solver.kz * u_hat - solver.kx * w_hat))
        wz = solver.ifft(1j * (solver.kx * v_hat - solver.ky * u_hat))
        
        vort_mag = np.sqrt(wx[:, :, mid]**2 + wy[:, :, mid]**2 + wz[:, :, mid]**2)
        
        ax = fig.add_subplot(gs[2, idx])
        im = ax.imshow(vort_mag.T, origin='lower', cmap='Blues', extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_title(f't={t}: |ω| (vorticity)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Validation summary
    peak_idx = np.argmax(sigma2_hc)
    peak_time = times[peak_idx]
    
    ax_text = fig.add_axes([0.65, 0.72, 0.32, 0.15])
    ax_text.axis('off')
    
    ens_peak_idx = np.argmax(enstrophies)
    ens_peak_time = times[ens_peak_idx]
    
    summary_text = (
        f"VALIDATION SUMMARY\n"
        f"{'='*35}\n\n"
        f"sigma2_HC peak: t = {peak_time:.1f}\n"
        f"Enstrophy peak: t = {ens_peak_time:.1f}\n"
        f"Expected peak:  t ≈ 9\n\n"
    )
    
    if abs(peak_time - 9) <= 2 and abs(ens_peak_time - 9) <= 2:
        summary_text += "[OK] Temporal correspondence\n     validated!"
        color = 'lightgreen'
    else:
        summary_text += f"[CHECK] Peak offset detected"
        color = 'lightyellow'
    
    ax_text.text(0.05, 0.95, summary_text,
                fontsize=11, verticalalignment='top', family='monospace',
                transform=ax_text.transAxes,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))
    
    plt.suptitle('Taylor-Green Vortex Validation: Full NS Evolution (Re=1600)', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('taylor_green_validation_v2.png', dpi=200, bbox_inches='tight')
    print("\nSaved: taylor_green_validation_v2.png")
    plt.close()


def main():
    print("="*70)
    print("Taylor-Green Vortex Validation v2 - Full NS Evolution")
    print("="*70)
    print("\nThis version includes:")
    print("  - Full nonlinear convection term")
    print("  - Spectral methods for spatial derivatives")
    print("  - RK4 time integration")
    print("  - Pressure projection for incompressibility")
    print("="*70)
    
    # Run simulation
    solver = TaylorGreenSolver(N=64, Re=1600)
    results = solver.run_simulation(t_end=20, dt=0.005)
    times, energies, enstrophies, sigma2_hc, snapshots = results
    
    # Visualize
    visualize_results(times, energies, enstrophies, sigma2_hc, snapshots, solver)
    
    # Final validation
    peak_idx = np.argmax(sigma2_hc)
    peak_time = times[peak_idx]
    ens_peak_idx = np.argmax(enstrophies)
    ens_peak_time = times[ens_peak_idx]
    
    print("\n" + "="*70)
    print("FINAL VALIDATION")
    print("="*70)
    print(f"\nsigma2_HC peak time: {peak_time:.1f}")
    print(f"Enstrophy peak time: {ens_peak_time:.1f}")
    print(f"Expected peak:       ~9")
    
    if abs(peak_time - 9) <= 2:
        print(f"\n[OK] VALIDATED: sigma2_HC correlates with small-scale generation")
        print(f"     Temporal correspondence established")
    else:
        print(f"\n[WARNING] Peak at t={peak_time}, expected ~9")
    
    if abs(ens_peak_time - 9) <= 2:
        print(f"[OK] Enstrophy correctly peaks at t≈9 (nonlinear cascade)")
    else:
        print(f"[WARNING] Enstrophy peak at t={ens_peak_time}")
    
    print("="*70)


if __name__ == "__main__":
    main()

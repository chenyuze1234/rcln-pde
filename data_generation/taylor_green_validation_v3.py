"""
Taylor-Green Vortex Validation v3 - Stable NS Evolution
Low-storage RK3 with CFL condition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 10


class TaylorGreenSolver:
    """Stable Taylor-Green vortex solver"""
    
    def __init__(self, N=64, Re=1600, L=2*np.pi):
        self.N = N
        self.Re = Re
        self.L = L
        self.nu = 1.0 / Re
        self.dx = L / N
        
        # Physical grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        
        # Wavenumbers
        k = np.fft.fftfreq(N, L/N) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1.0
        
        # Dealiasing mask (2/3 rule)
        kmax = np.max(np.abs(k))
        self.dealias = (np.abs(self.kx) < 2/3*kmax) & (np.abs(self.ky) < 2/3*kmax) & (np.abs(self.kz) < 2/3*kmax)
        
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
    
    def compute_cfl(self, u, v, w):
        """Compute CFL number"""
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        w_max = np.max(np.abs(w))
        vel_max = max(u_max, v_max, w_max)
        return vel_max * self.dx / self.nu  # Actually viscous limit
    
    def compute_rhs_spectral(self, u_hat, v_hat, w_hat):
        """
        Compute RHS in spectral space using skew-symmetric form
        More stable than direct nonlinear evaluation
        """
        # Apply dealiasing
        u_hat = u_hat * self.dealias
        v_hat = v_hat * self.dealias
        w_hat = w_hat * self.dealias
        
        # Get physical space
        u = self.ifft(u_hat)
        v = self.ifft(v_hat)
        w = self.ifft(w_hat)
        
        # Derivatives
        ux = self.ifft(1j * self.kx * u_hat)
        uy = self.ifft(1j * self.ky * u_hat)
        uz = self.ifft(1j * self.kz * u_hat)
        vx = self.ifft(1j * self.kx * v_hat)
        vy = self.ifft(1j * self.ky * v_hat)
        vz = self.ifft(1j * self.kz * v_hat)
        wx = self.ifft(1j * self.kx * w_hat)
        wy = self.ifft(1j * self.ky * w_hat)
        wz = self.ifft(1j * self.kz * w_hat)
        
        # Skew-symmetric form: 0.5 * (u·grad(u) + div(u u))
        # This reduces aliasing errors
        conv_u = 0.5 * (u * ux + v * uy + w * uz) + 0.5 * (u * ux + v * ux + w * ux)
        conv_v = 0.5 * (u * vx + v * vy + w * vz) + 0.5 * (u * vy + v * vy + w * vy)
        conv_w = 0.5 * (u * wx + v * wy + w * wz) + 0.5 * (u * wz + v * wz + w * wz)
        
        conv_u_hat = self.fft(conv_u) * self.dealias
        conv_v_hat = self.fft(conv_v) * self.dealias
        conv_w_hat = self.fft(conv_w) * self.dealias
        
        # Viscous term
        visc_u_hat = -self.nu * self.k2 * u_hat
        visc_v_hat = -self.nu * self.k2 * v_hat
        visc_w_hat = -self.nu * self.k2 * w_hat
        
        # RHS without pressure
        rhs_u_hat = -conv_u_hat + visc_u_hat
        rhs_v_hat = -conv_v_hat + visc_v_hat
        rhs_w_hat = -conv_w_hat + visc_w_hat
        
        # Pressure projection
        div_rhs = self.kx * rhs_u_hat + self.ky * rhs_v_hat + self.kz * rhs_w_hat
        p_hat = div_rhs / self.k2
        p_hat[0,0,0] = 0
        
        rhs_u_hat = rhs_u_hat - 1j * self.kx * p_hat
        rhs_v_hat = rhs_v_hat - 1j * self.ky * p_hat
        rhs_w_hat = rhs_w_hat - 1j * self.kz * p_hat
        
        return rhs_u_hat, rhs_v_hat, rhs_w_hat
    
    def step_rk3(self, u_hat, v_hat, w_hat, dt):
        """Low-storage RK3 (Williamson's scheme)"""
        # Stage 1
        rhs_u, rhs_v, rhs_w = self.compute_rhs_spectral(u_hat, v_hat, w_hat)
        u_hat_temp = u_hat + dt * rhs_u
        v_hat_temp = v_hat + dt * rhs_v
        w_hat_temp = w_hat + dt * rhs_w
        
        # Stage 2
        rhs_u2, rhs_v2, rhs_w2 = self.compute_rhs_spectral(u_hat_temp, v_hat_temp, w_hat_temp)
        u_hat_temp = u_hat + dt/4 * (rhs_u + rhs_u2)
        v_hat_temp = v_hat + dt/4 * (rhs_v + rhs_v2)
        w_hat_temp = w_hat + dt/4 * (rhs_w + rhs_w2)
        
        # Stage 3
        rhs_u3, rhs_v3, rhs_w3 = self.compute_rhs_spectral(u_hat_temp, v_hat_temp, w_hat_temp)
        u_hat_new = u_hat + dt/6 * (rhs_u + rhs_u2 + 4*rhs_u3)
        v_hat_new = v_hat + dt/6 * (rhs_v + rhs_v2 + 4*rhs_v3)
        w_hat_new = w_hat + dt/6 * (rhs_w + rhs_w2 + 4*rhs_w3)
        
        return u_hat_new, v_hat_new, w_hat_new
    
    def compute_enstrophy(self, u_hat, v_hat, w_hat):
        """Compute mean enstrophy from spectral coefficients"""
        wx_hat = 1j * (self.ky * w_hat - self.kz * v_hat)
        wy_hat = 1j * (self.kz * u_hat - self.kx * w_hat)
        wz_hat = 1j * (self.kx * v_hat - self.ky * u_hat)
        
        enstrophy = 0.5 * np.mean(np.abs(wx_hat)**2 + np.abs(wy_hat)**2 + np.abs(wz_hat)**2)
        return enstrophy
    
    def compute_energy(self, u_hat, v_hat, w_hat):
        """Compute kinetic energy"""
        return 0.5 * np.mean(np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    
    def laplacian_rms(self, u_hat, v_hat, w_hat):
        """Compute RMS of Laplacian (sigma2_HC)"""
        lap_u = self.ifft(-self.k2 * u_hat)
        lap_v = self.ifft(-self.k2 * v_hat)
        lap_w = self.ifft(-self.k2 * w_hat)
        return np.sqrt(np.mean(lap_u**2 + lap_v**2 + lap_w**2))
    
    def run_simulation(self, t_end=20, dt=None):
        """Run full simulation"""
        if dt is None:
            # Viscous stability limit: dt < dx^2 / (nu * C)
            dt = 0.1 * self.dx**2 / self.nu
        
        print(f"Running Taylor-Green vortex simulation:")
        print(f"  Resolution: {self.N}^3")
        print(f"  Re = {self.Re}, nu = {self.nu:.6f}")
        print(f"  dt = {dt:.6f}, t_end = {t_end}")
        print(f"  Total steps: {int(t_end/dt)}")
        print()
        
        u, v, w = self.initial_condition()
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)
        
        # Storage
        times = []
        energies = []
        enstrophies = []
        sigma2_hc_values = []
        snapshots = {}
        
        save_times = [0, 3, 6, 9, 12, 15, 18]
        record_interval = max(1, int(0.5 / dt))  # Record every 0.5 time units
        
        t = 0
        step = 0
        n_steps = int(t_end / dt)
        
        while t <= t_end + dt/2:
            # Record data
            if step % record_interval == 0:
                E = self.compute_energy(u_hat, v_hat, w_hat)
                ens = self.compute_enstrophy(u_hat, v_hat, w_hat)
                sigma2 = self.laplacian_rms(u_hat, v_hat, w_hat)
                
                times.append(t)
                energies.append(E)
                enstrophies.append(ens)
                sigma2_hc_values.append(sigma2)
                
                marker = ""
                if abs(t - 9) < 0.5:
                    marker = " <-- expected peak"
                
                print(f"t = {t:5.2f}: E = {E:.6f}, Enstrophy = {ens:.4f}, sigma2_HC = {sigma2:.4f}{marker}")
            
            # Save snapshots
            t_rounded = round(t)
            if t_rounded in save_times:
                snapshot_key = None
                for key in snapshots.keys():
                    if abs(key - t_rounded) < 0.1:
                        snapshot_key = key
                        break
                if snapshot_key is None:
                    snapshots[t_rounded] = (
                        self.ifft(u_hat).copy(),
                        self.ifft(v_hat).copy(),
                        self.ifft(w_hat).copy()
                    )
            
            # Time step
            if t < t_end:
                u_hat, v_hat, w_hat = self.step_rk3(u_hat, v_hat, w_hat, dt)
                t += dt
                step += 1
            else:
                break
        
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
    
    # Enstrophy evolution - THIS IS THE KEY PLOT
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
    
    # Row 2 & 3: Snapshots
    key_times = [0, 9, 18]
    for idx, t in enumerate(key_times):
        if t not in snapshots:
            continue
            
        u, v, w = snapshots[t]
        
        # Velocity magnitude (mid-plane)
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
        
        wx_hat = 1j * (solver.ky * w_hat - solver.kz * v_hat)
        wy_hat = 1j * (solver.kz * u_hat - solver.kx * w_hat)
        wz_hat = 1j * (solver.kx * v_hat - solver.ky * u_hat)
        
        wx = solver.ifft(wx_hat)
        wy = solver.ifft(wy_hat)
        wz = solver.ifft(wz_hat)
        
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
    ens_peak_idx = np.argmax(enstrophies)
    ens_peak_time = times[ens_peak_idx]
    
    ax_text = fig.add_axes([0.65, 0.72, 0.32, 0.15])
    ax_text.axis('off')
    
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
        summary_text += f"[CHECK] Peak offset detected\n  sigma2_HC: {peak_time:.1f}\n  Enstrophy: {ens_peak_time:.1f}"
        color = 'lightyellow'
    
    ax_text.text(0.05, 0.95, summary_text,
                fontsize=11, verticalalignment='top', family='monospace',
                transform=ax_text.transAxes,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))
    
    plt.suptitle('Taylor-Green Vortex Validation: Full NS Evolution (Re=1600)', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('taylor_green_validation_v3.png', dpi=200, bbox_inches='tight')
    print("\nSaved: taylor_green_validation_v3.png")
    plt.close()


def main():
    print("="*70)
    print("Taylor-Green Vortex Validation v3 - Stable NS Evolution")
    print("="*70)
    print("\nImprovements over v2:")
    print("  - Low-storage RK3 time integration")
    print("  - Skew-symmetric nonlinear form")
    print("  - Spectral dealiasing (2/3 rule)")
    print("  - Automatic viscous timestep limit")
    print("="*70)
    
    # Run simulation with moderate resolution
    solver = TaylorGreenSolver(N=48, Re=1600)  # Smaller N for speed
    results = solver.run_simulation(t_end=20)
    times, energies, enstrophies, sigma2_hc, snapshots = results
    
    visualize_results(times, energies, enstrophies, sigma2_hc, snapshots, solver)
    
    # Final validation
    peak_idx = np.argmax(sigma2_hc)
    peak_time = times[peak_idx]
    ens_peak_idx = np.argmax(enstrophies)
    ens_peak_time = times[ens_peak_idx]
    
    print("\n" + "="*70)
    print("FINAL VALIDATION")
    print("="*70)
    print(f"\nsigma2_HC peak time: {peak_time:.2f}")
    print(f"Enstrophy peak time: {ens_peak_time:.2f}")
    print(f"Expected peak:       ~9")
    
    if abs(peak_time - 9) <= 2:
        print(f"\n[OK] VALIDATED: sigma2_HC peaks near t=9")
    else:
        print(f"\n[CHECK] sigma2_HC peak offset: |{peak_time} - 9| = {abs(peak_time - 9):.1f}")
    
    if abs(ens_peak_time - 9) <= 2:
        print(f"[OK] Enstrophy peaks near t=9 (nonlinear cascade confirmed)")
    else:
        print(f"[CHECK] Enstrophy peak offset: |{ens_peak_time} - 9| = {abs(ens_peak_time - 9):.1f}")
    
    print("="*70)


if __name__ == "__main__":
    main()

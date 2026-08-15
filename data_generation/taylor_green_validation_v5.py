"""
Taylor-Green Vortex Validation v5 - Optimized
Shorter simulation (t=12), optimized for speed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 10


def run_taylor_green(N=32, Re=1600, t_end=12, dt=0.002):
    """
    Optimized Taylor-Green solver
    Returns: times, energies, enstrophies, sigma2_hc, snapshots
    """
    # Setup
    L = 2 * np.pi
    nu = 1.0 / Re
    dx = L / N
    
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Wavenumbers
    k = np.fft.fftfreq(N, L/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1.0
    
    # Dealiasing
    kmax = np.max(np.abs(k))
    dealias = (np.abs(kx) < 2/3*kmax) & (np.abs(ky) < 2/3*kmax) & (np.abs(kz) < 2/3*kmax)
    
    # Initial condition
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    # Storage
    times = []
    energies = []
    enstrophies = []
    sigma2_hc_values = []
    snapshots = {}
    
    save_times = [0, 3, 6, 9, 12]
    snapshot_saved = {t: False for t in save_times}
    
    record_interval = int(0.25 / dt)  # Record every 0.25 time units
    
    print(f"Taylor-Green N={N}^3, Re={Re}, dt={dt}, t_end={t_end}")
    print(f"Steps: {int(t_end/dt)}, recording every {record_interval} steps")
    print()
    
    t = 0
    step = 0
    
    while t <= t_end + dt/2:
        # Record
        if step % record_interval == 0:
            # Energy
            E = 0.5 * np.mean(np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
            
            # Enstrophy
            wx_h = 1j * (ky * w_hat - kz * v_hat)
            wy_h = 1j * (kz * u_hat - kx * w_hat)
            wz_h = 1j * (kx * v_hat - ky * u_hat)
            ens = 0.5 * np.mean(np.abs(wx_h)**2 + np.abs(wy_h)**2 + np.abs(wz_h)**2)
            
            # sigma2_HC
            lap_u = np.fft.ifftn(-k2 * u_hat).real
            lap_v = np.fft.ifftn(-k2 * v_hat).real
            lap_w = np.fft.ifftn(-k2 * w_hat).real
            sigma2 = np.sqrt(np.mean(lap_u**2 + lap_v**2 + lap_w**2))
            
            times.append(t)
            energies.append(E)
            enstrophies.append(ens)
            sigma2_hc_values.append(sigma2)
            
            marker = " <--" if abs(t - 9) < 0.2 else ""
            print(f"t={t:5.2f}: E={E:.4f}, Ens={ens:.2f}, sigma2={sigma2:.4f}{marker}")
        
        # Save snapshots
        for t_target in save_times:
            if abs(t - t_target) < dt and not snapshot_saved[t_target]:
                snapshots[t_target] = (
                    np.fft.ifftn(u_hat).real.copy(),
                    np.fft.ifftn(v_hat).real.copy(),
                    np.fft.ifftn(w_hat).real.copy()
                )
                snapshot_saved[t_target] = True
        
        # RK2 step
        def get_rhs(uh, vh, wh):
            uh = uh * dealias
            vh = vh * dealias
            wh = wh * dealias
            
            u = np.fft.ifftn(uh).real
            v = np.fft.ifftn(vh).real
            w = np.fft.ifftn(wh).real
            
            # Derivatives
            ux = np.fft.ifftn(1j * kx * uh).real
            uy = np.fft.ifftn(1j * ky * uh).real
            uz = np.fft.ifftn(1j * kz * uh).real
            vx = np.fft.ifftn(1j * kx * vh).real
            vy = np.fft.ifftn(1j * ky * vh).real
            vz = np.fft.ifftn(1j * kz * vh).real
            wx = np.fft.ifftn(1j * kx * wh).real
            wy = np.fft.ifftn(1j * ky * wh).real
            wz = np.fft.ifftn(1j * kz * wh).real
            
            # Convection
            cu = u * ux + v * uy + w * uz
            cv = u * vx + v * vy + w * vz
            cw = u * wx + v * wy + w * wz
            
            cu_h = np.fft.fftn(cu) * dealias
            cv_h = np.fft.fftn(cv) * dealias
            cw_h = np.fft.fftn(cw) * dealias
            
            # Viscosity
            vu_h = -nu * k2 * uh
            vv_h = -nu * k2 * vh
            vw_h = -nu * k2 * wh
            
            # RHS without pressure
            ru = -cu_h + vu_h
            rv = -cv_h + vv_h
            rw = -cw_h + vw_h
            
            # Pressure projection
            div = kx * ru + ky * rv + kz * rw
            p = div / k2
            p[0,0,0] = 0
            
            return ru - 1j*kx*p, rv - 1j*ky*p, rw - 1j*kz*p
        
        # RK2 integration
        k1_u, k1_v, k1_w = get_rhs(u_hat, v_hat, w_hat)
        
        u2 = u_hat + dt * k1_u
        v2 = v_hat + dt * k1_v
        w2 = w_hat + dt * k1_w
        
        k2_u, k2_v, k2_w = get_rhs(u2, v2, w2)
        
        u_hat = u_hat + 0.5 * dt * (k1_u + k2_u)
        v_hat = v_hat + 0.5 * dt * (k1_v + k2_v)
        w_hat = w_hat + 0.5 * dt * (k1_w + k2_w)
        
        t += dt
        step += 1
        
        if step % 1000 == 0:
            print(f"  Progress: {t:.1f}/{t_end} ({100*t/t_end:.0f}%)")
    
    print(f"\nCompleted: {step} steps")
    return (np.array(times), np.array(energies), np.array(enstrophies), 
            np.array(sigma2_hc_values), snapshots, kx, ky, kz)


def visualize(times, energies, enstrophies, sigma2_hc, snapshots, kx, ky, kz):
    """Create visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Time evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, sigma2_hc, 'o-', linewidth=2.5, markersize=8, color='darkred')
    ax1.axvline(9, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel(r'$\sigma^2_{HC}$')
    ax1.set_title(r'$\sigma^2_{HC}$ Evolution', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(times, enstrophies, 's-', linewidth=2.5, markersize=8, color='darkblue')
    ax2.axvline(9, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Enstrophy (log)')
    ax2.set_title('Enstrophy Evolution', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(times, energies, '^-', linewidth=2.5, markersize=8, color='darkgreen')
    ax3.axvline(9, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Energy')
    ax3.set_title('Energy Decay', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Snapshots
    N = list(snapshots.values())[0][0].shape[0]
    mid = N // 2
    
    for idx, t in enumerate([0, 9, 12]):
        if t not in snapshots:
            continue
        u, v, w = snapshots[t]
        
        # Velocity
        vel = np.sqrt(u[:,:,mid]**2 + v[:,:,mid]**2 + w[:,:,mid]**2)
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(vel.T, origin='lower', cmap='hot', extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_title(f't={t}: |u|', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Vorticity
        uh = np.fft.fftn(u)
        vh = np.fft.fftn(v)
        wh = np.fft.fftn(w)
        wx = np.fft.ifftn(1j * (ky * wh - kz * vh)).real
        wy = np.fft.ifftn(1j * (kz * uh - kx * wh)).real
        wz = np.fft.ifftn(1j * (kx * vh - ky * uh)).real
        vort = np.sqrt(wx[:,:,mid]**2 + wy[:,:,mid]**2 + wz[:,:,mid]**2)
        
        ax = fig.add_subplot(gs[2, idx])
        im = ax.imshow(vort.T, origin='lower', cmap='Blues', extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_title(f't={t}: |ω|', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Summary
    peak_idx = np.argmax(sigma2_hc)
    peak_time = times[peak_idx]
    ens_peak_idx = np.argmax(enstrophies)
    ens_peak_time = times[ens_peak_idx]
    
    fig.text(0.5, 0.02, 
             f'VALIDATION: sigma2_HC peak at t={peak_time:.1f}, Enstrophy peak at t={ens_peak_time:.1f} (expected ~9)',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', 
                      facecolor='lightgreen' if abs(peak_time-9)<=2 else 'lightyellow'))
    
    plt.suptitle('Taylor-Green Vortex: Full NS Evolution (Re=1600, N=32)', fontsize=14, fontweight='bold')
    plt.savefig('taylor_green_validation_v5.png', dpi=200, bbox_inches='tight')
    print("Saved: taylor_green_validation_v5.png")
    plt.close()
    
    return peak_time, ens_peak_time


def main():
    print("="*70)
    print("Taylor-Green Vortex Validation v5 - Optimized")
    print("="*70)
    
    # Run with moderate settings
    results = run_taylor_green(N=32, Re=1600, t_end=12, dt=0.002)
    times, energies, enstrophies, sigma2_hc, snapshots, kx, ky, kz = results
    
    # Visualize
    peak_time, ens_peak_time = visualize(times, energies, enstrophies, sigma2_hc, snapshots, kx, ky, kz)
    
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    print(f"sigma2_HC peak: t = {peak_time:.2f}")
    print(f"Enstrophy peak: t = {ens_peak_time:.2f}")
    print(f"Expected:       t ≈ 9")
    
    if abs(peak_time - 9) <= 2:
        print("\n[OK] VALIDATED: sigma2_HC correlates with small-scale generation")
    else:
        print(f"\n[CHECK] Peak offset: |{peak_time} - 9| = {abs(peak_time - 9):.1f}")
    
    if abs(ens_peak_time - 9) <= 2:
        print("[OK] Enstrophy peaks near t=9 (nonlinear cascade confirmed)")
    else:
        print(f"[CHECK] Enstrophy offset: |{ens_peak_time} - 9| = {abs(ens_peak_time - 9):.1f}")
    
    print("="*70)


if __name__ == "__main__":
    main()

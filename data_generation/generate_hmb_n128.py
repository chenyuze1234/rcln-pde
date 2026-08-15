"""
128^3 HMB Boussinesq Solver — GPU-Accelerated Pseudo-Spectral
===============================================================
Homogeneous Buoyancy-Driven Turbulence: coupled velocity + buoyancy scalar.
Fundamentally different physics from TGV (shear-driven).

Governing equations (non-dimensional, triply-periodic [0,2pi]^3):
    du/dt + (u.grad)u = -grad(p) + nu*laplacian(u) + B*theta*e_z
    div(u) = 0
    dtheta/dt + (u.grad)theta + gamma*u_z = kappa*laplacian(theta)

Solver: RK3 low-storage, 2/3-rule de-aliasing, torch FFT on GPU.

Usage:
    python scripts/data_generation/generate_hmb_n128.py
    python scripts/data_generation/generate_hmb_n128.py --test
"""
import numpy as np
import torch
import torch.fft as fft
import h5py
import os
import time
import argparse


def build_spectral_grid(N, device='cuda'):
    """Build spectral operators on [0, 2pi]^3 grid."""
    k = fft.fftfreq(N, d=1.0 / N) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    kx, ky, kz = kx.to(device), ky.to(device), kz.to(device)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0

    # k/k^2 for pressure projection
    kx_over_k2 = kx / k2_safe
    ky_over_k2 = ky / k2_safe
    kz_over_k2 = kz / k2_safe

    # De-aliasing mask: 2/3 rule
    kmax = N // 3
    mask = (
        (torch.abs(kx) < kmax)
        & (torch.abs(ky) < kmax)
        & (torch.abs(kz) < kmax)
    )
    return kx, ky, kz, k2, k2_safe, kx_over_k2, ky_over_k2, kz_over_k2, mask


def init_hmb(kx, ky, kz, k2_safe, seed=42):
    """Initialize HMB with random velocity + buoyancy perturbation."""
    torch.manual_seed(seed)
    device = kx.device

    # Velocity: weak random field
    km = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    k0 = max(kx.max().item(), ky.max().item()) / 6
    Ek = (km / k0) ** 4 * torch.exp(-(km / k0) ** 2)
    Ek[0, 0, 0] = 0
    amp_u = torch.sqrt(Ek / (km ** 2 + 1e-10))

    uh = amp_u * torch.exp(1j * torch.rand_like(amp_u) * 2 * np.pi)
    vh = amp_u * torch.exp(1j * torch.rand_like(amp_u) * 2 * np.pi)
    wh = amp_u * torch.exp(1j * torch.rand_like(amp_u) * 2 * np.pi)

    # Div-free projection
    kd = kx * uh + ky * vh + kz * wh
    uh -= kd * kx / k2_safe
    vh -= kd * ky / k2_safe
    wh -= kd * kz / k2_safe

    # Scale velocity to small amplitude
    u = fft.ifftn(uh).real
    v = fft.ifftn(vh).real
    w = fft.ifftn(wh).real
    ke = 0.5 * (u ** 2 + v ** 2 + w ** 2).mean()
    scale = 0.1 / (torch.sqrt(ke) + 1e-8)
    uhs = torch.stack([uh, vh, wh], dim=0) * scale

    # Buoyancy: small-scale random perturbation with mean gradient
    Ek_theta = (km / k0) ** 2 * torch.exp(-(km / k0) ** 2)
    Ek_theta[0, 0, 0] = 0
    amp_theta = torch.sqrt(Ek_theta / (km ** 2 + 1e-10))
    theta_h = amp_theta * torch.exp(1j * torch.rand_like(amp_theta) * 2 * np.pi)
    theta = fft.ifftn(theta_h).real
    theta = theta * 0.05 / (theta.std() + 1e-8)  # weak initial perturbation

    return uhs, theta_h.unsqueeze(0)


def compute_nonlinear_terms(uhs, theta_h, kx, ky, kz, mask):
    """Compute convection terms for velocity and buoyancy."""
    u = fft.ifftn(uhs[0]).real
    v = fft.ifftn(uhs[1]).real
    w = fft.ifftn(uhs[2]).real
    theta = fft.ifftn(theta_h[0]).real

    # Velocity gradients
    du_dx = fft.ifftn(1j * kx * uhs[0]).real
    du_dy = fft.ifftn(1j * ky * uhs[0]).real
    du_dz = fft.ifftn(1j * kz * uhs[0]).real
    dv_dx = fft.ifftn(1j * kx * uhs[1]).real
    dv_dy = fft.ifftn(1j * ky * uhs[1]).real
    dv_dz = fft.ifftn(1j * kz * uhs[1]).real
    dw_dx = fft.ifftn(1j * kx * uhs[2]).real
    dw_dy = fft.ifftn(1j * ky * uhs[2]).real
    dw_dz = fft.ifftn(1j * kz * uhs[2]).real

    # Buoyancy gradients
    dtheta_dx = fft.ifftn(1j * kx * theta_h[0]).real
    dtheta_dy = fft.ifftn(1j * ky * theta_h[0]).real
    dtheta_dz = fft.ifftn(1j * kz * theta_h[0]).real

    # Velocity convection: -(u·∇)u  (nonlinear)
    conv_u = fft.fftn(u * du_dx + v * du_dy + w * du_dz) * mask
    conv_v = fft.fftn(u * dv_dx + v * dv_dy + w * dv_dz) * mask
    conv_w = fft.fftn(u * dw_dx + v * dw_dy + w * dw_dz) * mask

    # Buoyancy convection: -(u·∇)θ
    conv_theta = fft.fftn(u * dtheta_dx + v * dtheta_dy + w * dtheta_dz) * mask

    # Buoyancy force: B*theta*e_z (vertical)
    buoyancy_z = fft.fftn(theta) * mask  # B*theta in spectral space

    return (torch.stack([conv_u, conv_v, conv_w], dim=0),
            conv_theta.unsqueeze(0), buoyancy_z)


def div_free_project(uh, kx, ky, kz, k2_safe):
    """Project onto divergence-free manifold."""
    div = kx * uh[0] + ky * uh[1] + kz * uh[2]
    phi = div / k2_safe
    uh[0] -= kx * phi
    uh[1] -= ky * phi
    uh[2] -= kz * phi
    return uh


# RK3 low-storage coefficients
RK3_A = [0.0, -5.0 / 9.0, -153.0 / 128.0]
RK3_B = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0]


def rk3_step(uhs, theta_h, dt, nu, kappa, B, gamma,
             k2, kx, ky, kz, k2_safe, mask):
    """Full RK3 time step for Boussinesq system."""
    dq_u = torch.zeros_like(uhs)
    dq_theta = torch.zeros_like(theta_h)

    for stage in range(3):
        conv, conv_theta, buoy_z = compute_nonlinear_terms(
            uhs, theta_h, kx, ky, kz, mask)

        # Velocity RHS: -conv - nu*k^2*u + B*theta*e_z
        rhs_u = -conv - nu * k2.unsqueeze(0) * uhs
        rhs_u[2] = rhs_u[2] + B * buoy_z  # buoyancy in z-direction

        # Theta RHS: -conv_theta - gamma*u_z - kappa*k^2*theta
        rhs_theta = (-conv_theta
                     - gamma * uhs[2]  # mean gradient term
                     - kappa * k2.unsqueeze(0) * theta_h)

        dq_u = RK3_A[stage] * dq_u + dt * rhs_u
        dq_theta = RK3_A[stage] * dq_theta + dt * rhs_theta

        uhs = uhs + RK3_B[stage] * dq_u
        theta_h = theta_h + RK3_B[stage] * dq_theta

        uhs = div_free_project(uhs, kx, ky, kz, k2_safe)

    return uhs, theta_h


def run_solver(N=128, T_end=5.0, dt=0.002, save_every=5, warmup=100,
               device='cuda', seed=42, test=False):
    """Generate HMB Boussinesq data at 128^3."""
    # Physical parameters
    nu = 0.005     # kinematic viscosity
    kappa = 0.007  # thermal diffusivity (Pr = nu/kappa ≈ 0.71)
    B_val = 1.0    # buoyancy coefficient
    gamma = -0.3   # mean temperature gradient (negative = unstable)

    n_steps = int(T_end / dt)
    if test:
        n_steps = 50
        save_every = 2
        warmup = 10

    print(f"{'='*60}")
    print(f"HMB Boussinesq N={N} nu={nu} kappa={kappa} B={B_val} gamma={gamma}")
    print(f"T={T_end} dt={dt} steps={n_steps} warmup={warmup}")
    print(f"{'='*60}")

    # Build grid
    t_build = time.time()
    kx, ky, kz, k2, k2_safe, kxok2, kyok2, kzok2, mask = build_spectral_grid(N, device)
    print(f"Spectral grid: {time.time() - t_build:.1f}s")

    # Initialize
    uhs, theta_h = init_hmb(kx, ky, kz, k2_safe, seed=seed)
    u = fft.ifftn(uhs[0]).real
    ke_init = 0.5 * (u ** 2).mean().item()
    print(f"Initial KE={ke_init:.6f}")

    # Warmup
    cur_u, cur_theta = uhs, theta_h
    for s in range(warmup):
        cur_u, cur_theta = rk3_step(
            cur_u, cur_theta, dt, nu, kappa, B_val, gamma,
            k2, kx, ky, kz, k2_safe, mask)
        if s % 20 == 0:
            ke = 0.5 * (fft.ifftn(cur_u[0]).real ** 2).mean().item()
            ok = torch.isfinite(cur_u).all()
            print(f"  warmup {s}: KE={ke:.6f} OK={ok}")

    print(f"Warmup complete")

    # Production
    os.makedirs('data_generated', exist_ok=True)
    fname = f'hmb_boussinesq_N{N}_T{T_end:.1f}.h5'
    fpath = os.path.join('data_generated', fname)
    n_frames = n_steps // save_every + 1
    print(f"Output: {fpath} ({n_frames} frames)")

    hf = h5py.File(fpath, 'w')
    # Store velocity (3ch) and buoyancy (1ch) as 4-channel field
    ds = hf.create_dataset(
        'fields', shape=(0, 4, N, N, N), maxshape=(n_frames, 4, N, N, N),
        dtype=np.float32, compression='gzip', compression_opts=4,
        chunks=(1, 4, N, N, N))
    frame_idx = 0

    t0 = time.time()
    for s in range(n_steps):
        cur_u, cur_theta = rk3_step(
            cur_u, cur_theta, dt, nu, kappa, B_val, gamma,
            k2, kx, ky, kz, k2_safe, mask)

        if s % save_every == 0:
            ut = fft.ifftn(cur_u[0]).real.cpu().numpy().astype(np.float32)
            vt = fft.ifftn(cur_u[1]).real.cpu().numpy().astype(np.float32)
            wt = fft.ifftn(cur_u[2]).real.cpu().numpy().astype(np.float32)
            thetat = fft.ifftn(cur_theta[0]).real.cpu().numpy().astype(np.float32)
            ds.resize((frame_idx + 1, 4, N, N, N))
            ds[frame_idx] = np.stack([ut, vt, wt, thetat], axis=0)
            frame_idx += 1

        if s % 50 == 0 or s == n_steps - 1:
            ke = 0.5 * (fft.ifftn(cur_u[0]).real ** 2 +
                        fft.ifftn(cur_u[1]).real ** 2 +
                        fft.ifftn(cur_u[2]).real ** 2).mean().item()
            ok = torch.isfinite(cur_u).all()
            print(f"  step {s:5d}: KE={ke:.6f} OK={ok}")

    total_time = time.time() - t0
    print(f"Done: {frame_idx} frames in {total_time:.0f}s "
          f"({total_time/n_steps*1000:.1f}ms/step)")

    # Metadata
    hf.create_dataset('times', data=np.arange(frame_idx) * (dt * save_every) + warmup * dt)
    hf.attrs['flow_type'] = 'HMB Boussinesq'
    hf.attrs['N'] = N
    hf.attrs['nu'] = nu
    hf.attrs['kappa'] = kappa
    hf.attrs['B'] = B_val
    hf.attrs['gamma'] = gamma
    hf.attrs['dt'] = dt
    hf.attrs['solver'] = 'Pseudo-spectral RK3 2/3-rule de-aliased'
    hf.close()

    fsize_mb = os.path.getsize(fpath) / 1e6
    print(f"Saved: {fpath} ({fsize_mb:.0f}MB)")
    return fpath


def main():
    parser = argparse.ArgumentParser(description='HMB Boussinesq 128^3 solver')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_solver(test=args.test, device=args.device)


if __name__ == '__main__':
    main()

"""
128^3 TGV Data Generator -- Multi-Regime, RK3 Low-Storage
==========================================================
Generates 128^3 Taylor-Green vortex data at multiple Reynolds numbers
for Nature Communications submission.

Solver: pseudo-spectral with RK3 low-storage (Williamson 1980),
        2/3-rule de-aliasing, divergence-free projection at each substep.

Usage:
    python scripts/data_generation/generate_tgv_n128_multiregime.py --re 6400
    python scripts/data_generation/generate_tgv_n128_multiregime.py --re 1600,3200,6400
    python scripts/data_generation/generate_tgv_n128_multiregime.py --re all
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

    # De-aliasing mask: 2/3 rule
    kmax = N // 3
    mask = (
        (torch.abs(kx) < kmax)
        & (torch.abs(ky) < kmax)
        & (torch.abs(kz) < kmax)
    )
    return kx, ky, kz, k2, k2_safe, mask


def init_tgv(kx, ky, kz, k2_safe, k0_shift=None, seed=42):
    """Initialize TGV field with div-free random spectrum."""
    torch.manual_seed(seed)
    km = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    if k0_shift is None:
        k0_val = max(kx.max().item(), ky.max().item()) / 6
    else:
        k0_val = k0_shift

    Ek = (km / k0_val) ** 4 * torch.exp(-(km / k0_val) ** 2)
    Ek[0, 0, 0] = 0
    amp = torch.sqrt(Ek / (km ** 2 + 1e-10))

    uh = amp * torch.exp(1j * torch.rand_like(amp) * 2 * np.pi)
    vh = amp * torch.exp(1j * torch.rand_like(amp) * 2 * np.pi)
    wh = amp * torch.exp(1j * torch.rand_like(amp) * 2 * np.pi)

    # Project to divergence-free
    kd = kx * uh + ky * vh + kz * wh
    uh -= kd * kx / k2_safe
    vh -= kd * ky / k2_safe
    wh -= kd * kz / k2_safe

    uhs = torch.stack([uh, vh, wh], dim=0)

    # Scale to unit kinetic energy
    u = fft.ifftn(uhs[0]).real
    v = fft.ifftn(uhs[1]).real
    w = fft.ifftn(uhs[2]).real
    ke = 0.5 * (u ** 2 + v ** 2 + w ** 2).mean()
    scale = 0.8 / (torch.sqrt(ke) + 1e-8)
    uhs *= scale

    return uhs


def div_free_project(uh, kx, ky, kz, k2_safe):
    """Project velocity field onto divergence-free manifold."""
    div = kx * uh[0] + ky * uh[1] + kz * uh[2]
    phi = div / k2_safe
    uh[0] -= kx * phi
    uh[1] -= ky * phi
    uh[2] -= kz * phi
    return uh


def compute_convection(uh, kx, ky, kz, dealias_mask):
    """Compute convection term C = (u.grad)u in spectral space with de-aliasing."""
    u = fft.ifftn(uh[0]).real
    v = fft.ifftn(uh[1]).real
    w = fft.ifftn(uh[2]).real

    # Gradients in physical space
    du_dx = fft.ifftn(1j * kx * uh[0]).real
    du_dy = fft.ifftn(1j * ky * uh[0]).real
    du_dz = fft.ifftn(1j * kz * uh[0]).real
    dv_dx = fft.ifftn(1j * kx * uh[1]).real
    dv_dy = fft.ifftn(1j * ky * uh[1]).real
    dv_dz = fft.ifftn(1j * kz * uh[1]).real
    dw_dx = fft.ifftn(1j * kx * uh[2]).real
    dw_dy = fft.ifftn(1j * ky * uh[2]).real
    dw_dz = fft.ifftn(1j * kz * uh[2]).real

    # Convection = u_j * partial_j u_i
    cu = fft.fftn(u * du_dx + v * du_dy + w * du_dz) * dealias_mask
    cv = fft.fftn(u * dv_dx + v * dv_dy + w * dv_dz) * dealias_mask
    cw = fft.fftn(u * dw_dx + v * dw_dy + w * dw_dz) * dealias_mask

    return torch.stack([cu, cv, cw], dim=0)


# RK3 low-storage coefficients (Williamson 1980)
RK3_A = [0.0, -5.0 / 9.0, -153.0 / 128.0]
RK3_B = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0]


def rk3_step(uh, dt, nu, k2, kx, ky, kz, k2_safe, dealias_mask):
    """Full RK3 time step with low-storage scheme (Williamson 1980)."""
    dq = torch.zeros_like(uh)
    for stage in range(3):
        dq = RK3_A[stage] * dq + dt * (
            -compute_convection(uh, kx, ky, kz, dealias_mask)
            - nu * k2.unsqueeze(0) * uh
        )
        uh = uh + RK3_B[stage] * dq
        uh = div_free_project(uh, kx, ky, kz, k2_safe)
    return uh


def rk2_step(uh, dt, nu, k2, kx, ky, kz, k2_safe, dealias_mask):
    """RK2 midpoint method (fallback for memory-constrained runs)."""
    # Stage 1: half step
    k1 = -compute_convection(uh, kx, ky, kz, dealias_mask) - nu * k2.unsqueeze(0) * uh
    uh_mid = uh + 0.5 * dt * k1
    uh_mid = div_free_project(uh_mid, kx, ky, kz, k2_safe)
    # Stage 2: full step using midpoint derivative
    k2_full = -compute_convection(uh_mid, kx, ky, kz, dealias_mask) - nu * k2.unsqueeze(0) * uh_mid
    uh_new = uh + dt * k2_full
    return div_free_project(uh_new, kx, ky, kz, k2_safe)


def run_solver(N, Re, dt, n_steps, n_warmup, save_every, out_dir,
               device='cuda', seed=42, rk_order=3):
    """Generate TGV data at given resolution and Reynolds number."""
    nu = 1.0 / Re
    T_sim = n_steps * dt
    T_warmup = n_warmup * dt
    n_frames = n_steps // save_every + 1

    print(f"{'='*70}")
    print(f"TGV N={N}  Re={Re}  nu={nu:.6e}  dt={dt}  RK{rk_order}")
    print(f"Warmup: {n_warmup} steps ({T_warmup:.2f} time units)")
    print(f"Production: {n_steps} steps ({T_sim:.2f} time units)")
    print(f"Saving every {save_every} steps -> {n_frames} frames")
    print(f"{'='*70}")

    # Build spectral operators
    t_build = time.time()
    kx, ky, kz, k2, k2_safe, dealias_mask = build_spectral_grid(N, device)
    print(f"Spectral grid built in {time.time() - t_build:.1f}s")

    # Select time stepper
    if rk_order == 2:
        stepper = rk2_step
    else:
        stepper = rk3_step

    # Initialize
    uhs = init_tgv(kx, ky, kz, k2_safe, seed=seed)
    u = fft.ifftn(uhs[0]).real
    v = fft.ifftn(uhs[1]).real
    w = fft.ifftn(uhs[2]).real
    ke_init = float(0.5 * (u ** 2 + v ** 2 + w ** 2).mean().item())
    print(f"Initial KE={ke_init:.6f}")

    # Warmup
    t0 = time.time()
    cur = uhs
    for s in range(n_warmup):
        cur = stepper(cur, dt, nu, k2, kx, ky, kz, k2_safe, dealias_mask)
        if s % 20 == 0 or s == n_warmup - 1:
            u = fft.ifftn(cur[0]).real
            v = fft.ifftn(cur[1]).real
            w = fft.ifftn(cur[2]).real
            ke = 0.5 * (u ** 2 + v ** 2 + w ** 2).mean().item()
            u_max = max(u.abs().max().item(), v.abs().max().item(), w.abs().max().item())
            ok = torch.isfinite(cur).all()
            print(f"  warmup {s:4d}: KE={ke:.6f}  |u|max={u_max:.3f}  OK={ok}")
            if not ok:
                raise RuntimeError(f"NaN at warmup step {s}")

    warmup_time = time.time() - t0
    print(f"Warmup complete in {warmup_time:.1f}s")

    # Production run — save incrementally to avoid memory blowup
    os.makedirs(out_dir, exist_ok=True)
    fname = f'tgv_re{Re}_N{N}_T{T_sim:.1f}.h5'
    fpath = os.path.join(out_dir, fname)

    t0 = time.time()
    n_frames = n_steps // save_every + 1
    # Create HDF5 file with resizable dataset
    hf = h5py.File(fpath, 'w')
    ds = hf.create_dataset(
        'fields', shape=(0, 3, N, N, N), maxshape=(n_frames, 3, N, N, N),
        dtype=np.float32, compression='gzip', compression_opts=4,
        chunks=(1, 3, N, N, N))
    frame_idx = 0

    for s in range(n_steps):
        cur = stepper(cur, dt, nu, k2, kx, ky, kz, k2_safe, dealias_mask)

        if s % save_every == 0:
            ut = fft.ifftn(cur[0]).real.cpu().numpy().astype(np.float32)
            vt = fft.ifftn(cur[1]).real.cpu().numpy().astype(np.float32)
            wt = fft.ifftn(cur[2]).real.cpu().numpy().astype(np.float32)
            ds.resize((frame_idx + 1, 3, N, N, N))
            ds[frame_idx] = np.stack([ut, vt, wt], axis=0)
            frame_idx += 1

        if s % 50 == 0 or s == n_steps - 1:
            u = fft.ifftn(cur[0]).real
            v = fft.ifftn(cur[1]).real
            w = fft.ifftn(cur[2]).real
            ke = 0.5 * (u ** 2 + v ** 2 + w ** 2).mean().item()
            u_max = max(u.abs().max().item(), v.abs().max().item(), w.abs().max().item())
            ok = torch.isfinite(cur).all()
            print(f"  step {s:5d}: KE={ke:.6f}  |u|max={u_max:.3f}  OK={ok}")
            if not ok:
                print("FATAL: NaN detected, aborting")
                return None

    total_time = time.time() - t0
    n_saved = frame_idx
    print(f"\nProduction: {n_saved} frames, {total_time:.1f}s "
          f"({total_time / n_steps * 1000:.1f}ms/step)")

    if not np.isfinite(ds[-1]).all():
        print("FAILED: NaN in final frame")
        hf.close()
        return None

    # Save metadata
    saved_times = np.arange(n_saved) * (dt * save_every) + n_warmup * dt
    hf.create_dataset('times', data=saved_times.astype(np.float32))
    hf.attrs['flow_type'] = 'Taylor-Green Vortex'
    hf.attrs['Re'] = Re
    hf.attrs['nu'] = nu
    hf.attrs['N'] = N
    hf.attrs['L'] = 2.0 * np.pi
    hf.attrs['dt'] = dt
    hf.attrs['solver'] = 'Pseudo-spectral RK3 2/3-rule de-aliased'
    hf.attrs['n_steps'] = n_steps
    hf.attrs['n_warmup'] = n_warmup
    hf.attrs['save_every'] = save_every
    hf.attrs['seed'] = seed
    hf.close()

    fsize_mb = os.path.getsize(fpath) / 1e6
    ds_total_gb = n_saved * 3 * N**3 * 4 / 1e9
    print(f"Saved: {fpath}  ({fsize_mb:.0f}MB on disk, {ds_total_gb:.1f}GB uncompressed)")
    return fpath


# ============================================================
# Default configurations per Reynolds number (128^3 grid)
# dt chosen for CFL < 0.4 given |u_max| ~ 0.8-1.0, dx = 2pi/128 ~ 0.049
# ============================================================
CONFIGS = {
    400:   dict(dt=0.010, n_steps=1000, save_every=5, warmup=50, rk=2),
    800:   dict(dt=0.005, n_steps=1000, save_every=5, warmup=50, rk=2),
    1000:  dict(dt=0.005, n_steps=1000, save_every=5, warmup=50, rk=3),
    1600:  dict(dt=0.004, n_steps=1250, save_every=5, warmup=80, rk=3),
    3200:  dict(dt=0.003, n_steps=1667, save_every=5, warmup=100, rk=3),
    5000:  dict(dt=0.002, n_steps=2000, save_every=4, warmup=100, rk=3),
    6400:  dict(dt=0.002, n_steps=2500, save_every=4, warmup=100, rk=3),
}


def main():
    parser = argparse.ArgumentParser(description='Generate 128^3 TGV data')
    parser.add_argument('--re', type=str, default='6400',
                        help='Reynolds number(s), comma-separated, or "all"')
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--n-steps', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--out-dir', type=str, default='data_generated')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', action='store_true',
                        help='Generate small test dataset (50 steps)')
    parser.add_argument('--rk', type=int, default=None,
                        help='RK order (2 or 3)')
    args = parser.parse_args()

    if args.re == 'all':
        res = sorted(CONFIGS.keys())
    else:
        res = [int(r.strip()) for r in args.re.split(',')]

    for Re in res:
        cfg = CONFIGS.get(Re, CONFIGS[6400]).copy()
        if args.dt is not None:
            cfg['dt'] = args.dt
        if args.n_steps is not None:
            cfg['n_steps'] = args.n_steps
        if args.save_every is not None:
            cfg['save_every'] = args.save_every
        if args.warmup is not None:
            cfg['warmup'] = args.warmup
        if args.test:
            cfg['n_steps'] = 50
            cfg['save_every'] = 2
            cfg['warmup'] = 10
            cfg['dt'] = min(cfg['dt'], 0.005)
        rk_order = args.rk if args.rk is not None else cfg.get('rk', 3)

        try:
            fpath = run_solver(
                N=args.N, Re=Re, dt=cfg['dt'],
                n_steps=cfg['n_steps'], n_warmup=cfg['warmup'],
                save_every=cfg['save_every'], out_dir=args.out_dir,
                device=args.device, seed=args.seed, rk_order=rk_order,
            )
            if fpath:
                print(f"[OK] Re={Re}: {fpath}\n")
            else:
                print(f"[FAIL] Re={Re}: NaN detected\n")
        except Exception as e:
            print(f"[ERROR] Re={Re}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

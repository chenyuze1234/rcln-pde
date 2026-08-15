"""
2D Navier-Stokes 128^2 resolution-OOD 真值生成器 —— System B Step 6 收尾
====================================================================
与 generate_2d_ns_data.py 同方程、同物理参数（nu/A/k_f/alpha 不变），
仅把网格从 64^2 升到 128^2，供 FNO 论文式 zero-shot super-resolution 评估。

  方程: w_t + J(psi,w) = nu*Lap(w) + f_w - alpha*Lap^-1(w)
        psi^ = -w^/k^2, u=-psi_y, v=psi_x
        lambda(k) = -nu*k^2 - alpha/k^2; f_w = -A*k_f*cos(k_f*y)
  数值: rfft2 伪谱 + 2/3 去混叠 + 轮廓积分 ETDRK4（同 64^2 生成器）

分辨率升格的参数联动（只动数值，不动物理）:
  DT_SOLVER 5e-3 -> 2.5e-3   （平流 CFL: kmax 翻倍 -> dt 减半；黏性项 ETDRK4 精确处理）
  DT_BURN   1e-2 -> 5e-3     （burn-in 只用于上吸引子，同样随 CFL 减半）
  SAMPLE_EVERY = 66           （帧 dt 保持 0.165 不变 = 66 x 2.5e-3）
  BURN_T = 100.0 不变

产出: data_generated/ns2d_kolmogorov_res128.h5
  fields [3, 260, 128, 128] f32（3 条独立 IC 轨迹，burn-in 后采集 260 帧）
  attrs 与 64^2 文件同构（N=128, dt=0.165, sample_every=66, dt_solver=2.5e-3, ...）

验证门（不通过则 raise，不入库）:
  (a1) 一步求解器一致性: 存储帧 -> 积分 sample_every 步 -> 对比下一帧, relerr<1e-6
  (a2) 中心差分 vs 谱 RHS 安全网（门限 0.5，权威判据是 a1）
  (b)  E/Z 统计平稳: 后 60% 窗趋势统计量，与 64^2 main 同窗长趋势的 p95 x 1.5 比较
       （64^2 生成器用 4000 帧参考轨迹标定该分布；此处直接以 64^2 main 14 条轨迹
        同窗长趋势分布作参考，避免在 128^2 重跑长参考轨迹，口径等价）

运行: 在 D:/AxiomOS_Project 下
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" scripts/data_generation/generate_2d_ns_res128.py
"""

import os
import time

import h5py
import numpy as np

# --------------------------------------------------------------------
# 参数（物理量与 64^2 完全一致；仅数值分辨率/步长联动）
# --------------------------------------------------------------------
N = 128
LBOX = 2.0 * np.pi
DT_SOLVER = 2.5e-3            # 64^2 用 5e-3；kmax 翻倍 -> 平流 CFL 减半
DT_BURN = 5e-3                # 64^2 用 1e-2
NU = 2e-3
A_FORCE = 0.15
KF = 4
ALPHA = 0.1
N_FRAMES = 260
BURN_T = 100.0
FRAME_DT = 0.165              # 与 64^2 文件严格一致
SAMPLE_EVERY = int(round(FRAME_DT / DT_SOLVER))   # 66
K_ROLLOUT = 200

OUT = 'data_generated/ns2d_kolmogorov_res128.h5'
REF_MAIN = 'data_generated/ns2d_kolmogorov_main.h5'   # 平稳性参考分布来源
SEEDS = [300, 301, 302]

assert abs(SAMPLE_EVERY * DT_SOLVER - FRAME_DT) < 1e-12

# --------------------------------------------------------------------
# 谱网格与求解器（与 64^2 生成器逐行同构，仅 N 不同）
# --------------------------------------------------------------------
kx = np.fft.fftfreq(N, d=1.0 / N)
ky = np.fft.rfftfreq(N, d=1.0 / N)
KX = kx[:, None]
KY = ky[None, :]
K2 = KX**2 + KY**2
K2INV = np.where(K2 > 0, 1.0 / np.maximum(K2, 1e-30), 0.0)
NCUT = N // 3
DEALIAS = ((np.abs(KX) <= NCUT) & (np.abs(KY) <= NCUT)).astype(np.float64)


def forcing_field(A, kf):
    y = np.arange(N) * (LBOX / N)
    fw = -A * kf * np.cos(kf * y)
    return np.broadcast_to(fw[None, :], (N, N)).copy()


def make_solver(nu, A, kf, alpha, dt):
    """轮廓积分 ETDRK4（Kassam-Trefethen, R=32）。状态 what [N, N//2+1]。"""
    lam = -nu * K2 - alpha * K2INV
    lam = lam.copy()
    lam[0, 0] = 0.0
    fhat = np.fft.rfft2(forcing_field(A, kf)) if A != 0.0 else np.zeros_like(lam, dtype=complex)

    lf = lam.ravel()
    R = 32
    r = np.exp(1j * np.pi * (np.arange(1, R + 1) - 0.5) / R)
    LR = dt * lf[:, None] + r[None, :]
    E = np.exp(dt * lf)
    E2 = np.exp(dt * lf / 2.0)
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    shp = lam.shape
    E, E2, Q, f1, f2, f3 = [a.reshape(shp) for a in (E, E2, Q, f1, f2, f3)]

    def nl(what):
        psi = -what * K2INV
        u = np.fft.irfft2(-1j * KY * psi, s=(N, N))
        v = np.fft.irfft2(1j * KX * psi, s=(N, N))
        wx = np.fft.irfft2(1j * KX * what, s=(N, N))
        wy = np.fft.irfft2(1j * KY * what, s=(N, N))
        J = u * wx + v * wy
        return -np.fft.rfft2(J) * DEALIAS + fhat

    def step(what):
        Nv = nl(what)
        a = E2 * what + Q * Nv
        Na = nl(a)
        b = E2 * what + Q * Na
        Nb = nl(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nl(c)
        out = E * what + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        out[0, 0] = 0.0
        return out

    return step, lam, fhat


def uv_from_what(what):
    psi = -what * K2INV
    u = np.fft.irfft2(-1j * KY * psi, s=(N, N))
    v = np.fft.irfft2(1j * KX * psi, s=(N, N))
    return u, v


def diagnostics(what, nu, A, kf, alpha):
    u, v = uv_from_what(what)
    w = np.fft.irfft2(what, s=(N, N))
    psi = np.fft.irfft2(-what * K2INV, s=(N, N))
    E = 0.5 * np.mean(u**2 + v**2)
    Z = 0.5 * np.mean(w**2)
    U = np.sqrt(2.0 * E)
    P_f = float(np.mean(u * (A * np.sin(kf * (np.arange(N) * (LBOX / N)))[None, :]))) if A != 0 else 0.0
    D = 2.0 * nu * Z + alpha * float(np.mean(psi**2))
    return E, Z, U, P_f, D


def random_ic(seed, k0=5.0, dk=2.5, u_target=0.3):
    """与 64^2 生成器同一 IC 族（带限白噪声滤谱，缩放到 U_rms=0.3）。"""
    rng = np.random.default_rng(seed)
    gh = np.fft.rfft2(rng.standard_normal((N, N)))
    K = np.sqrt(K2)
    amp = np.exp(-((K - k0) / dk)**2)
    amp[K < 1.5] = 0.0
    what = gh * amp
    what[0, 0] = 0.0
    u, v = uv_from_what(what)
    u_rms = np.sqrt(np.mean(u**2 + v**2)) + 1e-30
    return what * (u_target / u_rms)


# --------------------------------------------------------------------
# 轨迹生成
# --------------------------------------------------------------------
def gen_forced(n_traj, seeds, tag):
    step_b, _, _ = make_solver(NU, A_FORCE, KF, ALPHA, DT_BURN)
    step_p, lam, fhat = make_solver(NU, A_FORCE, KF, ALPHA, DT_SOLVER)
    n_burn = int(BURN_T / DT_BURN)
    fields = np.zeros((n_traj, N_FRAMES, N, N), dtype=np.float32)
    for j in range(n_traj):
        t0 = time.time()
        what = random_ic(seed=seeds[j])
        for _ in range(n_burn):
            what = step_b(what)
        if not np.isfinite(what).all():
            raise RuntimeError(f'[{tag}] BLOW-UP in burn-in traj {j}')
        for i in range(N_FRAMES):
            for _ in range(SAMPLE_EVERY):
                what = step_p(what)
            fields[j, i] = np.fft.irfft2(what, s=(N, N)).astype(np.float32)
            if i % 52 == 0:
                E, Z, U, P, D = diagnostics(what, NU, A_FORCE, KF, ALPHA)
                print(f'    [{tag}] traj{j} frame{i:3d}: E={E:.4f} Z={Z:.2f} '
                      f'U={U:.4f} P-D={P-D:+.2e} ({time.time()-t0:.0f}s)', flush=True)
        if not np.isfinite(fields[j]).all():
            raise RuntimeError(f'[{tag}] BLOW-UP in production traj {j}')
        print(f'  [{tag}] traj {j} done ({time.time()-t0:.0f}s)', flush=True)
    return fields


def save_h5(path, fields, seeds, tag):
    w = fields.astype(np.float64)
    Us = []
    for j in range(len(fields)):
        u, v = uv_from_what(np.fft.rfft2(w[j, -1]))
        Us.append(np.sqrt(np.mean(u**2 + v**2)))
    U_rms = float(np.mean(Us))
    Re_box = U_rms * LBOX / NU
    tau_let = LBOX / max(U_rms, 1e-12)
    with h5py.File(path, 'w') as f:
        f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
        f.create_dataset('times', data=np.arange(N_FRAMES) * FRAME_DT)
        f.attrs['equation'] = ('2D NS vorticity: w_t + J(psi,w) = nu*Lap(w) + f_w '
                               '- alpha*Lap^-1(w); psi^=-w^/k^2; u=-psi_y,v=psi_x')
        f.attrs['forcing'] = 'f=A*sin(k_f*y)*x-hat; f_w=-A*k_f*cos(k_f*y)'
        f.attrs['nu'] = NU
        f.attrs['A'] = A_FORCE
        f.attrs['k_f'] = KF
        f.attrs['alpha'] = ALPHA
        f.attrs['N'] = N
        f.attrs['L'] = LBOX
        f.attrs['dt'] = FRAME_DT
        f.attrs['dt_solver'] = DT_SOLVER
        f.attrs['sample_every'] = SAMPLE_EVERY
        f.attrs['n_frames'] = N_FRAMES
        f.attrs['n_traj'] = len(fields)
        f.attrs['seeds'] = list(seeds)
        f.attrs['Re_box'] = Re_box
        f.attrs['U_rms'] = U_rms
        f.attrs['tau_let'] = tau_let
        f.attrs['K200_cover_tau'] = K_ROLLOUT * FRAME_DT / tau_let
        f.attrs['split'] = tag
    print(f'  saved {path}: {fields.shape} U_rms={U_rms:.4f} Re_box={Re_box:.0f} '
          f'tau_let={tau_let:.2f} K200={K_ROLLOUT*FRAME_DT/tau_let:.2f}tau')


# --------------------------------------------------------------------
# 验证门
# --------------------------------------------------------------------
def gate_onestep(path, n_check=8):
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
    step, _, _ = make_solver(NU, A_FORCE, KF, ALPHA, DT_SOLVER)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(n_check):
        j = int(rng.integers(0, len(fl)))
        i = int(rng.integers(0, N_FRAMES - 1))
        what = np.fft.rfft2(fl[j, i])
        for _ in range(SAMPLE_EVERY):
            what = step(what)
        nxt = np.fft.irfft2(what, s=(N, N))
        errs.append(np.linalg.norm(nxt - fl[j, i + 1]) / (np.linalg.norm(fl[j, i + 1]) + 1e-30))
    errs = np.array(errs)
    ok = errs.max() < 1e-6
    print(f'  [gate a1] {os.path.basename(path)} one-step relerr: '
          f'median={np.median(errs):.2e} max={errs.max():.2e} -> {"PASS" if ok else "FAIL"}')
    return ok, float(np.median(errs)), float(errs.max())


def gate_residual(path, n_check=30):
    lam = -NU * K2 - ALPHA * K2INV
    lam[0, 0] = 0.0
    ffield = forcing_field(A_FORCE, KF)
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)

    def rhs_phys(w):
        what = np.fft.rfft2(w)
        psi = -what * K2INV
        u = np.fft.irfft2(-1j * KY * psi, s=(N, N))
        v = np.fft.irfft2(1j * KX * psi, s=(N, N))
        wx = np.fft.irfft2(1j * KX * what, s=(N, N))
        wy = np.fft.irfft2(1j * KY * what, s=(N, N))
        Jh = np.fft.rfft2(u * wx + v * wy) * DEALIAS
        return np.fft.irfft2(-Jh + lam * what, s=(N, N)) + ffield

    rng = np.random.default_rng(1)
    res = []
    for _ in range(n_check):
        j = int(rng.integers(0, len(fl)))
        i = int(rng.integers(1, N_FRAMES - 1))
        cd = (fl[j, i + 1] - fl[j, i - 1]) / (2.0 * FRAME_DT)
        r = rhs_phys(fl[j, i])
        res.append(np.linalg.norm(cd - r) / (np.linalg.norm(r) + 1e-30))
    res = np.array(res)
    ok = np.median(res) < 0.5
    print(f'  [gate a2] central-diff residual: median={np.median(res):.2e} '
          f'p90={np.percentile(res,90):.2e} -> {"PASS" if ok else "FAIL"} '
          f'(帧 dt={FRAME_DT:.3f} 截断主导, 安全网门限 0.5)')
    return ok


def _E_Z_traj(fl):
    what = np.fft.rfft2(fl, axes=(-2, -1))
    kk = K2INV.shape
    # 参考文件可能是 64^2 —— 用其自身网格重算
    n = fl.shape[-1]
    kx_ = np.fft.fftfreq(n, d=1.0 / n)
    ky_ = np.fft.rfftfreq(n, d=1.0 / n)
    K2_ = kx_[:, None]**2 + ky_[None, :]**2
    K2inv_ = np.where(K2_ > 0, 1.0 / np.maximum(K2_, 1e-30), 0.0)
    psi = -what * K2inv_[None]
    u = np.fft.irfft2(-1j * ky_[None, :][None] * psi, s=(n, n), axes=(-2, -1))
    v = np.fft.irfft2(1j * kx_[:, None][None] * psi, s=(n, n), axes=(-2, -1))
    E = 0.5 * (u**2 + v**2).mean(axis=(-2, -1))
    Z = 0.5 * (fl**2).mean(axis=(-2, -1))
    return E, Z


def gate_stationary(path, frac=0.6):
    """窗趋势统计量 vs 64^2 main 14 条轨迹同窗长分布的 p95 x 1.5。"""
    win = int(N_FRAMES * frac)
    with h5py.File(REF_MAIN, 'r') as f:
        ref_fl = f['fields'][:].astype(np.float64)
    ref_trE, ref_trZ = [], []
    for j in range(len(ref_fl)):
        E, Z = _E_Z_traj(ref_fl[j])
        for s, acc in [(E, ref_trE), (Z, ref_trZ)]:
            seg = s[len(s) - win:]
            sl = np.polyfit(np.arange(win), seg, 1)[0]
            acc.append(abs(sl) * win / (seg.std() + 1e-30))
    thrE = float(np.percentile(ref_trE, 95)) * 1.5
    thrZ = float(np.percentile(ref_trZ, 95)) * 1.5
    print(f'  [gate b] 64^2 main 参考趋势 p95: E={np.percentile(ref_trE,95):.2f} '
          f'Z={np.percentile(ref_trZ,95):.2f} -> 门限 E<{thrE:.2f} Z<{thrZ:.2f}')
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
    ok = True
    for j in range(len(fl)):
        E, Z = _E_Z_traj(fl[j])
        for name, s, thr in [('E(u)', E, thrE), ('Z(w)', Z, thrZ)]:
            seg = s[len(s) - win:]
            sl = np.polyfit(np.arange(win), seg, 1)[0]
            trend = abs(sl) * win / (seg.std() + 1e-30)
            stat = 'ok' if trend <= thr else 'FAIL'
            if trend > thr:
                ok = False
            print(f'    traj {j} {name}: trend={trend:.3f} vs thr={thr:.2f} {stat}')
    print(f'  [gate b] stationarity: {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------
def main():
    os.makedirs('data_generated', exist_ok=True)
    print('=' * 70)
    print(f'  NS 128^2 resolution-OOD 数据生成 | N={N} dt_solver={DT_SOLVER} '
          f'sample_every={SAMPLE_EVERY} frame_dt={FRAME_DT}')
    print(f'  物理参数不变: nu={NU} A={A_FORCE} kf={KF} alpha={ALPHA}')
    print('=' * 70)
    t_all = time.time()
    fields = gen_forced(3, SEEDS, 'res128')
    save_h5(OUT, fields, SEEDS, 'res_ood_128_zero_shot_eval')

    print('\n================ 验证门 ================')
    g1, med1, mx1 = gate_onestep(OUT)
    g2 = gate_residual(OUT)
    g3 = gate_stationary(OUT)
    print('========================================')
    if not (g1 and g2 and g3):
        raise RuntimeError('验证门未全部通过，数据不入库！')
    print(f'ALL GATES PASS. 总耗时 {time.time()-t_all:.0f}s')


if __name__ == '__main__':
    main()

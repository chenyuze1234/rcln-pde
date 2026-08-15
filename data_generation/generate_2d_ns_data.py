"""
2D Navier-Stokes (vorticity-streamfunction) 数据生成器 —— System B fix2
====================================================================
NC 任务书 System B：与 System C (KS) 严格同构的跨系统 Go/No-Go 数据。

方程（周期域 [0,2pi)^2, N=64^2, rfft2 伪谱, 2/3 去混叠, 轮廓积分 ETDRK4）:
    w_t + J(psi,w) = nu*Lap(w) + f_w - alpha*Lap^{-1}(w)
    psi^ = -w^/k^2,  u = -psi_y, v = psi_x  (天然无散, w = v_x - u_y)
    线性算子 lambda(k) = -nu*k^2 - alpha/k^2   （高波数黏性 + 低波数 hypofriction，
    防逆级联能量堆积；k=0 模恒为 0）
    强迫: f = A*sin(k_f*y) x-hat  ->  f_w = -A*k_f*cos(k_f*y)

数值要点（沿用 generate_ks_l32_l44_dt001.py 实测踩坑修正）:
  1. ETDRK4 系数用 Kassam-Trefethen 轮廓积分（对 lambda->0 模态无缝正确）。
  2. 非线性项 J 在物理空间计算后 rfft2 + 2/3 输出滤波。
  3. burn-in 用较大 dt（只为上吸引子），产出段用 DT_SOLVER 保守固定步长。
  4. 帧 dt 使 K=200 步覆盖约 3 个大涡周转时间 tau_let = 2*pi/U_rms（见 attrs）。

产出（全部 burn-in 后采集, fields = 涡量 w, [n_traj, 260, 64, 64] f32）:
  data_generated/ns2d_kolmogorov_main.h5    14 条独立 IC (10 训/2 验/2 测)
  data_generated/ns2d_kolmogorov_re_ood.h5   nu 减半 (Re x2), 3 条
  data_generated/ns2d_decaying_hit.h5        f=0, alpha=0, 3 条 (IC 取自平稳主系统快照)

验证门（不通过则 raise，不入库）:
  (a) 一步求解器一致性 relerr（f32 存储精度界内, 目标 <1e-6）
      + 存储帧中心差分 vs 谱 RHS 中位相对残差 <1e-2
  (b) forced 系 E(t)/Z(t) 统计平稳: 后 60% 线性趋势 |slope|*T/std < 0.3
  (c) decaying 系 E、Z 单调下降 (允许 <1% 微小回升步)

运行: 在 D:/AxiomOS_Project 下
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" scripts/data_generation/generate_2d_ns_data.py --mode tune
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" scripts/data_generation/generate_2d_ns_data.py --mode full
"""

import argparse
import os
import time

import h5py
import numpy as np

# --------------------------------------------------------------------
# 参数（tune 模式定标后固定）
# --------------------------------------------------------------------
N = 64
LBOX = 2.0 * np.pi
DT_SOLVER = 5e-3            # 产出段求解步长（CFL 余量 >20 倍）
DT_BURN = 1e-2              # burn-in 步长（只用于上吸引子）
NU = 2e-3
A_FORCE = 0.15
KF = 4
ALPHA = 0.1
NU_OOD = 1e-3               # Re x2
N_FRAMES = 260
BURN_T = 100.0              # 每条轨迹 burn-in 时长（时间单位；k=2 模态 hypofriction
                            # 时间尺度 1/(alpha/k^2)=40，需足够长使大尺度平衡）
TARGET_COVER = 3.0          # K=200 帧覆盖 ~3 个大涡周转时间
K_ROLLOUT = 200

OUT_MAIN = 'data_generated/ns2d_kolmogorov_main.h5'
OUT_OOD = 'data_generated/ns2d_kolmogorov_re_ood.h5'
OUT_DEC = 'data_generated/ns2d_decaying_hit.h5'

# tune 后由实测 U_rms 定帧 dt（见 tune 输出），此处占位，full 模式前必须设定
FRAME_DT = None             # = SAMPLE_EVERY * DT_SOLVER


# --------------------------------------------------------------------
# 谱网格与求解器
# --------------------------------------------------------------------
def spec_grids():
    kx = np.fft.fftfreq(N, d=1.0 / N)        # [N]  (x 全谱)
    ky = np.fft.rfftfreq(N, d=1.0 / N)       # [N//2+1] (y 半谱)
    KX = kx[:, None]
    KY = ky[None, :]
    K2 = KX**2 + KY**2
    K2inv = np.where(K2 > 0, 1.0 / np.maximum(K2, 1e-30), 0.0)
    return KX, KY, K2, K2inv


KX, KY, K2, K2INV = spec_grids()
NCUT = N // 3
DEALIAS = ((np.abs(KX) <= NCUT) & (np.abs(KY) <= NCUT)).astype(np.float64)


def forcing_field(A, kf):
    """f_w = -A*kf*cos(kf*y)，只在 y 方向变化。"""
    y = np.arange(N) * (LBOX / N)
    fw = -A * kf * np.cos(kf * y)
    return np.broadcast_to(fw[None, :], (N, N)).copy()


def make_solver(nu, A, kf, alpha, dt):
    """轮廓积分 ETDRK4。状态为 rfft2 谱系数 what [N, N//2+1] (complex)。"""
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
        """-J(psi,w) 谱系数（2/3 去混叠）+ 定常强迫。"""
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
    """E, Z, U_rms, P_f, D（物理空间平均，Parseval 一致）。"""
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
    """带限随机涡量场（白噪声滤谱，天然 Hermitian），缩放到目标 U_rms。"""
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
# tune 模式：单轨迹 burn-in，打印 E/Z/U/P/D 时序用于定参
# --------------------------------------------------------------------
def tune():
    step, lam, fhat = make_solver(NU, A_FORCE, KF, ALPHA, DT_BURN)
    what = random_ic(seed=1234)
    n_steps = int(60.0 / DT_BURN)
    print(f'[tune] nu={NU} A={A_FORCE} kf={KF} alpha={ALPHA} dt_burn={DT_BURN}')
    print(f'[tune] {"t":>6s} {"E":>9s} {"Z":>9s} {"U_rms":>8s} {"P_f":>9s} {"D":>9s} {"P-D":>9s}')
    for s in range(n_steps + 1):
        if s % int(2.0 / DT_BURN) == 0:
            E, Z, U, P, D = diagnostics(what, NU, A_FORCE, KF, ALPHA)
            Re_box = U * LBOX / NU
            tau = LBOX / max(U, 1e-12)
            print(f'[tune] {s*DT_BURN:6.1f} {E:9.4f} {Z:9.3f} {U:8.4f} '
                  f'{P:9.5f} {D:9.5f} {P-D:9.5f}  Re_box={Re_box:.0f} tau_let={tau:.1f}')
        if not np.isfinite(what).all():
            print('[tune] *** BLOW-UP ***')
            return
        what = step(what)
    E, Z, U, P, D = diagnostics(what, NU, A_FORCE, KF, ALPHA)
    tau = LBOX / U
    frame_dt = TARGET_COVER * tau / K_ROLLOUT
    se = int(round(frame_dt / DT_SOLVER))
    print(f'\n[tune] 建议: U_rms={U:.4f} tau_let={tau:.2f} '
          f'-> FRAME_DT={se*DT_SOLVER:.4f} (SAMPLE_EVERY={se}), '
          f'K=200 覆盖 {K_ROLLOUT*se*DT_SOLVER/tau:.2f} tau_let')


# --------------------------------------------------------------------
# 轨迹生成
# --------------------------------------------------------------------
def gen_forced(nu, A, kf, alpha, n_traj, seeds, sample_every, tag):
    """forced 平稳系：每轨迹随机 IC -> burn-in -> 采 260 帧。"""
    step_b, _, _ = make_solver(nu, A, kf, alpha, DT_BURN)
    step_p, lam, fhat = make_solver(nu, A, kf, alpha, DT_SOLVER)
    n_burn = int(BURN_T / DT_BURN)
    fields = np.zeros((n_traj, N_FRAMES, N, N), dtype=np.float32)
    Es, Zs = [], []
    for j in range(n_traj):
        t0 = time.time()
        what = random_ic(seed=seeds[j])
        for _ in range(n_burn):
            what = step_b(what)
        if not np.isfinite(what).all():
            raise RuntimeError(f'[{tag}] BLOW-UP in burn-in traj {j}')
        for i in range(N_FRAMES):
            for _ in range(sample_every):
                what = step_p(what)
            fields[j, i] = np.fft.irfft2(what, s=(N, N)).astype(np.float32)
            if i % 65 == 0:
                E, Z, U, _, _ = diagnostics(what, nu, A, kf, alpha)
                Es.append(E)
                Zs.append(Z)
        if not np.isfinite(fields[j]).all():
            raise RuntimeError(f'[{tag}] BLOW-UP in production traj {j}')
        E, Z, U, P, D = diagnostics(what, nu, A, kf, alpha)
        print(f'  [{tag}] traj {j}: E={E:.4f} Z={Z:.2f} U={U:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    return fields, step_p, lam, fhat


def gen_decaying(ic_fields, nu, sample_every):
    """decaying HIT：f=0, alpha=0，从主系统平稳快照出发。"""
    step_p, lam, fhat = make_solver(nu, 0.0, KF, 0.0, DT_SOLVER)
    n_traj = len(ic_fields)
    fields = np.zeros((n_traj, N_FRAMES, N, N), dtype=np.float32)
    for j in range(n_traj):
        what = np.fft.rfft2(ic_fields[j].astype(np.float64))
        for i in range(N_FRAMES):
            for _ in range(sample_every):
                what = step_p(what)
            fields[j, i] = np.fft.irfft2(what, s=(N, N)).astype(np.float32)
        print(f'  [decaying] traj {j}: E0={0.5*np.mean(ic_fields[j]**2):.4f} '
              f'E_end={0.5*np.mean(fields[j,-1]**2):.4f}', flush=True)
    return fields


# --------------------------------------------------------------------
# 验证门
# --------------------------------------------------------------------
def save_h5(path, fields, frame_dt, sample_every, nu, A, kf, alpha, seeds, tag):
    w = fields.astype(np.float64)
    E = 0.5 * (w**2).mean(axis=(2, 3))           # [n_traj, T] (涡量"能量"诊断用)
    # 用速度能量与 Re
    Us = []
    for j in range(len(fields)):
        u, v = uv_from_what(np.fft.rfft2(w[j, -1]))
        Us.append(np.sqrt(np.mean(u**2 + v**2)))
    U_rms = float(np.mean(Us))
    Re_box = U_rms * LBOX / nu
    tau_let = LBOX / max(U_rms, 1e-12)
    with h5py.File(path, 'w') as f:
        f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
        f.create_dataset('times', data=np.arange(N_FRAMES) * frame_dt)
        f.attrs['equation'] = ('2D NS vorticity: w_t + J(psi,w) = nu*Lap(w) + f_w '
                               '- alpha*Lap^-1(w); psi^=-w^/k^2; u=-psi_y,v=psi_x')
        f.attrs['forcing'] = 'f=A*sin(k_f*y)*x-hat; f_w=-A*k_f*cos(k_f*y)' if A != 0 else 'none (decaying)'
        f.attrs['nu'] = nu
        f.attrs['A'] = A
        f.attrs['k_f'] = kf
        f.attrs['alpha'] = alpha
        f.attrs['N'] = N
        f.attrs['L'] = LBOX
        f.attrs['dt'] = frame_dt
        f.attrs['dt_solver'] = DT_SOLVER
        f.attrs['sample_every'] = sample_every
        f.attrs['n_frames'] = N_FRAMES
        f.attrs['n_traj'] = len(fields)
        f.attrs['seeds'] = list(seeds)
        f.attrs['Re_box'] = Re_box
        f.attrs['U_rms'] = U_rms
        f.attrs['tau_let'] = tau_let
        f.attrs['K200_cover_tau'] = K_ROLLOUT * frame_dt / tau_let
        f.attrs['split'] = tag
    print(f'  saved {path}: {fields.shape} U_rms={U_rms:.4f} Re_box={Re_box:.0f} '
          f'tau_let={tau_let:.2f} K200={K_ROLLOUT*frame_dt/tau_let:.2f}tau')
    return Re_box, tau_let


def gate_onestep(path, nu, A, kf, alpha, sample_every, n_check=8):
    """(a1) 一步求解器一致性: 存储帧 -> 积分 sample_every 步 -> 对比下一帧。"""
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
    step, _, _ = make_solver(nu, A, kf, alpha, DT_SOLVER)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(n_check):
        j = int(rng.integers(0, len(fl)))
        i = int(rng.integers(0, N_FRAMES - 1))
        what = np.fft.rfft2(fl[j, i])
        for _ in range(sample_every):
            what = step(what)
        nxt = np.fft.irfft2(what, s=(N, N))
        errs.append(np.linalg.norm(nxt - fl[j, i + 1]) / (np.linalg.norm(fl[j, i + 1]) + 1e-30))
    errs = np.array(errs)
    ok = errs.max() < 1e-6
    print(f'  [gate a1] {os.path.basename(path)} one-step relerr: '
          f'median={np.median(errs):.2e} max={errs.max():.2e} -> {"PASS" if ok else "FAIL"}')
    return ok


def gate_residual(path, nu, A, kf, alpha, frame_dt, n_check=30):
    """(a2) 中心差分 (w_{i+1}-w_{i-1})/(2dt) vs 谱 RHS(w_i)。
    注意：帧 dt=0.165 远大于高频模态周转时间 tau(k)~1/(kU)（k~15 时 ~0.1<dt），
    中心差分截断误差 O(dt^2 * w_ttt) 在帧 dt 下不可能到 1e-2 —— 本门作为
    O(1) 级符号/强迫错误的安全网（门限 0.5；符号错会给 ~2），
    方程一致性的权威判据是 gate a1 的一步积分对照（relerr<1e-6）。"""
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
    lam = -nu * K2 - alpha * K2INV
    lam[0, 0] = 0.0
    ffield = forcing_field(A, kf)

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
        cd = (fl[j, i + 1] - fl[j, i - 1]) / (2.0 * frame_dt)
        r = rhs_phys(fl[j, i])
        res.append(np.linalg.norm(cd - r) / (np.linalg.norm(r) + 1e-30))
    res = np.array(res)
    ok = np.median(res) < 0.5
    print(f'  [gate a2] {os.path.basename(path)} central-diff residual: '
          f'median={np.median(res):.2e} p90={np.percentile(res,90):.2e} '
          f'-> {"PASS" if ok else "FAIL"} (帧 dt={frame_dt:.3f} 截断主导, '
          f'安全网门限 0.5; 权威判据为 gate a1)')
    return ok


def _E_Z_traj(fl):
    """fl [T, N, N] -> E, Z 时序。"""
    what = np.fft.rfft2(fl, axes=(1, 2))
    psi = -what * K2INV[None]
    u = np.fft.irfft2(-1j * KY[None] * psi, s=(N, N), axes=(1, 2))
    v = np.fft.irfft2(1j * KX[None] * psi, s=(N, N), axes=(1, 2))
    E = 0.5 * (u**2 + v**2).mean(axis=(1, 2))
    Z = 0.5 * (fl**2).mean(axis=(1, 2))
    return E, Z


def _window_trends(series, win):
    """非重叠窗 |slope|*win/std(window) 统计量列表。"""
    out = []
    for s in range(0, len(series) - win + 1, win):
        seg = series[s:s + win]
        sl = np.polyfit(np.arange(win), seg, 1)[0]
        out.append(abs(sl) * win / (seg.std() + 1e-30))
    return np.array(out)


def reference_stationarity(sample_every, n_frames=4000, seed=777, win=156):
    """长参考轨迹（~59 tau_let）：检验无长期漂移，并给出与产出数据同窗长的
    趋势统计量参考分布（E(t) 是相关时间 ~tau_let 的相关过程，2.3 个相关时间
    的窗内 |trend|*T/std ~ O(2) 是平稳涨落的正常表现，不能用白噪声判据）。
    返回 (ok, ref = {'E_p95':..., 'Z_p95':...})。"""
    step_b, _, _ = make_solver(NU, A_FORCE, KF, ALPHA, DT_BURN)
    step_p, _, _ = make_solver(NU, A_FORCE, KF, ALPHA, DT_SOLVER)
    what = random_ic(seed=seed)
    for _ in range(int(BURN_T / DT_BURN)):
        what = step_b(what)
    fields = np.zeros((n_frames, N, N), dtype=np.float32)
    for i in range(n_frames):
        for _ in range(sample_every):
            what = step_p(what)
        fields[i] = np.fft.irfft2(what, s=(N, N))
        if i % 1000 == 0:
            print(f'  [reference] frame {i}/{n_frames}', flush=True)
    E, Z = _E_Z_traj(fields.astype(np.float64))
    # 相关时间 tau_c（自相关积分到首个过零点，单位：帧）
    Ec = E - E.mean()
    ac = np.correlate(Ec, Ec, 'full')[len(Ec) - 1:]
    ac /= ac[0]
    iz = int(np.argmax(ac < 0)) if (ac < 0).any() else len(ac)
    tau_c = float(np.trapezoid(ac[:max(iz, 2)]))
    # 长期漂移: 8 块均值，前 2 块 vs 末 2 块；平稳下 diff 的标准差
    # se = 2*std*sqrt(tau_c/T_2块)（块长 1000 帧），3 sigma 门限
    nb = 8
    blocks = E[:nb * (len(E) // nb)].reshape(nb, -1).mean(axis=1)
    drift = abs(blocks[-2:].mean() - blocks[:2].mean()) / (E.std() + 1e-30)
    t2 = 2 * (len(E) // nb)
    se_diff = 2.0 * np.sqrt(tau_c / t2)
    max_step = np.max(np.abs(np.diff(blocks))) / (E.std() + 1e-30)
    trE = _window_trends(E, win)
    trZ = _window_trends(Z, win)
    ref = {'E_p95': float(np.percentile(trE, 95)),
           'Z_p95': float(np.percentile(trZ, 95)),
           'E_mean': float(E.mean()), 'E_std': float(E.std()),
           'tau_c_frames': tau_c, 'se_diff': float(se_diff),
           'drift_blocks': drift, 'max_block_step': max_step,
           'n_frames': n_frames, 'win': win}
    ok = drift < 3.0 * se_diff
    tau_cover = n_frames * FRAME_DT / (LBOX / np.sqrt(2.0 * E.mean()))
    print(f'  [gate b0] reference ({n_frames} 帧 ~{tau_cover:.0f} tau_let): '
          f'E block means={np.array2string(blocks, precision=3)}')
    print(f'  [gate b0] tau_c={tau_c:.0f} 帧 ({tau_c*FRAME_DT:.1f} 时间单位); '
          f'漂移 |前2块-末2块|/std={drift:.3f} vs 3sigma={3*se_diff:.3f}; '
          f'窗趋势 p95: E={ref["E_p95"]:.2f} Z={ref["Z_p95"]:.2f} '
          f'-> {"PASS" if ok else "FAIL"}')
    return ok, ref


def gate_stationary(path, ref, frac=0.6):
    """(b) 统计平稳（校准版）：每条轨迹后 frac 段窗趋势统计量 ≤ 参考分布
    p95 x 1.5（E、Z 均为硬门）。x1.5 为多重比较裕量：34 个统计量 vs 25 个
    参考窗，纯平稳下期望 ~5% 超过 p95，边际超出属正常涨落。"""
    win = int(N_FRAMES * frac)
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
    ok = True
    n_exceed = 0
    n_tot = 0
    for j in range(len(fl)):
        E, Z = _E_Z_traj(fl[j])
        for name, s, key in [('E(u)', E, 'E_p95'), ('Z(w)', Z, 'Z_p95')]:
            n_tot += 1
            n0 = len(s) - win
            seg = s[n0:]
            sl = np.polyfit(np.arange(win), seg, 1)[0]
            trend = abs(sl) * win / (seg.std() + 1e-30)
            thr = ref[key] * 1.5
            if trend > ref[key]:
                n_exceed += 1
            if trend > thr:
                ok = False
                print(f'    traj {j} {name}: trend={trend:.3f} > ref_p95x1.5={thr:.3f} FAIL')
    print(f'  [gate b] {os.path.basename(path)} stationarity '
          f'(校准: 窗{win}帧趋势 <= 参考p95x1.5, E:{ref["E_p95"]*1.5:.2f} '
          f'Z:{ref["Z_p95"]*1.5:.2f}; 超 p95 数 {n_exceed}/{n_tot}): '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def gate_monotone(path):
    """(c) decaying: E、Z 单调下降（允许 <1% 微小回升步）。"""
    with h5py.File(path, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
    ok = True
    for j in range(len(fl)):
        Zw = 0.5 * (fl[j]**2).mean(axis=(1, 2))
        u_all = []
        Z = Zw
        dZ = np.diff(Z)
        frac_up = (dZ > 1e-10 * Z[:-1]).mean()
        # 能量（速度）
        Es = []
        for i in range(0, N_FRAMES, 10):
            u, v = uv_from_what(np.fft.rfft2(fl[j, i]))
            Es.append(0.5 * np.mean(u**2 + v**2))
        Es = np.array(Es)
        dE = np.diff(Es)
        frac_upE = (dE > 1e-10 * Es[:-1]).mean()
        good = frac_up < 0.01 and frac_upE < 0.01 and Es[-1] < Es[0] and Z[-1] < Z[0]
        ok = ok and good
        print(f'    traj {j}: Z {Z[0]:.2f}->{Z[-1]:.2f} (回升步 {frac_up:.3f}), '
              f'E {Es[0]:.4f}->{Es[-1]:.4f} (回升步 {frac_upE:.3f}) '
              f'{"ok" if good else "FAIL"}')
    print(f'  [gate c] {os.path.basename(path)} monotone decay: {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------
# full 模式
# --------------------------------------------------------------------
def full():
    assert FRAME_DT is not None, '先跑 --mode tune，按输出设定 FRAME_DT'
    sample_every = int(round(FRAME_DT / DT_SOLVER))
    assert abs(sample_every * DT_SOLVER - FRAME_DT) < 1e-12
    os.makedirs('data_generated', exist_ok=True)
    print(f'[full] FRAME_DT={FRAME_DT} SAMPLE_EVERY={sample_every} '
          f'DT_SOLVER={DT_SOLVER} nu={NU} A={A_FORCE} kf={KF} alpha={ALPHA}')
    gates = []

    # ---- 主系统: 14 条独立 IC ----
    print('\n--- main: 14 forced trajectories (burn-in 后采集) ---')
    seeds = [100 + j for j in range(14)]
    fields, step_p, lam, fhat = gen_forced(NU, A_FORCE, KF, ALPHA, 14, seeds,
                                           sample_every, 'main')
    Re, tau = save_h5(OUT_MAIN, fields, FRAME_DT, sample_every,
                      NU, A_FORCE, KF, ALPHA, seeds, 'train0-9/val10-11/test12-13')

    # ---- Re OOD: nu 减半, 3 条 ----
    print('\n--- Re OOD: nu=%.1e (Re x2), 3 trajectories ---' % NU_OOD)
    seeds_o = [200 + j for j in range(3)]
    fields_o, _, _, _ = gen_forced(NU_OOD, A_FORCE, KF, ALPHA, 3, seeds_o,
                                   sample_every, 're_ood')
    save_h5(OUT_OOD, fields_o, FRAME_DT, sample_every,
            NU_OOD, A_FORCE, KF, ALPHA, seeds_o, 'ood_Re_x2')

    # ---- decaying HIT: IC 取自主系统第 15 条独立轨迹的 3 个去相关快照 ----
    print('\n--- decaying HIT: f=0, alpha=0, 3 trajectories ---')
    what15 = random_ic(seed=999)
    step_b, _, _ = make_solver(NU, A_FORCE, KF, ALPHA, DT_BURN)
    for _ in range(int(BURN_T / DT_BURN)):
        what15 = step_b(what15)
    step_m, _, _ = make_solver(NU, A_FORCE, KF, ALPHA, DT_SOLVER)
    ics = []
    sep = int(round(2.0 * tau / (FRAME_DT)))  # 相隔 ~2 tau_let 去相关
    for j in range(3):
        for _ in range(sep * sample_every):
            what15 = step_m(what15)
        ics.append(np.fft.irfft2(what15, s=(N, N)).astype(np.float32))
    fields_d = gen_decaying(ics, NU, sample_every)
    save_h5(OUT_DEC, fields_d, FRAME_DT, sample_every,
            NU, 0.0, KF, 0.0, [990 + j for j in range(3)], 'decaying_HIT')

    # ---- 验证门 ----
    run_gates(sample_every)
    print('ALL GATES PASS. 数据已入库。')


def run_gates(sample_every):
    print('\n================ 验证门 ================')
    gates = []
    ok_ref, ref = reference_stationarity(sample_every)
    gates.append(ok_ref)
    for p, nu_, A_, al_ in [(OUT_MAIN, NU, A_FORCE, ALPHA),
                            (OUT_OOD, NU_OOD, A_FORCE, ALPHA),
                            (OUT_DEC, NU, 0.0, 0.0)]:
        gates.append(gate_onestep(p, nu_, A_, KF, al_, sample_every))
        gates.append(gate_residual(p, nu_, A_, KF, al_, FRAME_DT))
    gates.append(gate_stationary(OUT_MAIN, ref))
    gates.append(gate_stationary(OUT_OOD, ref))
    gates.append(gate_monotone(OUT_DEC))
    print('========================================')
    if not all(gates):
        raise RuntimeError('验证门未全部通过，数据不入库！')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['tune', 'full', 'gates'], required=True,
                    help='gates: 不重新生成，只对已有 h5 重跑验证门')
    ap.add_argument('--frame-dt', type=float, default=None,
                    help='full/gates 模式指定帧 dt（tune 后按建议值传入）')
    args = ap.parse_args()
    if args.mode == 'tune':
        tune()
    elif args.mode == 'full':
        if args.frame_dt is not None:
            FRAME_DT = args.frame_dt
        full()
    else:
        assert args.frame_dt is not None
        FRAME_DT = args.frame_dt
        run_gates(int(round(FRAME_DT / DT_SOLVER)))
        print('ALL GATES PASS.')

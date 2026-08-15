"""
KS L=32 / L=44, N=64, frame dt=0.01 —— 标准 KS 重生成版（v2，轮廓积分 ETDRK4）
================================================================
背景（2026-08，system_c_fix2 线调查结论）：
  现有 data_generated/ks_L32_N64_T30.0.h5 / ks_L44_N64_T40.0.h5 经逐帧
  谱残差验证，满足的其实是【阻尼变体】u_t = -u*u_x + u_xx - u_xxxx
  （relres≈0.002），而不是生成器脚本定义的标准 KS（relres≈1.9）；
  能量全程单调衰减，无混沌。旧 L22 文件（dt=0.05）整文件 NaN。
  三文件彼此方程都不一致，旧线结论不可用。
本脚本生成标准 KS: u_t = -u*u_x - u_xx - u_xxxx，lambda(k)=k^2-k^4。
求解器要点（均为实测踩坑后的修正）：
  1. ETDRK4 系数用 Kassam-Trefethen 轮廓积分法计算 —— 直接公式里的
     1e-12 保护项会吞掉 L=44 的 n=7 近零 lambda 模态（kappa=0.9995,
     lambda≈+0.001），非线性耦合被错误压制约 4000 倍，能量堆积爆破。
  2. 非线性项 2/3 规则去混叠（输出滤波）。
  3. 求解器 dt=0.0005（dt=0.001 在 L=44 上 t≈10.4 处积分不稳定，已实测
     与 N 无关）；每 20 步采样 -> 帧 dt=0.01。
  4. warmup T=50 上混沌吸引子，产出段统计平稳。
产出: ks_L32_N64_T30.0_dt0.01.h5 (3000 帧), ks_L44_N64_T40.0_dt0.01.h5 (4000 帧)
运行: 在 D:/AxiomOS_Project 下
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" scripts/data_generation/generate_ks_l32_l44_dt001.py
"""

import numpy as np, h5py, os

DT_SOLVER = 0.0005
SAMPLE_EVERY = 20          # 帧 dt = 0.01
WARMUP = 100000            # T_warm = 50
N = 64

def make_etdrk4(L):
    """Kassam-Trefethen 轮廓积分 ETDRK4（对 lambda->0 模态无缝正确）。"""
    dt = DT_SOLVER
    k = 2j * np.pi * np.fft.fftfreq(N, d=L / N)
    Lop = -k**2 - k**4                       # linear: -u_xx - u_xxxx
    mask = np.ones(N)
    nc = N // 3
    mask[nc + 1: N - nc] = 0.0               # 2/3 去混叠

    R = 32
    r = np.exp(1j * np.pi * (np.arange(1, R + 1) - 0.5) / R)
    LR = dt * Lop[:, None] + r[None, :]
    E  = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)
    Q  = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0*LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0*LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))

    def g_hat(u):
        """非线性项 -u*u_x 的 Fourier 系数（2/3 去混叠）。"""
        uh = np.fft.fft(u)
        u_x = np.fft.ifft(k * uh).real
        return np.fft.fft(-u * u_x) * mask

    def step(u):
        v = np.fft.fft(u)
        Nv = g_hat(u)
        a = E2 * v + Q * Nv
        Na = g_hat(np.fft.ifft(a).real)
        b = E2 * v + Q * Na
        Nb = g_hat(np.fft.ifft(b).real)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g_hat(np.fft.ifft(c).real)
        return np.fft.ifft(E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3).real
    return step

def generate(L, T, out_path, seed):
    step = make_etdrk4(L)
    steps = int(T / DT_SOLVER)
    np.random.seed(seed)
    x = np.linspace(0, L, N, endpoint=False)
    u = 0.5*(np.cos(2*np.pi*x/L + 0.3) + np.cos(4*np.pi*x/L - 0.5) +
             0.3*np.sin(6*np.pi*x/L + 0.7) + 0.2*np.cos(8*np.pi*x/L))
    print(f'KS L={L} N={N} steps={steps} warmup={WARMUP} -> {out_path}')
    for s in range(WARMUP):
        u = step(u)
        if s % 20000 == 0:
            print(f'  warmup {s}: umax={np.abs(u).max():.3f} rms={np.std(u):.4f}')
        if not np.isfinite(u).all():
            raise RuntimeError(f'BLOW-UP in warmup at step {s}')
    fields_raw = []
    for s in range(steps):
        u = step(u)
        if s % SAMPLE_EVERY == 0:
            fields_raw.append(u.astype(np.float32))
        if s % 20000 == 0:
            print(f'  step {s}: umax={np.abs(u).max():.3f} rms={np.std(u):.4f}')
        if not np.isfinite(u).all():
            raise RuntimeError(f'BLOW-UP at step {s}')
    fields = np.stack(fields_raw, axis=0)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
        f.create_dataset('times', data=np.arange(len(fields)) * DT_SOLVER * SAMPLE_EVERY)
        f.attrs['equation'] = 'Kuramoto-Sivashinsky'
        f.attrs['L'] = L
        f.attrs['N'] = N
        f.attrs['dt'] = DT_SOLVER * SAMPLE_EVERY
    E = 0.5 * (fields**2).mean(axis=1)
    print(f'  saved {fields.shape}  E mean={E.mean():.4f} var={E.var():.4f} '
          f'E(first500)={E[:500].mean():.4f} E(last500)={E[-500:].mean():.4f}')

os.makedirs('data_generated', exist_ok=True)
generate(32.0, 30.0, 'data_generated/ks_L32_N64_T30.0_dt0.01.h5', seed=42)
generate(44.0, 40.0, 'data_generated/ks_L44_N64_T40.0_dt0.01.h5', seed=43)

# ---- sanity: 一步积分一致性（取帧 u_t，求解器推 20 步，对比 u_{t+1}）----
print('\n--- sanity: one-step solver consistency (relerr 应 ~1e-3 以下) ---')
for p, L in [('data_generated/ks_L32_N64_T30.0_dt0.01.h5', 32.0),
             ('data_generated/ks_L44_N64_T40.0_dt0.01.h5', 44.0)]:
    with h5py.File(p, 'r') as f:
        fl = f['fields'][:].astype(np.float64)
        print(p, 'attrs:', dict(f.attrs))
    step = make_etdrk4(L)
    errs = []
    for i in range(500, len(fl) - 1, 251):
        u = fl[i].copy()
        for _ in range(SAMPLE_EVERY):
            u = step(u)
        errs.append(np.linalg.norm(u - fl[i + 1]) / np.linalg.norm(fl[i + 1]))
    print(f'  one-step relerr: mean={np.mean(errs):.2e} max={np.max(errs):.2e}')

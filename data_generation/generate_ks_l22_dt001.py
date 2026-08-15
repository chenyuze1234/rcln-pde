"""
KS L=22, N=64, T=30, frame dt=0.01 (3000 frames) —— 重生成版（v2，轮廓积分 ETDRK4）
================================================================
旧文件 data_generated/ks_L22_N64_T30.0.h5 整文件 NaN 且 dt=0.05，不可用
（旧线 length-OOD 全崩的根因之一）。
本脚本与 generate_ks_l32_l44_dt001.py 完全同口径生成标准 KS：
  u_t = -u*u_x - u_xx - u_xxxx，lambda(k)=k^2-k^4，k=2*pi*n/L。
  ETDRK4 系数用 Kassam-Trefethen 轮廓积分法（对近零 lambda 模态正确；
  L=44 的教训：直接公式的 1e-12 保护项会吞掉 lambda≈0.001 的模态）。
  求解器 dt=0.0005，每 20 步采样 -> 帧 dt=0.01，T=30 -> 3000 帧。
  warmup T=50 上混沌吸引子（短 warmup 时产出段前半是能量爬升暂态，不可用）。
运行: 在 D:/AxiomOS_Project 下
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" scripts/data_generation/generate_ks_l22_dt001.py
"""

import numpy as np, h5py, os

L, N, dt, T = 22.0, 64, 0.0005, 30.0
steps, warmup = int(T / dt), 100000   # warmup T=50
sample_every = 20                     # 帧 dt = 0.0005 * 20 = 0.01

k = 2j * np.pi * np.fft.fftfreq(N, d=L / N)
Lop = -k**2 - k**4  # linear: -u_xx - u_xxxx
mask = np.ones(N)
nc = N // 3
mask[nc + 1: N - nc] = 0.0            # 2/3 去混叠

# Kassam-Trefethen 轮廓积分 ETDRK4 系数
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

def etdrk4_step(u):
    v = np.fft.fft(u)
    Nv = g_hat(u)
    a = E2 * v + Q * Nv
    Na = g_hat(np.fft.ifft(a).real)
    b = E2 * v + Q * Na
    Nb = g_hat(np.fft.ifft(b).real)
    c = E2 * a + Q * (2.0 * Nb - Nv)
    Nc = g_hat(np.fft.ifft(c).real)
    return np.fft.ifft(E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3).real

np.random.seed(42)
x = np.linspace(0, L, N, endpoint=False)
u = 0.5*(np.cos(2*np.pi*x/L + 0.3) + np.cos(4*np.pi*x/L - 0.5) +
         0.3*np.sin(6*np.pi*x/L + 0.7) + 0.2*np.cos(8*np.pi*x/L))

print(f'KS L={L} N={N} solver_dt={dt} steps={steps} warmup={warmup} '
      f'frame_dt={dt*sample_every} frames={steps//sample_every}')

# Warmup
for s in range(warmup):
    u = etdrk4_step(u)
    if s % 20000 == 0:
        print(f'  warmup {s}: umax={np.abs(u).max():.3f} rms={np.std(u):.4f}')
    if not np.isfinite(u).all():
        raise RuntimeError(f'BLOW-UP in warmup at step {s}')

# Production
fields_raw = []
for s in range(steps):
    u = etdrk4_step(u)
    if s % sample_every == 0:
        fields_raw.append(u.astype(np.float32))
    if s % 20000 == 0:
        print(f'  step {s}: umax={np.abs(u).max():.3f} rms={np.std(u):.4f}')
    if not np.isfinite(u).all():
        raise RuntimeError(f'BLOW-UP at step {s}')

fields = np.stack(fields_raw, axis=0)  # [3000, 64]
print(f'\n{fields.shape} {fields.nbytes/1e6:.1f}MB '
      f'[{fields.min():.3f},{fields.max():.3f}] RMS={np.std(fields):.4f}')

os.makedirs('data_generated', exist_ok=True)
p = 'data_generated/ks_L22_N64_T30.0_dt0.01.h5'
with h5py.File(p, 'w') as f:
    f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
    f.create_dataset('times', data=np.arange(len(fields)) * dt * sample_every)
    f.attrs['equation'] = 'Kuramoto-Sivashinsky'
    f.attrs['L'] = L
    f.attrs['N'] = N
    f.attrs['dt'] = dt * sample_every
print(f'Saved: {p}')

# ---- sanity: attrs / shape / E(t) / 一步积分一致性 ----
with h5py.File(p, 'r') as f:
    print('\n--- sanity ---')
    print('attrs:', dict(f.attrs))
    print('fields:', f['fields'].shape, f['fields'].dtype)
    fl = f['fields'][:].astype(np.float64)
Et = 0.5 * (fl**2).mean(axis=1)
print(f'E(t): mean={Et.mean():.4f} var={Et.var():.4f} '
      f'E(first500)={Et[:500].mean():.4f} E(last500)={Et[-500:].mean():.4f}')

errs = []
for i in range(500, len(fl) - 1, 251):
    uu = fl[i].copy()
    for _ in range(sample_every):
        uu = etdrk4_step(uu)
    errs.append(np.linalg.norm(uu - fl[i + 1]) / np.linalg.norm(fl[i + 1]))
print(f'one-step solver consistency relerr: mean={np.mean(errs):.2e} '
      f'max={np.max(errs):.2e}  (应 ~1e-3 以下)')

with h5py.File('data_generated/ks_L32_N64_T30.0_dt0.01.h5', 'r') as f:
    fl32 = f['fields'][:]
E32 = 0.5 * (fl32**2).mean(axis=1)
print(f'L32 ref E(t): mean={E32.mean():.4f} var={E32.var():.4f}（同量级即合理）')

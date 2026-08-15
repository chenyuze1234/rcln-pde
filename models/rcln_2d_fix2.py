"""
RCLN-2D fix2 —— 2D Navier-Stokes (vorticity) System B 跨系统 Go/No-Go 模型库
====================================================================
与 models/rcln_1d_fix2.py (KS / System C) 严格同构的 2D 版本。
方程: w_t + J(psi,w) = nu*Lap(w) + f_w - alpha*Lap^-1(w),
      psi^ = -w^/k^2, u=-psi_y, v=psi_x, lambda(k) = -nu*k^2 - alpha/k^2。
数据来自 scripts/data_generation/generate_2d_ns_data.py（轮廓积分 ETDRK4，
一步一致性 relerr ~1e-7，验证门见生成器日志）。

能量恒等式（周期边界）: dE/dt = P_f - D,
  E = <|u|^2>/2, P_f = <u·f>（已知强迫解析算）,
  D = 2*nu*Z + alpha*<psi^2>,  Z = <w^2>/2。
w 零均值守恒（k=0 模恒 0），线性传播子与仿射归一化对易，hard core
可直接作用于归一化空间（强迫项 f^/std 按当前归一化缩放）。

PhysicsSpec —— 任务书"同架构原则、不同 PhysicsSpec"展示点:
  主系统 (nu, A, kf, alpha) / Re-OOD (nu') / decaying (A=0, alpha=0)
  共用同一 anchor 架构，仅切换零参数物理件（从 h5 attrs 读入，不硬编码）。

组件:
  SpectralConv2d   —— 手写 rfft2 截断谱卷积（双角块，不依赖 neuralop）
  FNO2d            —— modes=12, width=32, 4 层，3 输入通道 (w,u,v)（三臂共用）
  LinearExactNS2D  —— 零参数 hard core: w^_{t+1} = e^{lam*dt} w^_t + phi1(lam*dt)·f^/std
  RCLN2DFix2       —— anchor 臂: pred = LinearExact(w) + FNO2d(w,u,v)，端到端
  NSPostproc2D     —— 推理路径零参数物理件: 均值投影 + 双边能量包络
  NSRHS2D          —— 谱精度 NS 右端项（2/3 去混叠），供 loss_level 臂物理残差
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------
# PhysicsSpec
# --------------------------------------------------------------------
class PhysicsSpec:
    def __init__(self, N=64, nu=2e-3, A=0.15, kf=4, alpha=0.1, dt=0.165):
        self.N = int(N)
        self.nu = float(nu)
        self.A = float(A)
        self.kf = int(kf)
        self.alpha = float(alpha)
        self.dt = float(dt)

    @classmethod
    def from_h5_attrs(cls, attrs, override=None):
        kw = dict(N=int(attrs['N']), nu=float(attrs['nu']), A=float(attrs['A']),
                  kf=int(attrs['k_f']), alpha=float(attrs['alpha']),
                  dt=float(attrs['dt']))
        if override:
            kw.update(override)
        return cls(**kw)

    def __repr__(self):
        return (f'PhysicsSpec(N={self.N}, nu={self.nu}, A={self.A}, '
                f'kf={self.kf}, alpha={self.alpha}, dt={self.dt})')


# --------------------------------------------------------------------
# 谱工具（numpy, 供标定/核验）
# --------------------------------------------------------------------
def ns_wavenumbers(N, device=None, dtype=torch.float32):
    """rfft2 布局: KX [N,1], KY [1,N//2+1]（单位域 2pi，波数为整数）。"""
    kx = np.fft.fftfreq(N, d=1.0 / N).astype(np.float64)
    ky = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float64)
    KX = torch.tensor(kx[:, None], dtype=dtype, device=device)
    KY = torch.tensor(ky[None, :], dtype=dtype, device=device)
    return KX, KY


def ns_lambda(spec):
    """lambda(k) = -nu*k^2 - alpha/k^2（实数, [N, N//2+1]），k=0 处为 0。"""
    kx = np.fft.fftfreq(spec.N, d=1.0 / spec.N)
    ky = np.fft.rfftfreq(spec.N, d=1.0 / spec.N)
    K2 = kx[:, None]**2 + ky[None, :]**2
    K2inv = np.where(K2 > 0, 1.0 / np.maximum(K2, 1e-30), 0.0)
    lam = -spec.nu * K2 - spec.alpha * K2inv
    lam[0, 0] = 0.0
    return lam, K2inv


def ns_forcing_field(spec):
    """f_w = -A*kf*cos(kf*y), [N, N]。"""
    y = np.arange(spec.N) * (2.0 * np.pi / spec.N)
    fw = -spec.A * spec.kf * np.cos(spec.kf * y)
    return np.broadcast_to(fw[None, :], (spec.N, spec.N)).copy()


# --------------------------------------------------------------------
# FNO-2D（手写谱卷积）
# --------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    """双角块 rfft2 谱卷积（正/负 x 频率各 modes1 个）。"""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        # x: [B, C, Nx, Ny]
        B, C, Nx, Ny = x.shape
        xf = torch.fft.rfft2(x, norm='ortho')
        of = torch.zeros(B, self.w1.shape[1], Nx, Ny // 2 + 1,
                         dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, Nx // 2)
        m2 = min(self.modes2, Ny // 2 + 1)
        of[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy',
                                          xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        of[:, :, -m1:, :m2] = torch.einsum('bixy,ioxy->boxy',
                                           xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(of, s=(Nx, Ny), norm='ortho')


class FNO2d(nn.Module):
    """modes=12, width=32, 4 层。输入 [B, in_ch, N, N]，输出 [B, 1, N, N]。"""

    def __init__(self, in_ch=3, modes=12, width=32, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.skip = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, x):
        h = self.lift(x)
        for sp, sk in zip(self.spec, self.skip):
            h = F.gelu(sp(h) + sk(h))
        return self.proj(h)


class UVChannels(nn.Module):
    """由 w（任意仿射尺度，k=0 模为 0）谱重建 u,v，拼成 3 通道输入。
    三臂共用，保证参数量对齐。"""

    def __init__(self, N=64):
        super().__init__()
        self.N = N
        KX, KY = ns_wavenumbers(N)
        K2 = (KX**2 + KY**2).numpy()
        K2inv = np.where(K2 > 0, 1.0 / np.maximum(K2, 1e-30), 0.0)
        self.register_buffer('KX', KX)
        self.register_buffer('KY', KY)
        self.register_buffer('K2inv', torch.tensor(K2inv, dtype=torch.float32))

    def uv(self, w):
        # w: [B, 1, N, N]
        what = torch.fft.rfft2(w.float(), norm='ortho')
        psi = -what * self.K2inv
        u = torch.fft.irfft2(-1j * self.KY * psi, s=(self.N, self.N), norm='ortho')
        v = torch.fft.irfft2(1j * self.KX * psi, s=(self.N, self.N), norm='ortho')
        return u, v

    def forward(self, w):
        u, v = self.uv(w)
        return torch.cat([w, u, v], dim=1)


class FNO2dUV(nn.Module):
    """loss_level / none 臂: UVChannels + FNO2d(w,u,v -> w)。"""

    def __init__(self, N=64, modes=12, width=32, layers=4):
        super().__init__()
        self.uvc = UVChannels(N)
        self.fno = FNO2d(in_ch=3, modes=modes, width=width, layers=layers)

    def forward(self, w):
        return self.fno(self.uvc(w))


# --------------------------------------------------------------------
# 零参数线性精确传播子 + 定常强迫精确积分（hard core）
# --------------------------------------------------------------------
class LinearExactNS2D(nn.Module):
    """w^_{t+1} = e^{lam*dt}·w^_t + phi1(lam*dt)·f^/std。
    phi1(z)=(e^z-1)/z，lam->0 处退化为 dt（对定常力的精确积分）。
    lam(0)=0 且 w/f 均无 k=0 模 -> 与仿射归一化对易（强迫项按 1/std 缩放）。"""

    def __init__(self, spec, mean=0.0, std=1.0):
        super().__init__()
        self.N = spec.N
        self.set_spec(spec, mean, std)

    def set_spec(self, spec, mean=None, std=None):
        self.spec = spec
        if mean is not None:
            self.mean = float(mean)
        if std is not None:
            self.std = float(std)
        lam, _ = ns_lambda(spec)                       # float64 [N, N//2+1]
        prop = np.exp(lam * spec.dt)
        # 与 torch norm='ortho' 对齐：ortho 正变换 = numpy 默认正变换 / N
        fhat = np.fft.rfft2(ns_forcing_field(spec)) / float(spec.N)
        z = lam * spec.dt
        phi1_dt = np.where(np.abs(lam) > 1e-14,
                           np.expm1(z) / np.where(np.abs(lam) > 1e-14, lam, 1.0),
                           spec.dt)
        fterm = phi1_dt * fhat / self.std              # 归一化空间
        dev = self.prop.device if hasattr(self, 'prop') else 'cpu'
        self.register_buffer('prop', torch.tensor(prop, dtype=torch.float32).to(dev))
        self.register_buffer('fterm', torch.tensor(fterm, dtype=torch.cfloat).to(dev))

    def forward(self, w):
        # w: [B, 1, N, N]（归一化空间）
        what = torch.fft.rfft2(w.float(), norm='ortho')
        out = what * self.prop + self.fterm
        return torch.fft.irfft2(out, s=(self.N, self.N), norm='ortho')


# --------------------------------------------------------------------
# anchor 臂: LinearExact + Soft(FNO2d)，端到端训练
# --------------------------------------------------------------------
class RCLN2DFix2(nn.Module):
    def __init__(self, spec, mean=0.0, std=1.0, modes=12, width=32, layers=4):
        super().__init__()
        self.hard = LinearExactNS2D(spec, mean, std)
        self.uvc = UVChannels(spec.N)
        self.soft = FNO2d(in_ch=3, modes=modes, width=width, layers=layers)

    def set_spec(self, spec, mean=None, std=None):
        self.hard.set_spec(spec, mean, std)

    def forward(self, w):
        return self.hard(w) + self.soft(self.uvc(w))


# --------------------------------------------------------------------
# 谱精度 NS 右端项（loss_level 物理残差；输入物理空间 w）
# --------------------------------------------------------------------
class NSRHS2D(nn.Module):
    """w_t = -J(psi,w) + lam·w + f_w（2/3 去混叠，与生成器同口径）。可微。"""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        N = spec.N
        self.N = N
        KX, KY = ns_wavenumbers(N)
        K2 = (KX**2 + KY**2).numpy()
        K2inv = np.where(K2 > 0, 1.0 / np.maximum(K2, 1e-30), 0.0)
        lam, _ = ns_lambda(spec)
        ncut = N // 3
        mask = ((np.abs(KX.numpy()) <= ncut) & (np.abs(KY.numpy()) <= ncut))
        self.register_buffer('KX', KX)
        self.register_buffer('KY', KY)
        self.register_buffer('K2inv', torch.tensor(K2inv, dtype=torch.float32))
        self.register_buffer('lam', torch.tensor(lam, dtype=torch.float32))
        self.register_buffer('mask', torch.tensor(mask.astype(np.float32)))
        self.register_buffer('ffield',
                             torch.tensor(ns_forcing_field(spec), dtype=torch.float32))

    def forward(self, w):
        # w: [B, 1, N, N] 物理空间 -> w_t 物理空间
        what = torch.fft.rfft2(w, norm='ortho')
        psi = -what * self.K2inv
        u = torch.fft.irfft2(-1j * self.KY * psi, s=(self.N, self.N), norm='ortho')
        v = torch.fft.irfft2(1j * self.KX * psi, s=(self.N, self.N), norm='ortho')
        wx = torch.fft.irfft2(1j * self.KX * what, s=(self.N, self.N), norm='ortho')
        wy = torch.fft.irfft2(1j * self.KY * what, s=(self.N, self.N), norm='ortho')
        Jh = torch.fft.rfft2(u * wx + v * wy, norm='ortho') * self.mask
        return torch.fft.irfft2(-Jh + self.lam * what,
                                s=(self.N, self.N), norm='ortho') + self.ffield


# --------------------------------------------------------------------
# 推理路径零参数物理件: 均值投影 + 双边能量包络
# --------------------------------------------------------------------
class NSPostproc2D(nn.Module):
    """包在 anchor 模型外的推理路径 wrapper（训练不用）。

    (a) 均值投影: pred <- pred - (mean(pred) - mean(u_in))（w 零均值守恒）。
    (b) 双边能量包络（物理空间，速度能量 E=<|u|^2>/2）:
        E_{t+1} ∈ [E_t - c_d·D(u_t)·dt,  E_t + c_p·P_f(u_t)·dt]
        D = 2·nu·Z + alpha·<psi^2>,  P_f = <u·f>（解析强迫）。
        越界则均匀缩放 alpha_env = sqrt(E_bound/E_pred)。
    c_d, c_p 由训练真值标定（脚本内计算）。命中率/校正量全程记录。
    Z/Z0 不强行包络，仅作诊断（2D 卖点）。
    """

    def __init__(self, inner, spec, mean=0.0, std=1.0, c_d=1.0, c_p=1.0,
                 mean_project=True, envelope=True):
        super().__init__()
        self.inner = inner
        self.N = spec.N
        self.c_d = float(c_d)
        self.c_p = float(c_p)
        self.mean_project = bool(mean_project)
        self.envelope = bool(envelope)
        self.mean = float(mean)
        self.std = float(std)
        KX, KY = ns_wavenumbers(spec.N)
        K2 = (KX**2 + KY**2).numpy()
        K2inv = np.where(K2 > 0, 1.0 / np.maximum(K2, 1e-30), 0.0)
        self.register_buffer('KX', KX)
        self.register_buffer('KY', KY)
        self.register_buffer('K2inv', torch.tensor(K2inv, dtype=torch.float32))
        self.set_spec(spec, mean, std)
        self.reset_stats()

    def set_spec(self, spec, mean=None, std=None):
        self.spec = spec
        if mean is not None:
            self.mean = float(mean)
        if std is not None:
            self.std = float(std)
        fx = np.zeros((self.N, self.N), dtype=np.float32)
        if spec.A != 0.0:
            y = np.arange(self.N) * (2.0 * np.pi / self.N)
            fx = np.broadcast_to((spec.A * np.sin(spec.kf * y))[None, :],
                                 (self.N, self.N)).astype(np.float32).copy()
        dev = self.fx.device if hasattr(self, 'fx') else 'cpu'
        self.register_buffer('fx', torch.tensor(fx).to(dev))
        if hasattr(self.inner, 'set_spec'):
            self.inner.set_spec(spec, self.mean, self.std)

    def set_norm(self, mean, std):
        self.set_spec(self.spec, mean, std)

    def reset_stats(self):
        self._n = 0
        self._n_floor = 0
        self._n_ceil = 0
        self._sum_abs_corr = 0.0

    def env_stats(self):
        return {
            'n_steps': self._n,
            'floor_hit_rate': self._n_floor / max(1, self._n),
            'ceil_hit_rate': self._n_ceil / max(1, self._n),
            'mean_abs_corr': self._sum_abs_corr / max(1, self._n),
            'c_d': self.c_d, 'c_p': self.c_p,
        }

    def _energy_rates(self, w_phys):
        """E, D, P_f。w_phys: [B,1,N,N] -> [B],[B],[B]（谱精度）。"""
        what = torch.fft.rfft2(w_phys.float(), norm='ortho')
        psi = -what * self.K2inv
        u = torch.fft.irfft2(-1j * self.KY * psi, s=(self.N, self.N), norm='ortho')
        v = torch.fft.irfft2(1j * self.KX * psi, s=(self.N, self.N), norm='ortho')
        psi_p = torch.fft.irfft2(psi, s=(self.N, self.N), norm='ortho')
        E = 0.5 * (u**2 + v**2).mean(dim=(1, 2, 3))
        Z = 0.5 * (w_phys.float()**2).mean(dim=(1, 2, 3))
        D = 2.0 * self.spec.nu * Z + self.spec.alpha * (psi_p**2).mean(dim=(1, 2, 3))
        P_f = (u * self.fx).mean(dim=(1, 2, 3))
        return E, D, P_f

    def forward(self, u_norm):
        pred = self.inner(u_norm)
        if not (self.mean_project or self.envelope):
            return pred
        u_phys = u_norm.float() * self.std + self.mean
        p_phys = pred.float() * self.std + self.mean

        if self.mean_project:
            p_phys = p_phys - (p_phys.mean(dim=(1, 2, 3), keepdim=True)
                               - u_phys.mean(dim=(1, 2, 3), keepdim=True))

        if self.envelope:
            E_in, D, P_f = self._energy_rates(u_phys)
            E_pred, _, _ = self._energy_rates(p_phys)
            dt = self.spec.dt
            E_max = E_in + self.c_p * P_f * dt
            E_min = (E_in - self.c_d * D * dt).clamp(min=0.0)

            alpha = torch.ones_like(E_pred)
            ceil = E_pred > E_max
            floor = (~ceil) & (E_pred < E_min)
            alpha[ceil] = torch.sqrt(E_max[ceil] / E_pred[ceil].clamp(min=1e-12))
            alpha[floor] = torch.sqrt(E_min[floor] / E_pred[floor].clamp(min=1e-12))

            self._n += int(E_pred.numel())
            self._n_ceil += int(ceil.sum())
            self._n_floor += int(floor.sum())
            self._sum_abs_corr += float((alpha - 1.0).abs().detach().sum())

            if bool((alpha != 1.0).any()):
                p_phys = p_phys * alpha.view(-1, 1, 1, 1)

        return (p_phys - self.mean) / self.std


# --------------------------------------------------------------------
if __name__ == '__main__':
    spec = PhysicsSpec()
    m_fno = FNO2dUV()
    m_anchor = RCLN2DFix2(spec)
    n_fno = sum(p.numel() for p in m_fno.parameters())
    n_anchor = sum(p.numel() for p in m_anchor.parameters())
    print(f'FNO2dUV params: {n_fno:,}')
    print(f'RCLN2DFix2 (anchor) params: {n_anchor:,} (hard core 零参数)')
    x = torch.randn(2, 1, 64, 64)
    print('anchor out:', m_anchor(x).shape)
    env = NSPostproc2D(m_anchor, spec)
    print('postproc out:', env(x).shape, env.env_stats())
    rhs = NSRHS2D(spec)
    print('rhs out:', rhs(x).shape)

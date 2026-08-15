"""
RCLN-1D fix2 —— KS (Kuramoto-Sivashinsky) System C 跨系统 Go/No-Go 模型库
================================================================
标准 KS: u_t = -u*u_x - u_xx - u_xxxx（与 scripts/data_generation/
generate_ks_l32_l44_dt001.py 一致，数据经一步积分一致性验证 relerr~3e-8）。
线性算子: lambda(kappa) = kappa^2 - kappa^4, kappa = 2*pi*n/L（实数）。
能量恒等式（周期边界）: dE/dt = P - D,  E=<u^2>/2, P=<|u_x|^2>, D=<|u_xx|^2>。
均值 u_bar 在周期边界下守恒；lambda(0)=0，故线性传播子与仿射归一化对易，
hard core 可直接作用于归一化空间。

组件:
  SpectralConv1d   —— 手写 rfft 截断谱卷积（不依赖 neuralop）
  FNO1d            —— modes=16, width=64, 4 层（三臂共用配置，参数量对齐）
  LinearExactKS    —— 零参数线性精确传播子 IFFT(exp(lambda*dt) * u_hat)
  RCLN1DFix2       —— anchor 臂: pred = LinearExact(u) + FNO1d(u)，端到端训练
  KSPostproc1D     —— 推理路径零参数物理件 wrapper：均值投影 + 双边能量包络
  rhs_ks_torch     —— 谱精度 KS 右端项（2/3 去混叠，与生成器同口径），
                      供 loss_level 臂物理残差损失使用
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------
# KS 谱工具
# --------------------------------------------------------------------
def ks_rfft_wavenumbers(N, L):
    """rfft 布局的实波数 kappa = 2*pi*n/L, shape [N//2+1]。"""
    return 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)


def ks_lambda(N, L):
    """线性算子 lambda(kappa) = kappa^2 - kappa^4（与生成器完全一致）。"""
    kap = ks_rfft_wavenumbers(N, L)
    return kap**2 - kap**4


def rhs_ks_torch(u_phys, L, dealias=True):
    """标准 KS 右端项: -u*u_x - u_xx - u_xxxx（谱精度，2/3 去混叠）。
    u_phys: [B, 1, N] 物理空间。返回同形状。"""
    B, _, N = u_phys.shape
    kap = torch.tensor(ks_rfft_wavenumbers(N, L), dtype=torch.float32,
                       device=u_phys.device)
    uh = torch.fft.rfft(u_phys, dim=-1)
    u_x = torch.fft.irfft(1j * kap * uh, n=N, dim=-1)
    u_xx = torch.fft.irfft(-kap**2 * uh, n=N, dim=-1)
    u_xxxx = torch.fft.irfft(kap**4 * uh, n=N, dim=-1)
    nl = -u_phys * u_x
    if dealias:
        nh = torch.fft.rfft(nl, dim=-1)
        mask = torch.ones(N // 2 + 1, device=u_phys.device)
        n_cut = N // 3
        mask[n_cut + 1:] = 0.0     # rfft 布局下 |n|>N/3 即高频段
        nl = torch.fft.irfft(nh * mask, n=N, dim=-1)
    return nl - u_xx - u_xxxx


# --------------------------------------------------------------------
# FNO-1D（手写谱卷积）
# --------------------------------------------------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        of = torch.zeros(B, self.weight.shape[1], N // 2 + 1,
                         dtype=torch.cfloat, device=x.device)
        m = min(self.modes, N // 2 + 1)
        of[:, :, :m] = torch.einsum('bim,iom->bom', xf[:, :, :m],
                                    self.weight[:, :, :m])
        return torch.fft.irfft(of, n=N, dim=-1)


class FNO1d(nn.Module):
    """modes=16, width=64, 4 层。输入输出 [B, 1, N]。"""

    def __init__(self, modes=16, width=64, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.skip = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(),
                                  nn.Conv1d(128, 1, 1))

    def forward(self, x):
        h = self.lift(x)
        for sp, sk in zip(self.spec, self.skip):
            h = F.gelu(sp(h) + sk(h))
        return self.proj(h)


# --------------------------------------------------------------------
# 零参数线性精确传播子（hard core）
# --------------------------------------------------------------------
class LinearExactKS(nn.Module):
    """u_{t+1} = IFFT(exp(lambda(kappa)*dt) * FFT(u_t))。零参数。
    lambda(0)=0 -> 常数模不变 -> 与仿射归一化对易，可直接用于归一化空间。"""

    def __init__(self, N=64, L=32.0, dt=0.01):
        super().__init__()
        self.N = N
        self.dt = dt
        self.set_L(L)

    def set_L(self, L):
        self.L = float(L)
        lam = ks_lambda(self.N, self.L)
        dev = self.prop.device if hasattr(self, 'prop') else 'cpu'
        self.register_buffer('prop',
                             torch.tensor(np.exp(lam * self.dt),
                                          dtype=torch.float32).to(dev))

    def forward(self, u):
        # u: [B, 1, N]（物理或归一化空间均可，线性+保常数 => 对易）
        uh = torch.fft.rfft(u.float(), dim=-1)
        return torch.fft.irfft(uh * self.prop, n=self.N, dim=-1)


# --------------------------------------------------------------------
# anchor 臂: LinearExact + Soft(FNO1d)，端到端训练
# --------------------------------------------------------------------
class RCLN1DFix2(nn.Module):
    def __init__(self, N=64, L=32.0, dt=0.01, modes=16, width=64, layers=4):
        super().__init__()
        self.hard = LinearExactKS(N, L, dt)
        self.soft = FNO1d(modes, width, layers)

    def set_L(self, L):
        self.hard.set_L(L)

    def forward(self, u):
        return self.hard(u) + self.soft(u)


# --------------------------------------------------------------------
# 推理路径零参数物理件: 均值投影 + 双边能量包络
# --------------------------------------------------------------------
class KSPostproc1D(nn.Module):
    """包在 anchor 模型外的推理路径 wrapper（训练不用）。

    (a) 均值投影: pred <- pred - (mean(pred) - mean(u_in))，保持 u_bar 守恒。
    (b) 双边能量包络（物理空间）:
        E_{t+1} ∈ [E_t - c_d * D(u_t) * dt,  E_t + c_p * P(u_t) * dt]
        越界则均匀缩放 alpha = sqrt(E_bound / E_pred)。
    c_d, c_p 由 L32 训练段真值标定（脚本内计算）。命中率/校正量全程记录。
    """

    def __init__(self, inner, mean, std, N=64, L=32.0, dt=0.01,
                 c_d=1.0, c_p=1.0, mean_project=True, envelope=True):
        super().__init__()
        self.inner = inner
        self.mean = float(mean)
        self.std = float(std)
        self.N = N
        self.dt = float(dt)
        self.c_d = float(c_d)
        self.c_p = float(c_p)
        self.mean_project = bool(mean_project)
        self.envelope = bool(envelope)
        self.set_L(L)
        self.reset_stats()

    def set_L(self, L):
        self.L = float(L)
        kap = ks_rfft_wavenumbers(self.N, self.L)
        dev = self.kap2.device if hasattr(self, 'kap2') else 'cpu'
        self.register_buffer('kap2', torch.tensor(kap**2, dtype=torch.float32).to(dev))
        if hasattr(self.inner, 'set_L'):
            self.inner.set_L(L)

    def set_norm(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

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

    def _prod_diss(self, u_phys):
        """P = <|u_x|^2>, D = <|u_xx|^2>（谱精度）。u_phys: [B,1,N] -> [B],[B]"""
        uh = torch.fft.rfft(u_phys, dim=-1)
        u_x = torch.fft.irfft(1j * torch.sqrt(self.kap2) * uh, n=self.N, dim=-1)
        u_xx = torch.fft.irfft(-self.kap2 * uh, n=self.N, dim=-1)
        P = (u_x**2).mean(dim=(1, 2))
        D = (u_xx**2).mean(dim=(1, 2))
        return P, D

    def forward(self, u_norm):
        pred = self.inner(u_norm)
        if not (self.mean_project or self.envelope):
            return pred
        u_phys = u_norm.float() * self.std + self.mean
        p_phys = pred.float() * self.std + self.mean

        if self.mean_project:
            p_phys = p_phys - (p_phys.mean(dim=(1, 2), keepdim=True)
                               - u_phys.mean(dim=(1, 2), keepdim=True))

        if self.envelope:
            E_in = 0.5 * (u_phys**2).mean(dim=(1, 2))     # [B]
            E_pred = 0.5 * (p_phys**2).mean(dim=(1, 2))   # [B]
            P, D = self._prod_diss(u_phys)
            E_max = E_in + self.c_p * P * self.dt
            E_min = (E_in - self.c_d * D * self.dt).clamp(min=0.0)

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
                p_phys = p_phys * alpha.view(-1, 1, 1)

        return (p_phys - self.mean) / self.std


# --------------------------------------------------------------------
if __name__ == '__main__':
    m_fno = FNO1d()
    m_anchor = RCLN1DFix2()
    n_fno = sum(p.numel() for p in m_fno.parameters())
    n_anchor = sum(p.numel() for p in m_anchor.parameters())
    print(f'FNO1d params: {n_fno:,}')
    print(f'RCLN1DFix2 (anchor) params: {n_anchor:,} (hard core 零参数)')
    x = torch.randn(2, 1, 64)
    print('anchor out:', m_anchor(x).shape)
    env = KSPostproc1D(m_anchor, mean=0.0, std=1.0)
    print('postproc out:', env(x).shape, env.env_stats())

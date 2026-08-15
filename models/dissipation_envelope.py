#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dissipation_envelope.py — P0-1: 耗散感知能量包络（dissipation-aware energy envelope）

替代旧 v5 的 E(u_{t+1})=E(u_t) 钉死逻辑与 fix2 的单边 1.05x 防爆网之外的空白：
推理路径上逐步施加物理能量约束（仅推理侧 wrapper，冻结 checkpoint 不动）：

    上限: E_{t+1} <= E_t * (1 + gamma)         # 衰减不可压流禁止能量增长
    下限: E_{t+1} >= E_t - c * eps(u_t) * dt    # 单步耗散不得超过物理允许速率

其中 eps(u_t) = nu * <|curl u_t|^2>（谱精度、物理单位），
c 由训练集真值标定（tgv_re6400: ratio=ΔE/(eps·dt) 的 p99.5=1.015，c=p99.5*1.2=1.218）。

设计约束：
- 不跟参考轨迹（无 soft oracle，OOD 可迁移——只需 nu 与归一化统计）
- 不越界零干预；干预率全程记录（干预率~100% 即变相钉死，必须如实报告）
- 均匀正缩放保持无散性（div 投影结论不被破坏）
- 速度场均值≈0（TGV 实测 1e-20），归一化空间缩放与物理空间缩放等价
"""
import numpy as np
import torch
import torch.nn as nn

NU_TGV_RE6400 = 1.0 / 6400.0
C_DEFAULT = 1.218       # p99.5(ΔE/(eps·dt)) * 1.2，训练集标定
GAMMA_DEFAULT = 0.0     # 衰减不可压流严格禁涨（真值 481 步零回弹证实）


class DissipationEnvelopeWrapper(nn.Module):
    """包在评测模型外的耗散一致性包络。接口与 RCLN_UPI_v5 评测口径一致：
    forward(u, target=None, return_components=True, ablation_mode=...) -> (pred, comp)
    每步前调用 set_step_dt(dt) 告知物理步长。"""

    def __init__(self, inner, mean, std, nu=NU_TGV_RE6400,
                 c=C_DEFAULT, gamma=GAMMA_DEFAULT):
        super().__init__()
        self.inner = inner
        # mean/std: (3,) 每通道归一化统计（物理 = norm * std + mean）
        self.register_buffer('ch_mean', torch.tensor(
            np.asarray(mean), dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.register_buffer('ch_std', torch.tensor(
            np.asarray(std), dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.nu = float(nu)
        self.c = float(c)
        self.gamma = float(gamma)
        self._dt = None
        self.reset_stats()

    # ---- 包络状态 ----
    def set_step_dt(self, dt):
        self._dt = float(dt)

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
            'c': self.c, 'gamma': self.gamma, 'nu': self.nu,
        }

    # ---- 物理量（物理单位，谱精度）----
    def _to_phys(self, u_norm):
        return u_norm * self.ch_std + self.ch_mean

    @staticmethod
    def _energy(u_phys):
        return 0.5 * (u_phys ** 2).sum(dim=1).flatten(1).mean(dim=1)  # [B]

    def _dissipation_rate(self, u_phys):
        """eps = nu * <|curl u|^2>，谱精度。u_phys: (B,3,N,N,N) -> [B]"""
        B, _, N, _, _ = u_phys.shape
        kx = torch.fft.fftfreq(N, device=u_phys.device) * N
        KX, KY, KZ = torch.meshgrid(kx, kx, kx, indexing='ij')
        uh = torch.fft.fftn(u_phys, dim=(2, 3, 4))
        wx = torch.fft.ifftn(1j * (KY * uh[:, 2] - KZ * uh[:, 1]), dim=(1, 2, 3))
        wy = torch.fft.ifftn(1j * (KZ * uh[:, 0] - KX * uh[:, 2]), dim=(1, 2, 3))
        wz = torch.fft.ifftn(1j * (KX * uh[:, 1] - KY * uh[:, 0]), dim=(1, 2, 3))
        om2 = (wx.abs() ** 2 + wy.abs() ** 2 + wz.abs() ** 2).flatten(1).mean(dim=1)
        return self.nu * om2  # [B]

    # ---- 前向 ----
    def forward(self, u, target=None, return_components=True, ablation_mode='full'):
        out = self.inner(u, target=target,
                         return_components=return_components,
                         ablation_mode=ablation_mode)
        if return_components:
            pred, comp = out
        else:
            pred, comp = out, None
        dt = self._dt
        if dt is None:
            raise RuntimeError('DissipationEnvelopeWrapper: set_step_dt() 未调用')

        u_phys = self._to_phys(u.float())
        pred_phys = self._to_phys(pred.float())
        E_in = self._energy(u_phys)            # [B]
        E_pred = self._energy(pred_phys)       # [B]
        eps_in = self._dissipation_rate(u_phys)  # [B]

        E_max = E_in * (1.0 + self.gamma)
        E_min = (E_in - self.c * eps_in * dt).clamp(min=0.0)

        alpha = torch.ones_like(E_pred)
        ceil = E_pred > E_max
        floor = (~ceil) & (E_pred < E_min)
        alpha[ceil] = torch.sqrt(E_max[ceil] / E_pred[ceil].clamp(min=1e-12))
        alpha[floor] = torch.sqrt(E_min[floor] / E_pred[floor].clamp(min=1e-12))

        self._n += int(E_pred.numel())
        self._n_ceil += int(ceil.sum())
        self._n_floor += int(floor.sum())
        self._sum_abs_corr += float((alpha - 1.0).abs().sum())

        if bool((alpha != 1.0).any()):
            a = alpha.view(-1, 1, 1, 1, 1)
            pred = ((pred_phys * a - self.ch_mean) / self.ch_std).to(pred.dtype)
        return (pred, comp) if return_components else pred

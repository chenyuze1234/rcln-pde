#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
physics_correct.py — PhysicsCorrect 3D（approximate reproduction）

源自 p0_final_battery.py（Huang & Perdikaris 风格的推理期谱校正，
对角雅可比近似）。推理期 wrapper：谱扩散参考 + 残差回退 + 无散投影。
按任务书许可标注为 approximate reproduction。
"""
import numpy as np
import torch


class PhysicsCorrect3D:
    """Simplified PhysicsCorrect — inference-time spectral correction."""

    def __init__(self, model, nu=1.0 / 6400, dt=0.033, alpha=0.5, N=64):
        self.model = model
        self.nu = nu
        self.dt = dt
        self.alpha = alpha
        self.N = N
        k = torch.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(k, k, k[:N // 2 + 1], indexing='ij')
        self.k2 = kx ** 2 + ky ** 2 + kz ** 2
        self.k2[0, 0, 0] = 1e-10

    def _spectral_diffusion(self, u):
        k2 = self.k2.to(u.device)
        u_hat = torch.fft.rfftn(u, dim=(2, 3, 4))
        diff = u_hat * torch.exp(-self.dt * self.nu * k2.unsqueeze(0).unsqueeze(0))
        out = torch.fft.irfftn(diff, s=u.shape[2:], dim=(2, 3, 4))
        return out.real if torch.is_complex(out) else out

    def _div_free_projection(self, u):
        if u.shape[1] < 3:
            return u
        B, C, H, W, D = u.shape
        u_hat = torch.fft.rfftn(u, dim=(2, 3, 4))
        kx = (torch.fft.fftfreq(H, device=u.device) * 2 * np.pi).view(1, 1, H, 1, 1)
        ky = (torch.fft.fftfreq(W, device=u.device) * 2 * np.pi).view(1, 1, 1, W, 1)
        kz = (torch.fft.rfftfreq(D, device=u.device) * 2 * np.pi).view(1, 1, 1, 1, D // 2 + 1)
        kdu = kx * u_hat[:, 0:1] + ky * u_hat[:, 1:2] + kz * u_hat[:, 2:3]
        ksq = kx ** 2 + ky ** 2 + kz ** 2
        ksq_s = torch.where(ksq < 1e-12, torch.ones_like(ksq), ksq + 1e-10)
        u_hat[:, 0:1] -= kdu * kx / ksq_s
        u_hat[:, 1:2] -= kdu * ky / ksq_s
        u_hat[:, 2:3] -= kdu * kz / ksq_s
        out = torch.fft.irfftn(u_hat, s=(H, W, D), dim=(2, 3, 4))
        return out.real if torch.is_complex(out) else out

    def correct(self, u_prev, u_pred):
        with torch.no_grad():
            s_u = self._spectral_diffusion(u_prev)
            residual = u_pred - s_u
            u_corr = u_pred - self.alpha * residual
            return self._div_free_projection(u_corr)

    def __call__(self, u):
        u_pred = self.model(u)
        return self.correct(u, u_pred)

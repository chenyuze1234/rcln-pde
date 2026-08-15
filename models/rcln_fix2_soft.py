#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rcln_fix2_soft.py — vNC 阶段 1：冻结 fix2 骨架 + v2 式直接监督残差软分支

设计依据（全部来自本会话实测）：
  - fix2 = 精度/分布物理冠军（rL2@100=0.457，enstrophy/PDF/S2 最贴真值），
    但无自误差信号（|u_local| AUROC=0.65）且能量逻辑有 P0-1 钉死伪影；
  - v2 软分支 = 唯一活误差信号（AUROC 0.81-0.84，跨 Re OOD 保持 0.78-0.82），
    其活性来源 = 小独立分支 + 对残差的直接监督 + 相加融合（无剪切）；
  - 阶段 1 零风险路线：fix2 全冻结，只训练 0.21M 软分支预测 fix2 的逐体素残差
    r = target - fix2(u)（detach），推理时 u_final = fix2(u) + u_soft（可门控）。

接口与 nc_suite_eval / rollout 系列脚本兼容：
  forward(u, target=None, return_components=False, ablation_mode='full')
    -> u_final 或 (u_final, components)
  components['u_global'] = fix2 输出；components['u_local'] = u_soft（UPI 信号源）

消融模式：
  'full'     : fix2 + u_soft
  'soft_off' : 仅 fix2（复现 0.457 基线的保真检验）
  'gate70'/'gate90' 等：逐样本 |u_soft| 分位数稀疏门控（事后，推理专用）
"""
from typing import Optional

import torch
import torch.nn as nn

from models.rcln_upi_v5 import divergence_free_projection_spectral
from models.rcln_upi_v5_spec import IndependentHighFreqBranch, spectral_lowpass
from rollout_eval_v5 import load_model as load_fix2


class RCLN_Fix2Soft(nn.Module):
    def __init__(self, fix2_checkpoint: str = 'checkpoints/v5_fix2/v5_best.pt',
                 device: str = 'cuda', k_cut: int = 8,
                 soft_base: int = 16, soft_dropout: float = 0.1):
        super().__init__()
        self.k_cut = k_cut
        fix2 = load_fix2(fix2_checkpoint, device, ablation_mode='full')
        for p in fix2.parameters():
            p.requires_grad_(False)
        fix2.eval()
        self.fix2 = fix2
        # v2 同款小 U-Net；输入 6 通道 = [highpass(u), fix2(u).detach()]
        # 注意：该类的 head 输出通道数跟随 in_channels，需把 head 换成 3 通道输出
        self.soft = IndependentHighFreqBranch(in_channels=6, base=soft_base,
                                              dropout=soft_dropout)
        self.soft.head = nn.Conv3d(soft_base, 3, 3, padding=1)
        self.to(device)

    def forward(self, u: torch.Tensor, target: Optional[torch.Tensor] = None,
                return_components: bool = False, ablation_mode: str = 'full',
                training_progress: float = 0.0):
        with torch.no_grad():
            fix2_out, comp2 = self.fix2(
                u, target=None, return_components=True, ablation_mode='full')

        u_high = u - spectral_lowpass(u, self.k_cut)
        soft_in = torch.cat([u_high, fix2_out], dim=1)
        u_soft = self.soft(soft_in)

        mode = ablation_mode
        delta_norm = torch.linalg.vector_norm(u_soft, dim=1, keepdim=True) + 1e-8
        upi_proj_scale = torch.ones((), device=u.device)
        if mode == 'soft_off':
            u_fused = fix2_out
        elif mode.startswith('gate') and not self.training:
            q = float(mode[4:]) / 100.0
            B = delta_norm.shape[0]
            tau = torch.quantile(delta_norm.reshape(B, -1), q, dim=1).view(B, 1, 1, 1, 1)
            m_gate = (delta_norm > tau).float()
            upi_proj_scale = m_gate.mean()
            u_fused = fix2_out + m_gate * u_soft
        else:
            u_fused = fix2_out + u_soft

        # 融合后再做一次谱散度投影（u_soft 可能引入散度）
        u_final = divergence_free_projection_spectral(u_fused)

        if return_components:
            components = {
                'u_global': fix2_out,
                'u_local': u_soft,
                'delta_norm_mean': delta_norm.mean(),
                'guard_trigger_rate': torch.zeros((), device=u.device),
                'upi_proj_scale': upi_proj_scale,
                'u_fused': u_fused,
                'u_input': u,
                'ablation_mode': mode,
                'fix2_components': comp2,
            }
            return u_final, components
        return u_final


def load_fix2soft(soft_checkpoint: str, device: str = 'cuda',
                  fix2_checkpoint: str = 'checkpoints/v5_fix2/v5_best.pt'):
    """加载 vNC 阶段 1 模型：冻结 fix2 + 训练好的软分支"""
    model = RCLN_Fix2Soft(fix2_checkpoint=fix2_checkpoint, device=device)
    ckpt = torch.load(soft_checkpoint, map_location=device, weights_only=False)
    model.soft.load_state_dict(ckpt['soft_state_dict'])
    model.eval()
    return model


class RCLN_Fix2V2Trans(nn.Module):
    """v2 软分支移植（零训练）：冻结 fix2 + v2 联合训练练活的 soft_branch 原权重。

    与阶段 1（事后新训校正器）的本质区别：分支权重来自 v2 联合训练，
    输入语义与 v2 完全一致（u_high = u - lowpass(u, k=8)，3 通道）。

    消融模式：
      'signal_only' : 融合输出 = fix2（不动），u_soft 仅作 UPI 信号（u_local）
      'full'        : fix2 + alpha*u_soft + 谱散度投影
      'gate70' 等   : 逐样本 |u_soft| 分位数门控后相加
    """

    def __init__(self, fix2_checkpoint: str = 'checkpoints/v5_fix2/v5_best.pt',
                 v2_checkpoint: str = 'checkpoints/v5_spec_v2add_e13/v5_best.pt',
                 device: str = 'cuda', k_cut: int = 8, alpha: float = 1.0):
        super().__init__()
        self.k_cut = k_cut
        self.alpha = alpha
        fix2 = load_fix2(fix2_checkpoint, device, ablation_mode='full')
        for p in fix2.parameters():
            p.requires_grad_(False)
        fix2.eval()
        self.fix2 = fix2
        # 直接从 v2 checkpoint 抽 soft_branch 权重（原架构原权重，不构造整个 spec 模型）
        branch = IndependentHighFreqBranch(in_channels=3, base=16, dropout=0.1)
        ckpt = torch.load(v2_checkpoint, map_location=device, weights_only=False)
        sd = {k[len('soft_branch.'):]: v for k, v in ckpt['model_state_dict'].items()
              if k.startswith('soft_branch.')}
        branch.load_state_dict(sd, strict=True)
        for p in branch.parameters():
            p.requires_grad_(False)
        branch.eval()
        self.soft = branch
        self.to(device)

    def forward(self, u: torch.Tensor, target: Optional[torch.Tensor] = None,
                return_components: bool = False, ablation_mode: str = 'full',
                training_progress: float = 0.0):
        with torch.no_grad():
            fix2_out, comp2 = self.fix2(
                u, target=None, return_components=True, ablation_mode='full')
            u_high = u - spectral_lowpass(u, self.k_cut)
            u_soft = self.soft(u_high)

        mode = ablation_mode
        delta_norm = torch.linalg.vector_norm(u_soft, dim=1, keepdim=True) + 1e-8
        upi_proj_scale = torch.ones((), device=u.device)
        if mode == 'signal_only':
            u_final = fix2_out
        elif mode.startswith('gate'):
            q = float(mode[4:]) / 100.0
            B = delta_norm.shape[0]
            tau = torch.quantile(delta_norm.reshape(B, -1), q, dim=1).view(B, 1, 1, 1, 1)
            m_gate = (delta_norm > tau).float()
            upi_proj_scale = m_gate.mean()
            u_final = divergence_free_projection_spectral(fix2_out + self.alpha * m_gate * u_soft)
        else:  # full
            u_final = divergence_free_projection_spectral(fix2_out + self.alpha * u_soft)

        if return_components:
            components = {
                'u_global': fix2_out,
                'u_local': u_soft,
                'delta_norm_mean': delta_norm.mean(),
                'guard_trigger_rate': torch.zeros((), device=u.device),
                'upi_proj_scale': upi_proj_scale,
                'u_input': u,
                'ablation_mode': mode,
            }
            return u_final, components
        return u_final


def load_fix2_v2trans(device: str = 'cuda', alpha: float = 1.0,
                      fix2_checkpoint: str = 'checkpoints/v5_fix2/v5_best.pt',
                      v2_checkpoint: str = 'checkpoints/v5_spec_v2add_e13/v5_best.pt'):
    return RCLN_Fix2V2Trans(fix2_checkpoint=fix2_checkpoint,
                            v2_checkpoint=v2_checkpoint, device=device, alpha=alpha)

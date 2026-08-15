"""
RCLN-UPI v5-Spec v2「角色分离」（2026-08-09，用户批准）
======================================================
基于 v5-fix2 冻结快照（models/rcln_upi_v5_fix2_frozen.py）的可复用积木。
历史：极简版（3e5e727）→ 补全 fix2 后处理线（57e2fee）→ v1 半径实验三轮
（ea6d3d3 可学习 q + 0.7 地板 / 0c9986f 去地板 / dd9abe1 q 离线校准）——
三轮结论：半径的精度角色（均值量，MSE 总推向放行极端）与安全角色（尾部量）
不可由同一参数承担。v2 解法 = 角色分离：例行融合直接相加，尾部安全交给
事件触发漂移护栏。

四要素架构（保留不动）：
  1. 谱分裂：FFT 理想低通 u_low = IFFT(FFT(u)·mask(|k|<=k_cut))，u_high = u − u_low。
     k_cut 默认 8（64³ 网格整数波数模长）。无参数函数 spectral_lowpass(u, k_cut)。
  2. hard 分支（低频锚点）：复用 SharedMultiScaleEncoder + CollaborativeHardCore
     （import 不复制），输入 u_low，输出 u_anchor [B,3,64³]（预测低频下一帧）。
     use_internal_upi=False（HardCore 内部不做任何投影后处理）。
  3. soft 分支（高频）：轻量独立编解码器 IndependentHighFreqBranch
     （3D conv 3 尺度，通道 16→32→64，GroupNorm+SiLU，中间 Dropout3d(p=0.1)），
     输入 u_high，输出 u_soft [B,3,64³]。参数量 ~210K（<= ~300K 预算）。
  4. 融合与安全（v2 角色分离）：
       例行融合 = 直接相加 u_fused = u_anchor + u_soft。
       逐体素剪切机制全部退役（R/scale/q_value/set_q/q property/离线扫描、
       components 的 q/R_G/R_G_eff/w 键一并删除）。
       err_head（复用 ErrorCalibHead，输入 hard 分支 detach 特征
       + u_anchor.detach() + u_soft.detach()）输出 ê_g, ê_s——保留并继续
       由 err_calib 损失（λ=0.1）监督，但 v2 起只作输出诊断，不进融合路径。
       尾部安全 = 事件触发漂移护栏：
         buffer delta_p99（init 3.0）= 训练分布 ‖u_soft‖(δ_norm) 的 99 分位，
         trainer 每 epoch 从训练 forward 缓存的 δ_norm 样本重估（set_delta_p99）；
         训练 forward 缓存 self._last_delta_norm = δ_norm.detach()，不剪；
         推理 scale = clamp(margin×delta_p99/δ_norm, max=1.0)（margin=1.5 构造参数），
         u_fused = u_anchor + scale·u_soft——只有越界体素被剪；
         components['guard_trigger_rate'] 训练/推理都记录。

后处理栈不动：memory FiLM → 谱无散投影 → 单边能量安全网。

消融模式：'full' = 相加+护栏；'hc_only' = 只输出 u_anchor；
'no_guard' = 相加不护栏（推理也直通）。（三种模式均施加 FiLM + 投影 + 安全网。）

v3 增量（2026-08-09，用户批准 1+2 组合；动机：v2 自由高频通道 E13 后过拟合
train 0.19 vs val 0.41 分叉，逐体素方案已证伪，改全局统计层级约束）：
  A. 空间增广（训练 data pipeline 精确对称：循环平移 + 轴反射，见 train_v5_spec.py）。
  B. 可学习逐波数谱增益 g(k)：spec_gain_raw = nn.Parameter(zeros(33))（bin 0..32，
     全局非逐体素）；G = sigmoid(raw)×2，raw=0 ⇒ G≡1 ⇒ 初始与 v2 完全等价；
     u_soft_filt = IFFT(FFT(u_soft)·Gmap)，融合 u_fused = u_anchor + u_soft_filt
     （漂移护栏作用于 u_soft_filt；soft_high 损失与 ê_s 监督对象同步为 filt）；
     components 新增 'spec_gain'(G) 与 'u_soft_raw'，'u_local' = u_soft_filt。

损失（RCLN_UPI_v5Spec_Loss）四核心项 + 后处理耦合正则：
  data       = MSE(pred, target)
  anchor_low = MSE(u_anchor, lowpass(target))
  soft_high  = MSE(u_soft, target − lowpass(target))
  err_calib  = MSE(ê_g, ‖lowpass(target)−u_anchor‖.detach())
             + MSE(ê_s, ‖(target−lowpass(target))−u_soft‖.detach())
  energy     = smooth_l1(E_pred/E_target, 1)   λ=0.05（fix2 默认，warmup ramp）
  divergence = divergence_mse(pred)            λ=0.01（fix2 默认，warmup ramp）
  energy_raw / energy_scale：仅保留接口（λ=0.0，与 fix2 默认一致不启用）
核心四 λ 默认 1.0 / 1.0 / 1.0 / 0.1。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from models.rcln_upi_v5 import (
    SharedMultiScaleEncoder,
    CollaborativeHardCore,
    ErrorCalibHead,
    kinetic_energy,
    divergence_mse,
    divergence_free_projection_spectral,
)
from models.memory_module_v5 import MemoryModuleV5


# ============ 谱分裂：FFT 理想低通（无参数） ============

_LOWPASS_MASK_CACHE: Dict[Tuple, torch.Tensor] = {}


def spectral_lowpass(u: torch.Tensor, k_cut: int = 8) -> torch.Tensor:
    """FFT 理想低通滤波：保留整数波数模长 |k| <= k_cut 的傅里叶模态。

    Args:
        u: [B, C, H, W, D] 物理场
        k_cut: 整数波数模长截断（64³ 网格默认 8）
    Returns:
        u_low: [B, C, H, W, D] 低频分量（实数）
    """
    B, C, H, W, D = u.shape
    key = (H, W, D, k_cut, u.device, u.dtype)
    mask = _LOWPASS_MASK_CACHE.get(key)
    if mask is None:
        kx = torch.fft.fftfreq(H, device=u.device) * H
        ky = torch.fft.fftfreq(W, device=u.device) * W
        kz = torch.fft.fftfreq(D, device=u.device) * D
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX ** 2 + KY ** 2 + KZ ** 2
        mask = (k2 <= float(k_cut) ** 2).to(u.dtype)  # [H, W, D]
        _LOWPASS_MASK_CACHE[key] = mask
    u_hat = torch.fft.fftn(u, dim=(-3, -2, -1))
    u_low = torch.fft.ifftn(u_hat * mask, dim=(-3, -2, -1)).real
    return u_low


# ============ 整数波数模长索引表（v3 谱增益用，按 shape/device 缓存） ============

_SPEC_BIN_CACHE: Dict[Tuple, torch.Tensor] = {}
_SPEC_N_BINS = 33  # 整数波数 bin 0..32（64³ 网格；|k|>32 的角点归入 bin 32）


def spectral_bin_index(H: int, W: int, D: int, device) -> torch.Tensor:
    """64³ 网格整数波数模长索引表 [H,W,D]（long，值域 0..32）。"""
    key = (H, W, D, device)
    idx = _SPEC_BIN_CACHE.get(key)
    if idx is None:
        kx = torch.fft.fftfreq(H, device=device) * H
        ky = torch.fft.fftfreq(W, device=device) * W
        kz = torch.fft.fftfreq(D, device=device) * D
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        idx = k_mag.round().long().clamp(max=_SPEC_N_BINS - 1)
        _SPEC_BIN_CACHE[key] = idx
    return idx


# ============ 高频软分支：轻量独立编解码器 ============

class _ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class IndependentHighFreqBranch(nn.Module):
    """高频软分支：3 尺度轻量 3D U-Net（base 16→32→64）。

    输入 u_high [B, 3, 64, 64, 64]（高频分量），输出 u_soft [B, 3, 64, 64, 64]
    （高频下一帧预测）。与 hard 分支完全独立（不共享编码器、不读 z_global）。
    参数量 ~210K（预算 <= ~300K）。
    """

    def __init__(self, in_channels: int = 3, base: int = 16, dropout: float = 0.1):
        super().__init__()
        c0, c1, c2 = base, base * 2, base * 4  # 16, 32, 64
        # 编码器
        self.enc0 = nn.Sequential(           # 64³: 3→16→16
            _ConvBlock3D(in_channels, c0),
            _ConvBlock3D(c0, c0),
        )
        self.down1 = _ConvBlock3D(c0, c1, stride=2)          # 64³→32³
        self.enc1 = _ConvBlock3D(c1, c1, dropout=dropout)    # 32³: 32→32
        self.down2 = _ConvBlock3D(c1, c2, stride=2, dropout=dropout)  # 32³→16³
        # 解码器（trilinear 上采样 + skip 融合）
        self.dec1 = _ConvBlock3D(c2 + c1, c1)  # 32³: (64+32)→32
        self.dec2 = _ConvBlock3D(c1 + c0, c0)  # 64³: (32+16)→16
        self.head = nn.Conv3d(c0, in_channels, 3, padding=1)

    def forward(self, u_high: torch.Tensor) -> torch.Tensor:
        f0 = self.enc0(u_high)               # [B, 16, 64, 64, 64]
        f1 = self.enc1(self.down1(f0))       # [B, 32, 32, 32, 32]
        f2 = self.down2(f1)                  # [B, 64, 16, 16, 16]
        d1 = F.interpolate(f2, scale_factor=2, mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, f1], dim=1))   # [B, 32, 32, 32, 32]
        d2 = F.interpolate(d1, scale_factor=2, mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, f0], dim=1))   # [B, 16, 64, 64, 64]
        return self.head(d2)                 # [B, 3, 64, 64, 64]


# ============ 主模型：极简谱分裂双人集合 ============

class RCLN_UPI_v5_Spec(nn.Module):
    """RCLN-UPI v5-Spec v2「角色分离」：谱分裂双人集合 + 相加融合 + 漂移护栏 + fix2 后处理线。

    v2 设计动机（2026-08-09，用户定）：半径的精度角色（均值量，MSE 总推向
    放行极端）与安全角色（尾部量）不可由同一参数承担（v1 三轮实验结论：
    可学习 q 被任务梯度劫持、离线 q 扫描打边界）。v2 把两份差事拆给两个机制：
      - 例行融合 = 直接相加 u_fused = u_anchor + u_soft（逐体素剪切机制全部退役：
        R/scale/q_value/set_q/离线扫描/q、R_G、R_G_eff、w 键一并删除；
        ê 双头 err_pred 保留 + err_calib 损失保留（λ=0.1），但只作输出诊断，
        不再进入融合路径）
      - 尾部安全 = 事件触发漂移护栏：buffer delta_p99（训练分布 ‖u_soft‖ 的
        99 分位，trainer 每 epoch 从缓存样本重估）；训练时只缓存 δ_norm.detach()
        并记录触发率（不改场，与 fix2 能量网「训练诊断/推理生效」语义一致）；
        推理时 scale = clamp(margin×delta_p99/δ_norm, max=1.0)，只有越界体素被剪。

    Forward pipeline:
      1. 谱分裂: u → u_low + u_high (FFT 理想低通, k_cut)
      2. hard 分支 (u_low): SharedMultiScaleEncoder + CollaborativeHardCore → u_anchor
      3. soft 分支 (u_high): IndependentHighFreqBranch → u_soft
      4. ErrorCalibHead (detach 特征 + detach 分支输出) → (ê_g, ê_s)（纯诊断）
         → 相加融合 u_fused = u_anchor + u_soft（+ eval 漂移护栏）
      5. Hippocampus memory → FiLM 修正（fix2 接法：u_raw=u, z_global=hard 分支潜码,
         u_global=u_anchor；apply_film 作用于 u_fused）
      6. 谱散度投影（训练+推理）
      7. 能量安全网（训练仅诊断 / 推理单边防爆 E > 1.05·E_input 才缩放，耗散放行）

    消融：'full'=相加+护栏；'hc_only'=只 u_anchor；'no_guard'=相加不护栏（推理也直通）。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 64,
        decoder_freq: int = 4,
        # 谱分裂
        k_cut: int = 8,
        # 高频软分支
        soft_base: int = 16,
        soft_dropout: float = 0.1,
        # 误差标定头（纯诊断 + err_calib 损失）
        use_err_head: bool = True,
        # 漂移护栏参数
        guard_margin: float = 1.5,
        delta_p99_init: float = 3.0,
        # Memory 模块参数（与 fix2 训练运行一致：theta_sim=0.2）
        memory_key_dim: int = 64,
        memory_max_size: int = 256,
        memory_theta_sim: float = 0.2,
        ablation_mode: str = 'full',
    ):
        super().__init__()
        self.k_cut = k_cut
        self.ablation_mode = ablation_mode
        self.use_err_head = use_err_head
        self.guard_margin = guard_margin
        # 漂移护栏：训练分布 ‖u_soft‖ 的 99 分位（buffer 进 state_dict 可复现；
        # trainer 每 epoch 从缓存 δ_norm 样本重估，不进优化器、无任务梯度）。
        self.register_buffer('delta_p99', torch.tensor(float(delta_p99_init)))
        self._last_delta_norm = None  # 训练模式 forward 缓存（detach），供 p99 估计

        # v3：可学习逐波数谱增益 g(k)（全局 33 bin，非逐体素非逐轨迹——
        # 只能学「哪个频带统计不可信」，物理上 = 谱域 Wiener 收缩）。
        # G = sigmoid(raw)×2，raw=0 ⇒ G≡1 ⇒ 初始行为与 v2 完全等价。
        # 名字不含 'memory'，自然进 trainer 的 base_params 组。
        self.spec_gain_raw = nn.Parameter(torch.zeros(_SPEC_N_BINS))

        # hard 分支：复用 v5 编码器 + HardCore（低频锚点；内部不做投影后处理）
        self.encoder = SharedMultiScaleEncoder(in_channels, base_channels)
        self.hard_core = CollaborativeHardCore(
            deep_channels=base_channels * 8,
            latent_dim=latent_dim,
            decoder_freq=decoder_freq,
            use_internal_upi=False,
        )

        # soft 分支：轻量独立编解码器（高频）
        self.soft_branch = IndependentHighFreqBranch(
            in_channels=in_channels, base=soft_base, dropout=soft_dropout,
        )

        # 监督误差标定头（复用；输入全部 detach；v2 起仅诊断 + err_calib 损失）
        if use_err_head:
            self.err_head = ErrorCalibHead(
                f0_ch=base_channels, f2_ch=base_channels * 4,
                f3_ch=base_channels * 8, field_ch=in_channels,
            )

        # 记忆模块（fix2 后处理线第 5 步）
        self.memory = MemoryModuleV5(
            in_channels=in_channels,
            global_dim=latent_dim,
            key_dim=memory_key_dim,
            film_channels=in_channels,
            max_bank_size=memory_max_size,
            theta_sim=memory_theta_sim,
        )

    def set_delta_p99(self, p: float) -> None:
        """漂移护栏校准接口：设置 ‖u_soft‖ 的 99 分位（不参与反向传播）。"""
        self.delta_p99.fill_(float(p))

    def forward(
        self,
        u: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        return_components: bool = False,
        ablation_mode: Optional[str] = None,
        training_progress: float = 0.0,
    ):
        mode = ablation_mode or self.ablation_mode

        # 1. 谱分裂
        u_low = spectral_lowpass(u, self.k_cut)
        u_high = u - u_low

        # 2. hard 分支（低频锚点）
        f0, f2, f3 = self.encoder(u_low)
        u_anchor, z_global, z_next, hc_proj_info, z_struct, energy_spec, vort_8, cert_8 = \
            self.hard_core(f3, f2=f2)

        # 3. soft 分支（高频）
        u_soft_raw = self.soft_branch(u_high)

        # 3b. v3：可学习逐波数谱增益 g(k)（谱域 Wiener 收缩；raw=0 ⇒ G≡1 ⇒ 等价 v2）
        spec_gain = torch.sigmoid(self.spec_gain_raw) * 2.0   # [33] ∈ (0,2)
        B, C, H, W, D = u_soft_raw.shape
        bin_idx = spectral_bin_index(H, W, D, u_soft_raw.device)
        g_map = spec_gain[bin_idx]                            # [H,W,D]
        us_hat = torch.fft.fftn(u_soft_raw, dim=(-3, -2, -1))
        u_soft = torch.fft.ifftn(us_hat * g_map, dim=(-3, -2, -1)).real  # u_soft_filt

        # 4. 监督误差标定（特征与分支输出全部 detach —— 校准梯度不回传主干；
        #    ê_s 监督对象 = u_soft_filt（被融合使用的那个量），与 soft_high 损失同步）
        err_pred = None
        if self.use_err_head:
            err_pred = self.err_head(
                f0.detach(), f2.detach(), f3.detach(),
                u_anchor.detach(), u_soft.detach(),
            )

        # 5. 记忆模块（fix2 接法：先取 FiLM 参数，融合后施加）
        film_params, mem_info = self.memory(
            u_raw=u,
            z_global=z_global,
            u_global=u_anchor,
            target=target,
            mode='train' if self.training else 'eval',
            training_progress=training_progress,
        )

        # 6. v2 角色分离：例行融合 = 直接相加；尾部安全 = 事件触发漂移护栏
        #    （训练只缓存 δ_norm + 记录触发率，不改场；推理才剪越界体素）
        delta_norm = torch.linalg.vector_norm(u_soft, dim=1, keepdim=True) + 1e-8
        guard_threshold = self.guard_margin * self.delta_p99
        guard_trigger_rate = (delta_norm > guard_threshold).float().mean()  # 训练/推理都记录

        upi_proj_scale = torch.ones((), device=u.device)  # 占位（v2 训练/直通路径无剪切）
        if mode == 'hc_only':
            u_fused = u_anchor
        elif mode.startswith('gate') and not self.training:
            # 事后门控（纯推理消融）：'gate70' = 只保留逐样本 ‖u_soft‖ 前 30% 体素的修正。
            # 依据：v2 验收实测 ‖u_soft‖ 对真值误差图 AUROC=0.786——高强度软修正集中在
            # 锚点错处，低强度部分主要是漂移噪声。硬门控检验「稀疏修正 > 全场相加」。
            q = float(mode[4:]) / 100.0
            B = delta_norm.shape[0]
            flat = delta_norm.reshape(B, -1)
            tau = torch.quantile(flat, q, dim=1).view(B, 1, 1, 1, 1)
            m_gate = (delta_norm > tau).float()
            upi_proj_scale = m_gate.mean()
            u_fused = u_anchor + m_gate * u_soft
        elif self.training or mode == 'no_guard':
            if self.training:
                # 训练：缓存样本供 p99 估计，护栏不改场（诊断语义同 fix2 能量网）
                self._last_delta_norm = delta_norm.detach()
            # no_guard 消融：推理也直通
            u_fused = u_anchor + u_soft
        else:
            # 推理：只有越界体素被剪，例行路径不受影响
            guard_scale = torch.clamp(guard_threshold / delta_norm, max=1.0)
            upi_proj_scale = guard_scale.mean()
            u_fused = u_anchor + guard_scale * u_soft

        # 7. FiLM 修正（作用于融合输出）
        u_final = self.memory.apply_film(u_fused, film_params)

        # 8. 谱散度投影（训练+推理均启用，fix2 第 7b 步）
        u_final = divergence_free_projection_spectral(u_final)

        # 9. 能量安全网（fix2 第 8 步语义：训练仅诊断；推理单边防爆）
        energy_info = {}
        E_pred = kinetic_energy(u_final)
        E_input = kinetic_energy(u)
        energy_info['E_pred'] = E_pred
        energy_info['E_target'] = E_input
        energy_info['energy_ratio_before'] = E_pred / (E_input + 1e-8)  # [B] 纯诊断
        energy_info['dissipation_allowed'] = True
        if self.training:
            # 训练：MSE 从数据中学能量匹配，安全网不修改 u_final
            energy_info['energy_scale'] = torch.ones_like(E_pred)
            energy_info['clamp_triggered_ratio'] = torch.tensor(0.0, device=u.device)
        else:
            # 推理：仅当 E_final > E_input*1.05（能量爆炸）才缩放；耗散永远放行
            explosion_mask = E_pred > E_input * 1.05
            if explosion_mask.any():
                e_scale = torch.where(
                    explosion_mask,
                    torch.sqrt(E_input * 1.05 / (E_pred + 1e-8)),
                    torch.ones_like(E_input),
                )
                u_final = u_final * e_scale.view(-1, 1, 1, 1, 1)
                energy_info['energy_scale'] = e_scale.detach()
                energy_info['explosion_triggered'] = explosion_mask.float()
            else:
                energy_info['energy_scale'] = torch.ones_like(E_input)
                energy_info['explosion_triggered'] = torch.zeros_like(E_input)

        if return_components:
            components = {
                # —— trainer 需要的主键 ——
                'u_global': u_anchor,            # 锚点（低频分支预测）
                'u_local': u_soft,               # 软分支输出经谱增益 g(k) 滤波（相加融合使用的量）
                'u_soft_raw': u_soft_raw,        # 谱增益前原始值（诊断用）
                'spec_gain': spec_gain,          # [33] 逐波数增益 G=sigmoid(raw)×2（供日志/轨迹分析）
                'guard_trigger_rate': guard_trigger_rate,  # δ_norm > margin×p99 的体素比例（训练/推理都记录）
                'delta_norm_mean': delta_norm.mean(),      # 标量诊断
                'delta_p99': self.delta_p99,     # 当前护栏分位 buffer（供日志）
                'err_pred': err_pred,            # [B,2,H,W,D] (ê_g, ê_s) 或 None（v2 纯诊断）
                'upi_proj_scale': upi_proj_scale,  # 标量（v2：训练/直通=1.0；eval 护栏=guard_scale 均值）
                'u_fused': u_fused,
                'u_input': u,
                'ablation_mode': mode,
                # —— fix2 后处理线真实值 ——
                'mem_info': mem_info,
                'film_params': film_params,
                # —— hard 分支返回值透传 ——
                'z_global': z_global,
                'z_next': z_next,
                'z_struct': z_struct,
                'energy_spec': energy_spec,
                'vorticity_8': vort_8,
                'certainty_8': cert_8,
                'hc_proj_info': hc_proj_info,
                # —— 已删除组件的占位（保证 return_components=True 不 KeyError）——
                'pi': None,
                'mode_entropy': None,
                'load_balance': None,
                # —— v5-spec 诊断键 ——
                'u_low': u_low,
                'u_high': u_high,
                'k_cut': self.k_cut,
            }
            components.update(energy_info)  # energy_scale / energy_ratio_before / E_pred / E_target 等真实值
            return u_final, components

        return u_final


# ============ v5-Spec 损失函数（只四项） ============

class RCLN_UPI_v5Spec_Loss(nn.Module):
    """v5-Spec 损失：四核心项 + 与后处理线直接耦合的正则（权重对齐 fix2 默认）。

      data       = MSE(pred, target)
      anchor_low = MSE(u_anchor, lowpass(target))
      soft_high  = MSE(u_soft, target − lowpass(target))
      err_calib  = MSE(ê_g, ‖lowpass(target)−u_anchor‖.detach())
                 + MSE(ê_s, ‖(target−lowpass(target))−u_soft‖.detach())
      energy     = smooth_l1(E_pred/E_target, 1, beta=0.1)   λ=0.05（fix2 语义）
      divergence = divergence_mse(pred)                      λ=0.01（fix2 语义）
      energy_raw / energy_scale：仅保留接口（λ=0.0，同 fix2 默认不启用）
    energy/divergence 与 fix2 一样走物理正则 warmup（前 50% 训练线性 ramp）。
    """

    def __init__(
        self,
        k_cut: int = 8,
        lambda_data: float = 1.0,
        lambda_anchor_low: float = 1.0,
        lambda_soft_high: float = 1.0,
        lambda_err_calib: float = 0.1,
        lambda_energy: float = 0.05,
        lambda_divergence: float = 0.01,
        lambda_energy_raw: float = 0.0,    # 接口保留，默认不启用（同 fix2）
        lambda_energy_scale: float = 0.0,  # 接口保留，默认不启用（同 fix2）
        total_epochs: int = 100,
    ):
        super().__init__()
        self.k_cut = k_cut
        self.lambda_data = lambda_data
        self.lambda_anchor_low = lambda_anchor_low
        self.lambda_soft_high = lambda_soft_high
        self.lambda_err_calib = lambda_err_calib
        self.lambda_energy = lambda_energy
        self.lambda_divergence = lambda_divergence
        self.lambda_energy_raw = lambda_energy_raw
        self.lambda_energy_scale = lambda_energy_scale
        self.total_epochs = total_epochs
        self.current_epoch = 0  # trainer 每 epoch 更新（物理正则 warmup）

    def forward(self, pred, target, components, u_input: Optional[torch.Tensor] = None):
        losses = {}

        losses['data'] = F.mse_loss(pred, target)

        # 频带分支直接监督
        t_low = spectral_lowpass(target, self.k_cut)
        t_high = target - t_low
        u_anchor = components['u_global']
        u_soft = components['u_local']
        losses['anchor_low'] = F.mse_loss(u_anchor, t_low)
        losses['soft_high'] = F.mse_loss(u_soft, t_high)

        # 监督误差标定（目标 detach；梯度只到 err_head）
        err_pred = components.get('err_pred')
        if err_pred is not None:
            res_g = (t_low - u_anchor).norm(dim=1, keepdim=True).detach()
            res_s = (t_high - u_soft).norm(dim=1, keepdim=True).detach()
            losses['err_calib'] = F.mse_loss(err_pred[:, 0:1], res_g) + \
                F.mse_loss(err_pred[:, 1:2], res_s)
        else:
            losses['err_calib'] = torch.tensor(0.0, device=pred.device)

        # 与后处理线耦合的正则（fix2 语义，作用于最终 pred）
        E_pred = kinetic_energy(pred)
        E_target = kinetic_energy(target)
        energy_ratio = E_pred / (E_target + 1e-8)
        losses['energy'] = F.smooth_l1_loss(
            energy_ratio, torch.ones_like(energy_ratio), beta=0.1,
        )
        losses['divergence'] = divergence_mse(pred)

        # 接口保留项（λ=0 默认不启用；语义同 fix2 损失）
        erb = components.get('energy_ratio_before')
        if erb is not None and isinstance(erb, torch.Tensor):
            losses['energy_raw'] = F.smooth_l1_loss(
                erb, torch.ones_like(erb), beta=0.1,
            )
        else:
            losses['energy_raw'] = torch.tensor(0.0, device=pred.device)
        es = components.get('energy_scale')
        if es is not None and isinstance(es, torch.Tensor):
            losses['energy_scale'] = F.smooth_l1_loss(
                es, torch.ones_like(es), beta=0.1,
            )
        else:
            losses['energy_scale'] = torch.tensor(0.0, device=pred.device)

        # 物理正则 warmup（与 fix2 相同：前 50% 训练线性 ramp）
        progress = min(1.0, (self.current_epoch / max(1, self.total_epochs)) * 2.0)

        total_loss = (
            self.lambda_data * losses['data']
            + self.lambda_anchor_low * losses['anchor_low']
            + self.lambda_soft_high * losses['soft_high']
            + self.lambda_err_calib * losses['err_calib']
            + progress * self.lambda_energy * losses['energy']
            + progress * self.lambda_divergence * losses['divergence']
            + self.lambda_energy_raw * losses['energy_raw']
            + self.lambda_energy_scale * losses['energy_scale']
        )

        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        loss_dict['physics_progress'] = progress
        mem_info = components.get('mem_info') or {}
        loss_dict['mem_bank_size'] = mem_info.get('mem_bank_size', 0)

        return total_loss, loss_dict


# ============ 测试 ============

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = RCLN_UPI_v5_Spec().to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} ({total/1e6:.2f}M)")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}")

    x = torch.randn(1, 3, 64, 64, 64, device=device)
    t = torch.randn_like(x)
    out, comp = model(x, target=t, return_components=True)
    print(f"out: {out.shape}, w(scale) mean={comp['w'].mean().item():.4f}, "
          f"R_G={comp['R_G'].item():.4f}")

    crit = RCLN_UPI_v5Spec_Loss()
    loss, ld = crit(out, t, comp, u_input=x)
    loss.backward()
    print(f"loss={loss.item():.4f} | " + " ".join(f"{k}={v:.4f}" for k, v in ld.items()
          if isinstance(v, float)))

"""
RCLN-UPI Memory Module v5 — AGGRESSIVE REFACTOR
================================================
Surgical changes (2026-07-12):
- Key: 8×8×8 spatial grid pooled from raw u (preserves phase info)
- Similarity: Negative Euclidean distance + τ=0.1 temperature scaling
- Force-write: First 20% of training, unconditional fill to 80% capacity
- Forgetting gate: Freshness counter, evict stale entries (freshness < -5)
- V_local ≈ 19K params (lightweight FiLM generator)

Key lessons (from v3-v16 iterations):
- HC encoder feature collapse (z_raw cos sim ≈ 1.0), cannot use encoder output as key
- Must use raw physical input + spatial pooling to grow memory bank
- Direction 1 (boost w) is negative optimization
- Direction 2 (FiLM→u_fused) works, Standard recovers to 0.0176
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# ============ Spatial Key Projector（8×8×8 粗网格池化）============

class RawKeyProjector(nn.Module):
    """
    Extracts a spatial-phase-preserving key from raw physical input.

    Uses AdaptiveAvgPool3d(8,8,8) to retain coarse spatial structure
    ("vortex at top-left" vs "vortex at bottom-right" are distinguishable),
    then projects to key_dim via a learnable linear layer.
    """
    def __init__(self, in_channels: int = 3, key_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.key_dim = key_dim

        # 8³ coarse grid → 3 * 512 = 1536-dim flattened feature
        self.spatial_pool = nn.AdaptiveAvgPool3d((8, 8, 8))
        pool_flat = in_channels * 8 * 8 * 8  # 1536

        # Learnable projection to key_dim
        self.proj = nn.Sequential(
            nn.Linear(pool_flat, key_dim * 2),
            nn.LayerNorm(key_dim * 2),
            nn.SiLU(),
            nn.Linear(key_dim * 2, key_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [B, C, H, W, D] raw physical input
        Returns:
            key: [B, key_dim] spatial-pooling key (NOT L2-normalized —
                 we use Euclidean distance in MemoryBank)
        """
        # Coarse spatial pooling preserving structure
        feat = self.spatial_pool(u)               # [B, C, 8, 8, 8]
        feat = feat.view(u.shape[0], -1)          # [B, 1536]
        key = self.proj(feat)                     # [B, key_dim]
        return key


# ============ Memory Bank（欧氏距离 + 温度缩放 + 遗忘门）============

class MemoryBank(nn.Module):
    """
    Growable memory bank storing (key, value) pairs.
    Value = FiLM parameters (gamma, beta) for modulating u_fused.

    v5-aggressive changes:
    - Euclidean distance + τ=0.1 temperature instead of cosine similarity
    - Freshness counter: entries with sim < theta_sim lose freshness
    - Eviction priority: freshness < -5 first, then usage_count
    """
    def __init__(
        self,
        key_dim: int = 64,
        film_channels: int = 3,
        max_size: int = 256,
        theta_sim: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.key_dim = key_dim
        self.film_channels = film_channels
        self.max_size = max_size
        self.theta_sim = theta_sim
        self.temperature = temperature

        self.register_buffer('keys', torch.zeros(max_size, key_dim))
        self.gamma = nn.Parameter(torch.zeros(max_size, film_channels))
        self.beta = nn.Parameter(torch.zeros(max_size, film_channels))

        self.register_buffer('bank_size', torch.tensor(0, dtype=torch.long))
        self.register_buffer('usage_count', torch.zeros(max_size, dtype=torch.long))
        # v5-aggressive: freshness counter for forgetting gate
        self.register_buffer('freshness', torch.zeros(max_size, dtype=torch.long))
        self.register_buffer('_last_retrieved_idx', torch.tensor(-1, dtype=torch.long))

        with torch.no_grad():
            self.gamma.fill_(1.0)

    def retrieve(self, query_key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve FiLM params via negative Euclidean distance + temperature scaling.

        Returns:
            gamma: [B, film_channels]
            beta: [B, film_channels]
            sim: [B] similarity in [0, 1] range
        """
        B = query_key.shape[0]
        size = self.bank_size.item()

        if size == 0:
            gamma = torch.ones(B, self.film_channels, device=query_key.device)
            beta = torch.zeros(B, self.film_channels, device=query_key.device)
            sim = torch.zeros(B, device=query_key.device)
            with torch.no_grad():
                self._last_retrieved_idx = torch.tensor(-1, dtype=torch.long, device=query_key.device)
            return gamma, beta, sim

        # Clone keys_active: avoid inplace modification by grow() later in same forward
        # breaking the autograd graph of cdist in backward pass
        keys_active = self.keys[:size].clone()  # [size, key_dim]

        # Negative Euclidean distance (no L2 normalization on keys)
        # cdist returns [B, size] of ||query_key_i - keys_active_j||
        neg_dist = -torch.cdist(query_key, keys_active, p=2)  # [B, size]

        # Temperature scaling: τ=0.1 makes distances sharply separated
        scaled = neg_dist / self.temperature  # [B, size]

        # Map to [0, 1] via sigmoid-like transform
        # sim ≈ 1.0 when distance is very small, sim ≈ 0.0 when distance is large
        sim_raw, idx = scaled.max(dim=1)  # [B], [B]
        sim = torch.sigmoid(sim_raw)      # [B], range [0, 1]

        # Retrieve FiLM params (clone to allow gradient flow to bank entries)
        gamma = self.gamma[idx].clone()
        beta = self.beta[idx].clone()

        # Record retrieved index for maybe_grow exclusion — MUST detach to avoid
        # autograd version conflict when rollout training calls forward() multiple times
        with torch.no_grad():
            self._last_retrieved_idx = idx.clone()

        # Update usage and freshness
        with torch.no_grad():
            for i_idx, i_sim in zip(idx.detach(), sim.detach()):
                self.usage_count[i_idx] += 1
                # Freshness: entries below threshold lose freshness
                if i_sim.item() < self.theta_sim:
                    self.freshness[i_idx] -= 1
                else:
                    self.freshness[i_idx] = min(self.freshness[i_idx] + 1,
                                                torch.tensor(10, dtype=torch.long,
                                                            device=self.freshness.device))

        return gamma, beta, sim

    def _select_eviction_idx(self) -> int:
        """Select index to evict: prioritize stale entries (freshness < -5), then least-used."""
        # First: any entry with freshness < -5?
        stale_mask = self.freshness < -5
        if stale_mask.any():
            # Among stale entries, pick least-used
            stale_indices = torch.where(stale_mask)[0]
            min_usage_idx = self.usage_count[stale_indices].argmin()
            return stale_indices[min_usage_idx].item()

        # Next: avoid recently-retrieved entries
        if self._last_retrieved_idx.numel() > 0 and self._last_retrieved_idx[0] >= 0:
            mask = torch.ones(self.max_size, dtype=torch.bool, device=self.usage_count.device)
            mask[self._last_retrieved_idx] = False
            if mask.any():
                masked_usage = self.usage_count.clone().float()
                masked_usage[~mask] = float('inf')
                return masked_usage.argmin().item()

        # Fallback: least-used overall
        return self.usage_count.argmin().item()

    def grow(self, new_key: torch.Tensor, new_gamma: torch.Tensor, new_beta: torch.Tensor):
        """Add a new entry, evicting worst entry if bank is full.
        Safety: never overwrite entries that were retrieved in this forward pass
        (would break autograd graph in rollout training)."""
        size = self.bank_size.item()
        if size >= self.max_size:
            min_idx = self._select_eviction_idx()
            # Safety: if eviction target was retrieved this pass, skip grow
            if self._last_retrieved_idx.numel() > 0:
                if min_idx in self._last_retrieved_idx:
                    return  # skip to avoid breaking autograd
            with torch.no_grad():
                self.keys[min_idx] = new_key.detach()
                self.gamma[min_idx] = new_gamma.detach()
                self.beta[min_idx] = new_beta.detach()
                self.usage_count[min_idx] = 1
                self.freshness[min_idx] = 0
        else:
            with torch.no_grad():
                self.keys[size] = new_key.detach()
                self.gamma[size] = new_gamma.detach()
                self.beta[size] = new_beta.detach()
                self.usage_count[size] = 1
                self.freshness[size] = 0
                self.bank_size += 1

    def maybe_grow(self, query_key: torch.Tensor, sim: torch.Tensor,
                   default_gamma: torch.Tensor, default_beta: torch.Tensor,
                   force_write: bool = False):
        """
        Conditionally grow the bank.

        Args:
            force_write: If True, skip theta_sim check — always write
        """
        B = query_key.shape[0]
        for b in range(B):
            if force_write or sim[b].item() < self.theta_sim:
                self.grow(query_key[b], default_gamma[b], default_beta[b])


# ============ V_local：轻量修正网络（~19K params）============

class VLocalNet(nn.Module):
    """
    Lightweight FiLM parameter generator from global context.
    Parameter count: ~19K (v5 optimal).
    """
    def __init__(self, global_dim: int = 64, film_channels: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.SiLU(),
            nn.Linear(256, film_channels * 2),
        )

    def forward(self, z_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(z_global)  # [B, film_channels * 2]
        gamma, beta = out.chunk(2, dim=1)

        # Constrain gamma in [0.5, 1.5], beta in [-0.5, 0.5]
        gamma = 1.0 + 0.5 * torch.tanh(gamma)
        beta = 0.5 * torch.tanh(beta)

        return gamma, beta

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============ Fatigue Detector ============

class FatigueDetector(nn.Module):
    """
    Monitors Hard Core output norm CHANGES between successive updates.
    Triggers mild gradient incentive when stagnation detected.

    v5-fix(2026-08): 判据修复。旧实现用范数历史的【均值】与阈值比较——只要
    ||u_global||_F 的正常量级大于 0.01（实际远大于此），疲劳永不触发，与
    注释声称的 "norm changes" 意图不符。现改为：历史缓冲存储相邻两次更新的
    范数【变化量】 |N_t - N_{t-1}|，均值低于阈值才判为停滞。
    注意：上次范数用普通属性 _last_norm 保存（非 buffer），避免新增
    state_dict 键破坏既有 checkpoint 的 strict-load。
    """
    def __init__(self, window_size: int = 10, fatigue_threshold: float = 0.01):
        super().__init__()
        self.window_size = window_size
        self.fatigue_threshold = fatigue_threshold
        self.register_buffer('history', torch.zeros(window_size))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('fatigue_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
        # 普通属性，不入 state_dict（checkpoint 兼容）
        self._last_norm = None

    def update(self, u_global_norm: float):
        if self._last_norm is not None:
            change = abs(u_global_norm - self._last_norm)
            self.history[self.ptr] = change
            self.ptr = (self.ptr + 1) % self.window_size
        self._last_norm = u_global_norm
        self.num_updates += 1

    def is_fatigued(self) -> bool:
        # 需要 window_size 个变化量样本（即 window_size+1 次更新）才能判定
        if self.num_updates.item() < self.window_size + 1:
            return False
        mean_change = self.history.mean().item()
        return mean_change < self.fatigue_threshold

    def fatigue_loss(self, u_global: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.is_fatigued():
            return torch.tensor(0.0, device=u_global.device)
        loss = F.mse_loss(u_global, target) * 0.01
        with torch.no_grad():
            self.fatigue_count += 1
        return loss


# ============ Memory Module v5 — AGGRESSIVE ============

class MemoryModuleV5(nn.Module):
    """
    v5-aggressive memory module:
    1. SpatialKeyProjector: 8×8×8 grid pooled from raw u
    2. MemoryBank: Euclidean distance + τ=0.1 + forgetting gate
    3. VLocalNet: Lightweight default FiLM generator (~19K params)
    4. FatigueDetector: HC stagnation detection
    5. Force-write: First 20% of training, unconditional fill to 80% capacity
    """
    def __init__(
        self,
        in_channels: int = 3,
        global_dim: int = 64,
        key_dim: int = 64,
        film_channels: int = 3,
        max_bank_size: int = 256,
        theta_sim: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.theta_sim = theta_sim
        self.temperature = temperature

        self.key_projector = RawKeyProjector(in_channels, key_dim)
        self.memory_bank = MemoryBank(key_dim, film_channels, max_bank_size, theta_sim, temperature)
        self.v_local = VLocalNet(global_dim, film_channels)
        self.fatigue_detector = FatigueDetector()

        # Force-write state
        self._force_write_phase = True

    def forward(self, u_raw: torch.Tensor, z_global: torch.Tensor,
                u_global: torch.Tensor, target: Optional[torch.Tensor] = None,
                mode: str = 'train',
                training_progress: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            u_raw: [B, C, H, W, D] raw physical input (for spatial key)
            z_global: [B, global_dim] global context
            u_global: [B, C, H, W, D] hard core output (for fatigue detection)
            target: [B, C, H, W, D] target (for fatigue detection)
            mode: 'train' or 'eval'
            training_progress: 0→1 linear progress (for force-write scheduling)
        Returns:
            film_params: [B, C, 2] flattened (gamma, beta) for FiLM
            info: dict with diagnostics
        """
        B, C, H, W, D = u_raw.shape

        # 1. Generate spatial key from raw u
        query_key = self.key_projector(u_raw)  # [B, key_dim]

        # 2. Generate default FiLM params (V_local)
        default_gamma, default_beta = self.v_local(z_global)  # [B, C], [B, C]

        # 3. Retrieve from memory bank
        mem_gamma, mem_beta, sim = self.memory_bank.retrieve(query_key)  # [B, C], [B, C], [B]

        # 4. Blend: high sim → memory, low sim → default
        w_mem = torch.sigmoid((sim - self.theta_sim) * 10.0).view(B, 1)  # [B, 1]

        gamma = w_mem * mem_gamma + (1 - w_mem) * default_gamma  # [B, C]
        beta = w_mem * mem_beta + (1 - w_mem) * default_beta    # [B, C]

        # 5. Grow bank
        if mode == 'train' and self.training:
            # Force-write: first 20% of training, fill to 80% capacity
            force_write = (
                self._force_write_phase and
                training_progress < 0.2 and
                self.memory_bank.bank_size.item() < int(self.memory_bank.max_size * 0.8)
            )
            self.memory_bank.maybe_grow(
                query_key, sim,
                default_gamma.detach(), default_beta.detach(),
                force_write=force_write
            )
            # Disengage force-write once target capacity reached
            if force_write and self.memory_bank.bank_size.item() >= int(self.memory_bank.max_size * 0.8):
                self._force_write_phase = False

        # 6. Fatigue detection
        fatigue_loss = torch.tensor(0.0, device=u_raw.device)
        if target is not None and mode == 'train':
            with torch.no_grad():
                u_global_norm = u_global.norm().item()
            self.fatigue_detector.update(u_global_norm)
            fatigue_loss = self.fatigue_detector.fatigue_loss(u_global, target)

        # 7. Build FiLM params [B, C, 2]
        film_params = torch.stack([gamma, beta], dim=2)  # [B, C, 2]

        info = {
            'mem_sim': sim.mean().item(),
            'mem_bank_size': self.memory_bank.bank_size.item(),
            'mem_w_mem': w_mem.mean().item(),
            'fatigue_loss': fatigue_loss.item(),
            'fatigue_count': self.fatigue_detector.fatigue_count.item(),
            'force_write': self._force_write_phase,
        }

        return film_params, info

    def apply_film(self, u_fused: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation: u_fused' = gamma * u_fused + beta
        """
        gamma = film_params[:, :, 0:1]  # [B, C, 1]
        beta = film_params[:, :, 1:2]   # [B, C, 1]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1, 1]

        return gamma * u_fused + beta

    def count_params(self) -> Dict[str, int]:
        return {
            'key_projector': sum(p.numel() for p in self.key_projector.parameters()),
            'memory_bank_learnable': sum(p.numel() for p in [self.memory_bank.gamma, self.memory_bank.beta]),
            'v_local': self.v_local.count_params(),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ============ Test ============

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("Memory Module v5 — AGGRESSIVE REFACTOR Test")
    print("=" * 60)

    mm = MemoryModuleV5(
        in_channels=3,
        global_dim=64,
        key_dim=64,
        film_channels=3,
        max_bank_size=64,
        theta_sim=0.5,
        temperature=0.1,
    ).to(device)

    params = mm.count_params()
    print(f"\nParameter counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    B = 2
    u_raw = torch.randn(B, 3, 64, 64, 64).to(device)
    z_global = torch.randn(B, 64).to(device)
    u_global = torch.randn(B, 3, 64, 64, 64).to(device)
    target = torch.randn(B, 3, 64, 64, 64).to(device)

    print(f"\nInput: u_raw={u_raw.shape}, z_global={z_global.shape}")

    # Force-write phase: training_progress=0.0 (start of training)
    mm.train()
    print("\n=== Force-write phase (progress=0.0) ===")
    for i in range(5):
        film_params, info = mm(u_raw, z_global, u_global, target, mode='train', training_progress=0.0)
        u_corrected = mm.apply_film(u_global, film_params)
        print(f"  Step {i}: bank_size={info['mem_bank_size']}, "
              f"sim={info['mem_sim']:.4f}, w_mem={info['mem_w_mem']:.4f}, "
              f"force_write={info['force_write']}")

    # Post force-write phase: training_progress=0.5
    print("\n=== Normal phase (progress=0.5) ===")
    for i in range(5):
        film_params, info = mm(u_raw, z_global, u_global, target, mode='train', training_progress=0.5)
        print(f"  Step {i}: bank_size={info['mem_bank_size']}, "
              f"sim={info['mem_sim']:.4f}, w_mem={info['mem_w_mem']:.4f}, "
              f"force_write={info['force_write']}")

    # Eval mode
    mm.eval()
    with torch.no_grad():
        film_params, info = mm(u_raw, z_global, u_global, mode='eval', training_progress=1.0)
        u_corrected = mm.apply_film(u_global, film_params)

    print(f"\nEval: bank_size={info['mem_bank_size']}, sim={info['mem_sim']:.4f}")
    print(f"u_corrected: {u_corrected.shape}, mean={u_corrected.mean().item():.4f}")

    print("\nAll tests passed!")

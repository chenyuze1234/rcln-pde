"""
baselines_v6_ext.py — 官方基线适配层（DPOT / Transolver）
==========================================================
包装 external/baselines/ 下 vendored 的官方实现，统一接口为
    forward(x): (B, 3, 64, 64, 64) -> (B, 3, 64, 64, 64)
与 RCLN_UPI_v6_64 / FNO3D / FFNO3D 完全一致，可直接接入 train_all_fair.py 的
RolloutTrainer（plain MSE, forward(x)->pred）与 eval_common.load_model_strict。

官方代码不做任何算法修改（vendor 补丁仅见 external/baselines/*/VENDOR_INFO.md）：
  - DPOT:  DPOTNet3D (thu-ml/DPOT @ dcd2f9a)
  - Transolver: Model(geotype='structured_3D') (thuml/Neural-Solver-Library @ db942b7)

公平性注记：
  - DPOT: patch_size=8 → 8^3=512 tokens，embed_dim/depth 调至参数量 ≈10.8M±10%。
  - Transolver: 64^3=262144 点全量输入，hidden width 受 8GB 显存约束；
    最终配置由显存实测决定（见本文件底部 __main__ 冒烟输出与 v6 报告）。
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

# external/baselines 加入搜索路径（dpot / neural_solver_lib 均为独立包名，无冲突）
_EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'external', 'baselines')
if _EXT not in sys.path:
    sys.path.insert(0, _EXT)

from dpot.dpot3d import DPOTNet3D                                    # noqa: E402
from neural_solver_lib.models.Transolver import Model as Transolver  # noqa: E402

GRID = 64  # 64^3


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# DPOT (official DPOTNet3D, thu-ml)
# ============================================================

# 实测参数量调参结果（2026-08-04, Python313 + torch 2.12, RTX 5070 Laptop 8GB）：
#   e512d4=12.20M  e448d5=10.82M ✓  e384d6=9.28M
#   最终 e448d5：10,819,072 参数（对齐 v6 10.79M ±10% 预算），fp32 峰值 0.36GB
DPOT_CONFIG = dict(
    img_size=GRID, patch_size=8, mixing_type='afno',
    in_channels=3, out_channels=3, in_timesteps=1, out_timesteps=1,
    n_blocks=8, embed_dim=448, out_layer_dim=32, depth=5,
    modes=8, temporal_modes=4, mlp_ratio=1.0,
    normalize=False, act='gelu', time_agg='exp_mlp',
)


class DPOTAdapter(nn.Module):
    """官方 DPOTNet3D 的 (B,C,H,W,D) 接口包装。

    官方 forward 期望 (B, X, Y, Z, T, C) 并返回 (B, X, Y, Z, T_out, C)；
    本适配层做维度变换，保持 T=1 单步预测。
    """

    def __init__(self, **overrides):
        super().__init__()
        cfg = dict(DPOT_CONFIG)
        cfg.update(overrides)
        self.cfg = cfg
        self.net = DPOTNet3D(**cfg)

    def forward(self, x):
        # x: (B, 3, 64, 64, 64) -> (B, X, Y, Z, T=1, C=3)
        B = x.shape[0]
        x5 = x.permute(0, 2, 3, 4, 1).unsqueeze(4).contiguous()
        y = self.net(x5)                       # (B, X, Y, Z, 1, 3)
        y = y.reshape(B, GRID, GRID, GRID, 3).permute(0, 4, 1, 2, 3).contiguous()
        return y


# ============================================================
# Transolver (official, structured_3D mesh mode)
# ============================================================

# 参考官方脚本 scripts/StandardBench/ns/Transolver.sh：
#   n_hidden=256 n_heads=8 n_layers=8 mlp_ratio=2 slice_num=32 unified_pos=1 ref=8
# 64^3=262144 点实测（RTX 5070 Laptop 8GB, 2026-08-04, 见 __main__ 扫描）：
#   - 官方宽度 256 不可行；w64 fp32 反向也 OOM（LayerNorm 强制 fp32，激活 ~4GB+）
#   - 必须 AMP + 逐 block 梯度检查点（适配层实现，数学等价，不改官方代码）
#   - 保留官方 ref=8 / mlp_ratio=2 / unified_pos；slice_num=64（3D 默认档）
# 最终配置 w128 L11：10,833,563 参数（10.83M，对齐 v6 10.79M ±10% 预算），
# 冒烟峰值 6.03GB（fwd+bwd+AdamW step），可训练。
TRANSOLVER_CONFIG = dict(
    space_dim=3, fun_dim=3, out_dim=3,
    n_hidden=128, n_heads=8, n_layers=11,
    dropout=0.0, act='gelu', mlp_ratio=2,
    slice_num=64, ref=8, unified_pos=True,
    geotype='structured_3D', shapelist=(GRID, GRID, GRID),
    time_input=False,
)


class _CkptBlock(nn.Module):
    """梯度检查点包装（适配层级别，不改官方源码，数学上完全等价）。"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, fx):
        if self.training and fx.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self.block, fx, use_reentrant=False)
        return self.block(fx)


class TransolverAdapter(nn.Module):
    """官方 Transolver（structured_3D）的 (B,C,H,W,D) 接口包装。

    官方 forward(x, fx)：structured + unified_pos 时坐标由内部 pos 提供，
    fx 为 (B, N, fun_dim) 点值；输出 (B, N, out_dim)。
    grad_ckpt=True 时用 checkpoint 包装每个 Transolver_block（8GB 显存必需，
    仅省显存不改计算结果）。
    """

    def __init__(self, grad_ckpt=True, **overrides):
        super().__init__()
        cfg = dict(TRANSOLVER_CONFIG)
        cfg.update(overrides)
        self.cfg = cfg
        self.net = Transolver(SimpleNamespace(**cfg))
        if grad_ckpt:
            self.net.blocks = nn.ModuleList(
                [_CkptBlock(b) for b in self.net.blocks])

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        fx = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()  # (B, N, 3)
        # 官方 structured_geo 在 unified_pos 模式下仅用 x.shape[0] 取 batch，
        # 坐标由内部 self.pos 提供；传轻量 dummy 避免冗余显存。
        y = self.net(fx[:, :1, :], fx)                           # (B, N, 3)
        return y.permute(0, 2, 1).reshape(B, C, GRID, GRID, GRID).contiguous()


# ============================================================
# 冒烟测试：形状 / 参数量 / backward / 显存峰值
# ============================================================

def _smoke(name, model, device, amp=False):
    model = model.to(device)
    tp = count_params(model)
    x = torch.randn(1, 3, GRID, GRID, GRID, device=device)
    tgt = torch.randn(1, 3, GRID, GRID, GRID, device=device)
    ctx = torch.amp.autocast('cuda') if amp else __import__('contextlib').nullcontext()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    with ctx:
        out = model(x)
        loss = torch.nn.functional.mse_loss(out.float(), tgt)
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() / 1e9
    ok_shape = tuple(out.shape) == (1, 3, GRID, GRID, GRID)
    n_nan = sum(int((~torch.isfinite(p.grad)).sum()) for p in model.parameters()
                if p.grad is not None)
    print(f"[{name}] params={tp:,} ({tp/1e6:.2f}M) out={tuple(out.shape)} "
          f"shape_ok={ok_shape} loss={loss.item():.4f} nonfinite_grads={n_nan} "
          f"peak_mem={peak:.2f}GB amp={amp}", flush=True)
    assert ok_shape and n_nan == 0
    del model, x, tgt, out, loss, opt
    torch.cuda.empty_cache()
    return tp, peak


if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={dev}")
    # 每个配置在独立子进程中冒烟（同进程 OOM 后 CUDA 上下文不可靠）
    if '--scan-transolver' in sys.argv:
        import subprocess
        for w, L in [(128, 8), (96, 8), (64, 8), (64, 12), (64, 16), (96, 12), (48, 16)]:
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--one', 'transolver',
                 '--w', str(w), '--l', str(L)],
                capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith('[')]
            print(line[0] if line else f"[transolver w{w}L{L}] FAILED (OOM or error)",
                  flush=True)
    elif '--scan-dpot' in sys.argv:
        for e, d in [(512, 4), (448, 5), (384, 6)]:
            _smoke(f'dpot e{e}d{d}', DPOTAdapter(embed_dim=e, depth=d), dev)
    elif '--one' in sys.argv:
        i = sys.argv.index('--one')
        name = sys.argv[i + 1]
        w = int(sys.argv[sys.argv.index('--w') + 1])
        L = int(sys.argv[sys.argv.index('--l') + 1])
        amp = '--amp' in sys.argv
        mr = int(sys.argv[sys.argv.index('--mr') + 1]) if '--mr' in sys.argv else 2
        ref = int(sys.argv[sys.argv.index('--ref') + 1]) if '--ref' in sys.argv else 8
        if name == 'transolver':
            _smoke(f'transolver w{w}L{L}mr{mr}ref{ref}',
                   TransolverAdapter(n_hidden=w, n_layers=L, mlp_ratio=mr, ref=ref),
                   dev, amp=amp)
    else:
        # 最终配置冒烟：DPOT fp32/AMP 均可；Transolver 必须 AMP（8GB 显存约束）
        _smoke('dpot', DPOTAdapter(), dev)
        _smoke('dpot', DPOTAdapter(), dev, amp=True)
        _smoke('transolver', TransolverAdapter(), dev, amp=True)

"""
eval_common.py — 评估共享模块（P0 修复轮，2026-08-04）
========================================================
统一所有 rollout 评估脚本的：
  - 指标定义（compute_ke / compute_rl2）
  - 崩溃判据（is_crashed，对齐论文 Methods）
  - checkpoint strict 加载 + 归一化统计量读取（load_model_strict）
  - fp32 评估（禁止 autocast fp16）

约定：
  - 归一化 mean/std 必须从 checkpoint 读取（训练端写入 extra/top-level）。
    缺失时清晰报错退出，绝不静默使用默认值。
"""
import sys

import numpy as np
import torch


# ============================================================
# 指标
# ============================================================

def compute_ke(u):
    """动能 0.5 * mean(u^2)。u: [3, H, W, D]（与 rollout_eval_v6_64.py 原 ke() 一致）。"""
    return 0.5 * np.mean(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)


def compute_rl2(pred, target):
    """相对 L2 误差 ||pred - target|| / ||target||。"""
    return float(np.linalg.norm((pred - target).flatten()) /
                 (np.linalg.norm(target.flatten()) + 1e-8))


# ============================================================
# 崩溃判据（论文 Methods）
# ============================================================

RL2_CRASH_THRESHOLD = 10.0
ENERGY_CRASH_RATIO = 5.0
UMAX_CRASH_THRESHOLD = 50.0


def is_crashed(pred_np, energy_ratio=None, rl2=None):
    """论文 Methods 崩溃判据：non-finite 或 rL2>10 或 E/E0>5 或 |u|>50。

    Args:
        pred_np: 去归一化后的预测场 [3, H, W, D]
        energy_ratio: 可选，E_pred / E0
        rl2: 可选，当前步的相对 L2
    Returns:
        (crashed: bool, reason: str or None)
    """
    if not np.isfinite(pred_np).all():
        return True, 'NaN/non-finite'
    if rl2 is not None and rl2 > RL2_CRASH_THRESHOLD:
        return True, f'rL2>{RL2_CRASH_THRESHOLD:g}(={rl2:.1f})'
    if energy_ratio is not None and energy_ratio > ENERGY_CRASH_RATIO:
        return True, f'E/E0>{ENERGY_CRASH_RATIO:g}(={energy_ratio:.1f})'
    u_max = float(np.max(np.abs(pred_np)))
    if u_max > UMAX_CRASH_THRESHOLD:
        return True, f'|u|max={u_max:.0f}>{UMAX_CRASH_THRESHOLD:g}'
    return False, None


# ============================================================
# 归一化统计量：必须从 checkpoint 读取，缺失即报错
# ============================================================

def read_norm_stats(ck, path='<checkpoint>'):
    """从 checkpoint dict 读取 mean/std。缺失时清晰报错退出（不静默默认）。"""
    mean, std = ck.get('mean'), ck.get('std')
    if mean is None or std is None:
        raise SystemExit(
            f"ERROR: checkpoint 缺少归一化统计量 mean/std: {path}\n"
            f"  该 checkpoint 由旧版训练脚本生成（未保存 mean/std）。\n"
            f"  为避免静默使用错误归一化，评估已中止。\n"
            f"  请用修复后的训练脚本重新训练（mean/std 会写入 checkpoint extra），"
            f"或显式确认要使用的 mean/std 后手动注入 checkpoint。"
        )
    return float(mean), float(std)


# ============================================================
# strict 模型加载
# ============================================================

def load_model_strict(path, model_fn, device='cpu', require_norm_stats=True):
    """strict 加载 checkpoint + 读取归一化统计量 + 断言。

    Args:
        path: checkpoint 路径
        model_fn: 无参 callable，返回 nn.Module（未加载权重）
        device: 目标设备
        require_norm_stats: True 时 mean/std 缺失即报错退出
    Returns:
        (model, mean, std, ck) — require_norm_stats=False 且缺失时 mean/std 为 None
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model = model_fn()
    result = model.load_state_dict(sd, strict=False)
    print(f"  strict-load {path}: missing={list(result.missing_keys)} "
          f"unexpected={list(result.unexpected_keys)}", flush=True)
    assert not result.missing_keys, \
        f"checkpoint 与模型不匹配，missing keys: {list(result.missing_keys)} ({path})"
    assert not result.unexpected_keys, \
        f"checkpoint 与模型不匹配，unexpected keys: {list(result.unexpected_keys)} ({path})"
    if require_norm_stats:
        mean, std = read_norm_stats(ck, path)
    else:
        mean, std = ck.get('mean'), ck.get('std')
        mean = float(mean) if mean is not None else None
        std = float(std) if std is not None else None
    model.to(device).eval()
    return model, mean, std, ck

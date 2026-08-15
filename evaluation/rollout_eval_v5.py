"""
RCLN-UPI v5 统一 Rollout 评估脚本
==================================
支持：
- 多个 rollout 长度 K
- 多个初始条件（num_ic）
- 噪声鲁棒性测试
- OOD 泛化测试
- 自动计算 rel_L2, energy_ratio, crash_rate, crash_step
- 保存 JSON 结果供论文表格使用
"""

import torch
import torch.nn.functional as F
import h5py
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.rcln_upi_v5 import RCLN_UPI_v5, AblationMode


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_energy_projection: bool = True,
    ablation_mode: AblationMode = 'full',
    base_channels: int = 32,
    latent_dim: int = 64,
    use_physical_radius: bool = False,
    use_internal_upi: bool = False,
    encoder_type: str = 'cnn',
    transformer_num_blocks: int = 4,
    transformer_num_heads: int = 8,
    use_calibrated_gate: bool = True,
    calib_beta: float = 1.0,
    calib_scale: float = 1.54,
):
    """加载模型，兼容 plain state_dict 和 checkpoint dict（v5-fix2 新默认架构）"""
    model = RCLN_UPI_v5(
        in_channels=3,
        base_channels=base_channels,
        latent_dim=latent_dim,
        decoder_freq=4,
        memory_key_dim=64,
        memory_max_size=256,
        memory_theta_sim=0.5,
        use_energy_projection=use_energy_projection,
        energy_target_mode='input',
        ablation_mode=ablation_mode,
        use_physical_radius=use_physical_radius,
        use_internal_upi=use_internal_upi,
        encoder_type=encoder_type,
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        use_calibrated_gate=use_calibrated_gate,
        calib_beta=calib_beta,
        calib_scale=calib_scale,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    ms = dict(model.state_dict())
    for k, v in state.items():
        if k in ms and isinstance(v, torch.Tensor) and isinstance(ms[k], torch.Tensor) and ms[k].shape == v.shape:
            ms[k] = v
    model.load_state_dict(ms, strict=True)
    model.eval()
    return model


def compute_metrics(pred: np.ndarray, target: np.ndarray, initial: np.ndarray) -> Dict[str, float]:
    """计算单步 rollout 指标"""
    pred_f = pred.flatten()
    target_f = target.flatten()
    initial_f = initial.flatten()

    mse = float(np.mean((pred - target) ** 2))
    rel_l2 = float(np.linalg.norm(pred_f - target_f) / (np.linalg.norm(target_f) + 1e-8))

    energy_pred = float(np.sum(pred ** 2))
    energy_initial = float(np.sum(initial ** 2))
    energy_target = float(np.sum(target ** 2))
    energy_ratio = energy_pred / (energy_initial + 1e-8)
    energy_ratio_vs_target = energy_pred / (energy_target + 1e-8)

    max_abs = float(np.abs(pred).max())
    has_nan = bool(np.isnan(pred).any())
    has_inf = bool(np.isinf(pred).any())

    return {
        'mse': mse,
        'rel_l2': rel_l2,
        'energy_ratio': energy_ratio,
        'energy_ratio_vs_target': energy_ratio_vs_target,
        'max_abs': max_abs,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


def detect_crash(metrics_history: List[Dict],
                 energy_threshold: float = 5.0,
                 rel_l2_threshold: float = 5.0,
                 max_abs_threshold: float = 50.0,
                 mse_threshold: float = 1.0) -> int:
    """检测崩溃时间步，-1 表示未崩溃"""
    for i, m in enumerate(metrics_history):
        if m['has_nan'] or m['has_inf']:
            return i + 1
        if m['energy_ratio'] > energy_threshold:
            return i + 1
        if m['rel_l2'] > rel_l2_threshold:
            return i + 1
        if m['max_abs'] > max_abs_threshold:
            return i + 1
        if m['mse'] > mse_threshold:
            return i + 1
    return -1


def rollout_one(
    model: RCLN_UPI_v5,
    initial: torch.Tensor,
    targets: torch.Tensor,
    mean: float,
    std: float,
    device: torch.device,
    noise_level: float = 0.0,
) -> Tuple[List[Dict], int]:
    """对单个初始条件做 rollout，返回指标历史和崩溃步"""
    model.eval()
    metrics_history = []

    initial_np = initial.cpu().numpy()[0]
    u = initial.clone()

    if noise_level > 0.0:
        u = u + noise_level * std * torch.randn_like(u)

    with torch.no_grad():
        for step in range(len(targets)):
            pred = model(u, target=None, return_components=False, ablation_mode=model.ablation_mode)
            pred_np = pred.cpu().numpy()[0]
            target_np = targets[step].cpu().numpy()

            metrics = compute_metrics(pred_np, target_np, initial_np)
            metrics['step'] = step + 1
            metrics_history.append(metrics)

            if metrics['has_nan'] or metrics['has_inf']:
                break

            u = pred

    crash_step = detect_crash(metrics_history)
    return metrics_history, crash_step


def aggregate_metrics(metrics_list: List[List[Dict]], K: int) -> Dict[str, float]:
    """对多个 IC 的 rollout 结果做聚合"""
    rel_l2_at_K = []
    energy_ratio_at_K = []
    crash_flags = []
    crash_steps = []

    for m_hist in metrics_list:
        if len(m_hist) >= K:
            rel_l2_at_K.append(m_hist[K - 1]['rel_l2'])
            energy_ratio_at_K.append(m_hist[K - 1]['energy_ratio'])
        else:
            # 在到达 K 之前崩溃
            crash_flags.append(1)
            crash_steps.append(len(m_hist))
            continue

        crash = detect_crash(m_hist[:K])
        if crash > 0:
            crash_flags.append(1)
            crash_steps.append(crash)
        else:
            crash_flags.append(0)
            crash_steps.append(K)

    return {
        'rel_l2_mean': float(np.mean(rel_l2_at_K)) if rel_l2_at_K else float('inf'),
        'rel_l2_std': float(np.std(rel_l2_at_K)) if rel_l2_at_K else 0.0,
        'energy_ratio_mean': float(np.mean(energy_ratio_at_K)) if energy_ratio_at_K else float('inf'),
        'energy_ratio_std': float(np.std(energy_ratio_at_K)) if energy_ratio_at_K else 0.0,
        'crash_rate': float(np.mean(crash_flags)),
        'crash_step_mean': float(np.mean(crash_steps)) if crash_steps else float(K),
        'n_ic': len(metrics_list),
    }


def main():
    parser = argparse.ArgumentParser(description='RCLN-UPI v5 Unified Rollout Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--K', type=int, nargs='+', default=[50, 100, 200, 500],
                        help='rollout 长度列表')
    parser.add_argument('--num_ic', type=int, default=10,
                        help='初始条件数量（从 fields 前 num_ic 帧选取）')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='输入噪声水平（相对 std）')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'no_upi', 'no_stim', 'no_film', 'hc_only', 'no_energy_proj', 'fixed_rg'])
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--use_physical_radius', action='store_true')
    parser.add_argument('--use_internal_upi', action='store_true')
    parser.add_argument('--encoder_type', type=str, default='cnn',
                        choices=['cnn', 'transformer'],
                        help='编码器类型: cnn=纯卷积, transformer=CNN+全局自注意力')
    parser.add_argument('--transformer_num_blocks', type=int, default=4)
    parser.add_argument('--transformer_num_heads', type=int, default=8)
    parser.add_argument('--no_calibrated_gate', action='store_true',
                        help='关闭校准门（v5-fix2 默认开启；关闭后 w≡1 软壳直通）')
    parser.add_argument('--calib_beta', type=float, default=1.0,
                        help='校准门温度（1=逆误差 2=逆方差）')
    parser.add_argument('--calib_scale', type=float, default=1.54,
                        help='ê 的事后标定系数（R_eff = ê_g × calib_scale）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.h5_path}")
    print(f"K values: {args.K}, num_ic: {args.num_ic}, noise: {args.noise_level}")

    # 加载数据
    with h5py.File(args.h5_path, 'r') as f:
        if 'fields' in f:
            fields = f['fields'][()]
        else:
            raise KeyError(f"No 'fields' in {args.h5_path}")
        train_inputs = f['train_inputs'][()] if 'train_inputs' in f else fields[:-1]

    mean = float(np.mean(train_inputs))
    std = float(np.std(train_inputs)) + 1e-8
    print(f"Fields: {fields.shape}, mean={mean:.6f}, std={std:.6f}")

    # 加载模型
    model = load_model(
        args.checkpoint,
        device,
        ablation_mode=args.ablation_mode,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        use_physical_radius=args.use_physical_radius,
        use_internal_upi=args.use_internal_upi,
        encoder_type=args.encoder_type,
        transformer_num_blocks=args.transformer_num_blocks,
        transformer_num_heads=args.transformer_num_heads,
        use_calibrated_gate=not args.no_calibrated_gate,
        calib_beta=args.calib_beta,
        calib_scale=args.calib_scale,
    )

    max_K = max(args.K)
    num_ic = min(args.num_ic, len(fields) - max_K - 1)
    print(f"Using {num_ic} initial conditions")

    # 对每个 IC 做最长 rollout
    all_metrics = []
    for ic_idx in range(num_ic):
        initial = torch.from_numpy((fields[ic_idx] - mean) / std).float().unsqueeze(0).to(device)
        targets = torch.from_numpy((fields[ic_idx + 1:ic_idx + 1 + max_K] - mean) / std).float().to(device)
        m_hist, crash = rollout_one(model, initial, targets, mean, std, device, noise_level=args.noise_level)
        all_metrics.append(m_hist)
        print(f"  IC {ic_idx + 1}/{num_ic}: crash_step={crash if crash > 0 else f'>{max_K}'}")

    # 聚合结果
    results = {
        'checkpoint': args.checkpoint,
        'h5_path': args.h5_path,
        'ablation_mode': args.ablation_mode,
        'noise_level': args.noise_level,
        'num_ic': num_ic,
        'max_K': max_K,
        'by_K': {},
    }

    print("\n" + "=" * 70)
    print(f"{'K':<8} {'rel_L2':<12} {'E_ratio':<12} {'crash_rate':<12} {'crash_step':<12}")
    print("=" * 70)
    for K in sorted(args.K):
        agg = aggregate_metrics(all_metrics, K)
        results['by_K'][str(K)] = agg
        print(
            f"{K:<8} {agg['rel_l2_mean']:.4f}±{agg['rel_l2_std']:.4f}  "
            f"{agg['energy_ratio_mean']:.4f}±{agg['energy_ratio_std']:.4f}  "
            f"{agg['crash_rate']:.2f}        {agg['crash_step_mean']:.1f}"
        )
    print("=" * 70)

    # 保存结果
    out_path = args.output
    if out_path is None:
        out_dir = Path('results/rollout_eval')
        out_dir.mkdir(parents=True, exist_ok=True)
        h5_name = Path(args.h5_path).stem
        cp_name = Path(args.checkpoint).stem
        out_path = out_dir / f"{h5_name}_{cp_name}_noise{args.noise_level}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

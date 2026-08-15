#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_suite_eval.py — Nature Communications 冲刺版任务书统一评估套件

对全部在手模型按任务书 Table 1 / Fig.3-6 / Go-No-Go 要求的口径统一评估：
  模型: fix2, fix2_hc, v2, v2gate70, v2gate90, v3
  指标: rL2 / E(t) / 耗散 eps(t) / enstrophy / E(k) 谱 / divergence(mean,p95,max)
        / survival(崩溃率,首崩步) / 推理成本(单步 ms, 参数量)
        / U1-U2 UPI 诊断(spread Pearson + top-10% AUROC, 信号=|u_local|)
  协议: fair 口径(同 rollout_eval_v5), 10 IC, K=200, 无 mse>1 误判判据
  输出: results/paper_experiments/nc_suite_k200.json

用法:
  python -u nc_suite_eval.py --K 200 --num_ic 10 \
      --output results/paper_experiments/nc_suite_k200.json
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from models.rcln_upi_v5_spec import RCLN_UPI_v5_Spec
from rollout_eval_v5 import load_model as load_fix2

FIX2_CKPT = 'checkpoints/v5_fix2/v5_best.pt'
FIX2SOFT_CKPT = 'checkpoints/v5_fix2_soft/v5_best.pt'
V2_CKPT = 'checkpoints/v5_spec_v2add_e13/v5_best.pt'
V3_CKPT = 'checkpoints/v5_spec/v5_best.pt'

SPECTRA_STEPS = (1, 10, 50, 100, 200)


# ---------------- 物理统计工具（物理单位场 u: (3,H,W,D)） ----------------

def div_field(u):
    """中心差分散度场"""
    dudx = (np.roll(u[0], -1, 0) - np.roll(u[0], 1, 0)) / 2
    dvdy = (np.roll(u[1], -1, 1) - np.roll(u[1], 1, 1)) / 2
    dwdz = (np.roll(u[2], -1, 2) - np.roll(u[2], 1, 2)) / 2
    return dudx + dvdy + dwdz


def div_stats(u):
    d = div_field(u)
    return float(np.abs(d).mean()), float(np.quantile(np.abs(d), 0.95)), float(np.abs(d).max())


def enstrophy(u):
    """0.5*mean(|curl u|^2)，中心差分"""
    def d(f, ax):
        return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / 2
    wx = d(u[2], 1) - d(u[1], 2)
    wy = d(u[0], 2) - d(u[2], 0)
    wz = d(u[1], 0) - d(u[0], 1)
    return 0.5 * float(np.mean(wx ** 2 + wy ** 2 + wz ** 2))


def energy_spectrum(u, kmax=32):
    """各向同性能谱 E(k)，k=1..kmax（整数 bin）"""
    N = u.shape[1]
    uhat = np.fft.fftn(u, axes=(1, 2, 3)) / (N ** 3)
    E3d = 0.5 * (np.abs(uhat) ** 2).sum(0)
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).astype(int)
    spec = np.zeros(kmax + 1)
    cnt = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        m = kmag == k
        spec[k] = E3d[m].sum()
        cnt[k] = m.sum()
    return spec[1:].tolist(), cnt[1:].tolist()


def kinetic_energy(u):
    return 0.5 * float((u ** 2).sum(0).mean())


# ---------------- UPI spread-skill 工具 ----------------

def auroc_top_frac(score, target, frac=0.1):
    s = score.ravel()
    t = target.ravel()
    thr = np.quantile(t, 1.0 - frac)
    y = (t >= thr).astype(np.float64)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return float('nan')
    return float((a * b).sum() / denom)


# ---------------- 模型构建 ----------------

class BaselineWrapper(torch.nn.Module):
    """把 forward(x)->pred 的基线适配成套件接口：
    forward(u, target=None, return_components=True, ablation_mode=...) -> (pred, {})"""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, u, target=None, return_components=True, ablation_mode='full'):
        pred = self.base(u)
        return (pred, {}) if return_components else pred


T20_ARM2CKPT = {
    't20_fno': 'fno_official', 't20_unet': 'unet', 't20_dpot': 'dpot',
    't20_physreg': 'fno_physreg', 't20_transolver': 'transolver',
    'pc_fno': 'fno_official',   # PhysicsCorrect 包 fno_official
}


def build_model(name, device, mean=None, std=None):
    """返回 (model, mode, params)"""
    if name in T20_ARM2CKPT:
        from train_baselines_t20 import build_baseline
        key = T20_ARM2CKPT[name]
        base = build_baseline(key, device)
        ck = torch.load(f'checkpoints/t20_{key}/best.pt',
                        map_location=device, weights_only=False)
        base.load_state_dict(ck['model_state_dict'], strict=True)
        base.eval()
        params = sum(p.numel() for p in base.parameters())
        if name == 'pc_fno':
            from models.physics_correct import PhysicsCorrect3D
            model = BaselineWrapper(PhysicsCorrect3D(base))
        else:
            model = BaselineWrapper(base)
        return model, 'full', params
    if name in ('fix2', 'fix2_hc', 'fix2_no_upi', 'fix2_env'):
        ab = {'fix2': 'full', 'fix2_hc': 'hc_only', 'fix2_no_upi': 'no_upi',
              'fix2_env': 'full'}[name]
        model = load_fix2(FIX2_CKPT, device, ablation_mode=ab)
        if name == 'fix2_env':
            from models.dissipation_envelope import DissipationEnvelopeWrapper
            model = DissipationEnvelopeWrapper(model, mean, std).to(device)
        mode = ab
    elif name in ('fix2soft', 'fix2soft_off', 'fix2soft_g70', 'fix2soft_g90'):
        from models.rcln_fix2_soft import load_fix2soft
        model = load_fix2soft(FIX2SOFT_CKPT, device)
        mode = {'fix2soft': 'full', 'fix2soft_off': 'soft_off',
                'fix2soft_g70': 'gate70', 'fix2soft_g90': 'gate90'}[name]
    elif name in ('fix2_v2trans_sig', 'fix2_v2trans', 'fix2_v2trans_g70'):
        from models.rcln_fix2_soft import load_fix2_v2trans
        model = load_fix2_v2trans(device)
        mode = {'fix2_v2trans_sig': 'signal_only', 'fix2_v2trans': 'full',
                'fix2_v2trans_g70': 'gate70'}[name]
    elif name in ('v2', 'v2gate70', 'v2gate90'):
        model = RCLN_UPI_v5_Spec().to(device)
        ckpt = torch.load(V2_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        with torch.no_grad():
            model.spec_gain_raw.zero_()  # G=1 等价 v2 语义
        model.eval()
        mode = {'v2': 'no_guard', 'v2gate70': 'gate70', 'v2gate90': 'gate90'}[name]
    elif name == 'v3':
        model = RCLN_UPI_v5_Spec().to(device)
        ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.eval()
        mode = 'full'
    else:
        raise ValueError(name)
    params = sum(p.numel() for p in model.parameters())
    return model, mode, params


# ---------------- 单条 rollout ----------------

@torch.no_grad()
def rollout_one(model, mode, ic_norm, tgt_seq_norm, mean, std, device, K,
                dt_phys, spectra_steps, step_dts=None):
    u = ic_norm.unsqueeze(0).to(device).float()

    rL2 = np.zeros(K)
    E = np.zeros(K)
    ens = np.zeros(K)
    divm = np.zeros(K)
    divp95 = np.zeros(K)
    divmax = np.zeros(K)
    sr = np.full(K, np.nan)
    sa = np.full(K, np.nan)
    spectra = {}
    crashed = False
    crash_step = -1
    step_times = []

    ic_phys = ic_norm.numpy() * std[:, None, None, None] + mean[:, None, None, None]
    E0 = kinetic_energy(ic_phys)

    for k in range(K):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        if hasattr(model, 'set_step_dt'):
            model.set_step_dt(float(step_dts[k]) if step_dts is not None else dt_phys)
        pred, comp = model(u, target=None, return_components=True, ablation_mode=mode)
        if device == 'cuda':
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)

        pred_np = pred[0].cpu().numpy()
        pred_phys = pred_np * std[:, None, None, None] + mean[:, None, None, None]
        tgt_phys = tgt_seq_norm[k] * std[:, None, None, None] + mean[:, None, None, None]

        num = float(((pred_phys - tgt_phys) ** 2).sum())
        den = float((tgt_phys ** 2).sum()) + 1e-12
        rL2[k] = np.sqrt(num / den)
        E[k] = kinetic_energy(pred_phys) / (E0 + 1e-12)
        ens[k] = enstrophy(pred_phys)
        divm[k], divp95[k], divmax[k] = div_stats(pred_phys)
        if (k + 1) in spectra_steps:
            spectra[str(k + 1)], _ = energy_spectrum(pred_phys)

        # UPI 信号：|u_local| 通道范数 vs 真值误差图（归一化单位）
        u_local = comp.get('u_local', None)
        if u_local is not None:
            ul = u_local[0].cpu().numpy()
            score = np.sqrt((ul ** 2).sum(0))
            err_true = np.abs(pred_np - tgt_seq_norm[k]).mean(0)
            sr[k] = pearson(score, err_true)
            sa[k] = auroc_top_frac(score, err_true, 0.1)

        if (not np.isfinite(pred_np).all()) or E[k] > 5.0 or rL2[k] > 5.0 \
                or float(np.abs(pred_np).max()) > 50.0:
            crashed = True
            crash_step = k + 1
            rL2[k:] = rL2[k]
            E[k:] = E[k]
            ens[k:] = ens[k]
            divm[k:] = divm[k]
            divp95[k:] = divp95[k]
            divmax[k:] = divmax[k]
            break

        u = pred

    result = {
        'rL2': rL2, 'E': E, 'enstrophy': ens,
        'div_mean': divm, 'div_p95': divp95, 'div_max': divmax,
        'spread_r': sr, 'spread_auroc': sa, 'spectra': spectra,
        'crashed': crashed, 'crash_step': crash_step,
        'step_time_ms': float(np.mean(step_times) * 1000),
    }
    if hasattr(model, 'env_stats'):
        result['env_stats'] = model.env_stats()
        model.reset_stats()
    return result


def truth_stats(fields_phys, K, spectra_steps):
    """真值参考轨迹统计（逐 IC 平均）"""
    n_ic = fields_phys.shape[0] - K
    E = np.zeros((n_ic, K + 1))
    ens = np.zeros((n_ic, K + 1))
    spectra = {str(s): [] for s in spectra_steps}
    for ic in range(n_ic):
        for k in range(K + 1):
            u = fields_phys[ic + k]
            E[ic, k] = kinetic_energy(u)
            ens[ic, k] = enstrophy(u)
            if k in spectra_steps:
                sp, _ = energy_spectrum(u)
                spectra[str(k)].append(sp)
    E0 = E[:, 0:1] + 1e-12
    return {
        'E_over_E0': (E / E0).mean(0).tolist(),
        'enstrophy': ens.mean(0).tolist(),
        'spectra': {s: np.mean(v, 0).tolist() for s, v in spectra.items()},
        'div_mean': float(np.mean([div_stats(fields_phys[i])[0] for i in range(min(20, fields_phys.shape[0]))])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5_path', type=str,
                    default='data_generated/tgv_re6400_N64_T20.0_fluidsim.h5')
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--num_ic', type=int, default=10)
    ap.add_argument('--models', type=str, nargs='+',
                    default=['fix2', 'fix2_hc', 'v2', 'v2gate70', 'v2gate90', 'v3'])
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device} models={args.models} K={args.K}', flush=True)

    with h5py.File(args.h5_path, 'r') as f:
        ti = f['train_inputs'][:]
        mean = ti.mean(axis=(0, 2, 3, 4))
        std = ti.std(axis=(0, 2, 3, 4))
        del ti
        n_frames = args.num_ic + args.K + 1
        fields = f['fields'][:n_frames].astype(np.float32)
        times = f['times'][:n_frames]
    dt_phys = float(times[1] - times[0])
    print(f'fields {fields.shape} dt={dt_phys}', flush=True)
    fields_norm = (fields - mean[:, None, None, None]) / std[:, None, None, None]

    spectra_steps = tuple(s for s in SPECTRA_STEPS if s <= args.K)

    results = {'K': args.K, 'num_ic': args.num_ic, 'dt_phys': dt_phys,
               'models': {}}

    # 真值参考（传完整 fields，内部按 ic+k 索引，n_ic = num_ic+1）
    print('computing truth stats...', flush=True)
    results['truth'] = truth_stats(fields, args.K, spectra_steps)

    for name in args.models:
        model, mode, params = build_model(name, device, mean=mean, std=std)
        print(f'[{name}] mode={mode} params={params/1e6:.2f}M', flush=True)
        allr = []
        t0 = time.time()
        for ic in range(args.num_ic):
            ic_norm = torch.tensor(fields_norm[ic])
            tgt_seq = fields_norm[ic + 1: ic + 1 + args.K]
            step_dts = times[ic + 1: ic + 1 + args.K] - times[ic: ic + args.K]
            r = rollout_one(model, mode, ic_norm, tgt_seq, mean, std, device,
                            args.K, dt_phys, spectra_steps, step_dts=step_dts)
            allr.append(r)
            print(f'[{name}] IC{ic}: rL2@1={r["rL2"][0]:.4f} '
                  f'rL2@{args.K}={r["rL2"][-1]:.4f} E@{args.K}={r["E"][-1]:.4f} '
                  f'crash={r["crashed"]}@{r["crash_step"]} '
                  f'{r["step_time_ms"]:.0f}ms/step', flush=True)

        agg = lambda key: np.array([r[key] for r in allr])
        rL2a, Ea, ensa = agg('rL2'), agg('E'), agg('enstrophy')
        dm, dp, dx = agg('div_mean'), agg('div_p95'), agg('div_max')
        sra, saa = agg('spread_r'), agg('spread_auroc')
        crashes = sum(int(r['crashed']) for r in allr)
        ks = sorted({k for k in (1, 10, 50, 100, args.K) if k <= args.K})

        # eps(t) = -dE/dt（物理时间），由 E 轨迹差分
        eps = -np.diff(Ea.mean(0) * 1.0) / dt_phys  # E/E0 单位的耗散率

        spectra_mean = {}
        for s in spectra_steps:
            arr = np.array([r['spectra'][str(s)] for r in allr if str(s) in r['spectra']])
            if len(arr):
                spectra_mean[str(s)] = arr.mean(0).tolist()

        results['models'][name] = {
            'params': params,
            'crashes': crashes,
            'crash_steps': [r['crash_step'] for r in allr],
            'step_time_ms': float(np.mean([r['step_time_ms'] for r in allr])),
            'rL2': {f'@{k}': {'mean': float(rL2a[:, k - 1].mean()),
                              'std': float(rL2a[:, k - 1].std())} for k in ks},
            'E_over_E0': {f'@{k}': {'mean': float(Ea[:, k - 1].mean()),
                                    'std': float(Ea[:, k - 1].std())} for k in ks},
            'enstrophy': {f'@{k}': float(ensa[:, k - 1].mean()) for k in ks},
            'div': {f'@{k}': {'mean': float(dm[:, k - 1].mean()),
                              'p95': float(dp[:, k - 1].mean()),
                              'max': float(dx[:, k - 1].mean())} for k in ks},
            'spread_pearson_mean': float(np.nanmean(sra)),
            'spread_auroc_mean': float(np.nanmean(saa)),
            'trajectories': {
                'rL2_mean': rL2a.mean(0).tolist(),
                'E_mean': Ea.mean(0).tolist(),
                'eps_mean': eps.tolist(),
                'enstrophy_mean': ensa.mean(0).tolist(),
                'div_mean': dm.mean(0).tolist(),
                'div_p95': dp.mean(0).tolist(),
                'div_max': dx.mean(0).tolist(),
                'spread_auroc_mean': np.nanmean(saa, 0).tolist(),
                'spread_r_mean': np.nanmean(sra, 0).tolist(),
            },
            'spectra': spectra_mean,
            'wall_s': time.time() - t0,
        }
        if any('env_stats' in r for r in allr):
            es = [r['env_stats'] for r in allr if 'env_stats' in r]
            results['models'][name]['envelope'] = {
                'floor_hit_rate': float(np.mean([e['floor_hit_rate'] for e in es])),
                'ceil_hit_rate': float(np.mean([e['ceil_hit_rate'] for e in es])),
                'mean_abs_corr': float(np.mean([e['mean_abs_corr'] for e in es])),
                'c': es[0]['c'], 'gamma': es[0]['gamma'], 'nu': es[0]['nu'],
            }
        print(f'[{name}] done in {results["models"][name]["wall_s"]:.0f}s '
              f'crashes={crashes}/{args.num_ic}', flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()

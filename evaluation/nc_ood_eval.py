#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_ood_eval.py — NC 任务书 Generality(G) + Mechanism M3 零训练评估

G  Generality（模型全部只在 TGV Re6400 上训练，以下均为真 OOD）:
   jhtdb   : JHTDB isotropic Re_l=433 HIT, 4 traj x 5 IC, K=25  —— 流型 OOD（+U4）
   re3200  : TGV Re3200, 10 IC, K=100                            —— Re OOD
   re1000  : TGV Re1000, 10 IC, K=30                             —— Re OOD
M3 约束压力测试（主分布 TGV Re6400 + IC 高斯噪声）:
   noise1  : sigma=0.01xstd, 5 IC, K=100
   noise5  : sigma=0.05xstd, 5 IC, K=100

归一化: 一律用 Re6400 训练统计（部署约定），指标在目标数据物理单位下计算。
输出: results/paper_experiments/nc_ood_eval.json
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from nc_suite_eval import build_model, enstrophy, kinetic_energy, auroc_top_frac, pearson

MAIN_H5 = 'data_generated/tgv_re6400_N64_T20.0_fluidsim.h5'

DATASETS = {
    'jhtdb': dict(h5=[f'data_generated/jhtdb_iso_re433_N64_traj{i:02d}.h5' for i in range(4)],
                  K=25, ic_per_traj=5, noise=0.0),
    're3200': dict(h5=['data_generated/tgv_re3200_N64_T5.0_fluidsim.h5'],
                   K=100, ic_per_traj=10, noise=0.0),
    're1000': dict(h5=['data_generated/tgv_re1000_N64_T2.0_fluidsim.h5'],
                   K=30, ic_per_traj=10, noise=0.0),
    'noise1': dict(h5=[MAIN_H5], K=100, ic_per_traj=5, noise=0.01),
    'noise5': dict(h5=[MAIN_H5], K=100, ic_per_traj=5, noise=0.05),
}


@torch.no_grad()
def rollout_one(model, mode, ic_norm, tgt_seq_norm, mean, std, device, K, noise_std, rng):
    u = ic_norm.unsqueeze(0).to(device).float()
    if noise_std > 0:
        u = u + noise_std * torch.tensor(rng.standard_normal(u.shape), device=device).float()

    rL2 = np.zeros(K)
    E = np.zeros(K)
    ens = np.zeros(K)
    sr = np.full(K, np.nan)
    sa = np.full(K, np.nan)
    crashed = False
    crash_step = -1

    ic_phys = ic_norm.numpy() * std[:, None, None, None] + mean[:, None, None, None]
    E0 = kinetic_energy(ic_phys)

    for k in range(K):
        pred, comp = model(u, target=None, return_components=True, ablation_mode=mode)
        pred_np = pred[0].cpu().numpy()
        pred_phys = pred_np * std[:, None, None, None] + mean[:, None, None, None]
        tgt_phys = tgt_seq_norm[k] * std[:, None, None, None] + mean[:, None, None, None]

        num = float(((pred_phys - tgt_phys) ** 2).sum())
        den = float((tgt_phys ** 2).sum()) + 1e-12
        rL2[k] = np.sqrt(num / den)
        E[k] = kinetic_energy(pred_phys) / (E0 + 1e-12)
        ens[k] = enstrophy(pred_phys)

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
            break
        u = pred

    return {'rL2': rL2, 'E': E, 'enstrophy': ens, 'spread_r': sr, 'spread_auroc': sa,
            'crashed': crashed, 'crash_step': crash_step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, nargs='+',
                    default=['jhtdb', 're3200', 're1000', 'noise1', 'noise5'])
    ap.add_argument('--models', type=str, nargs='+',
                    default=['fix2', 'v2', 'v2gate70'])
    ap.add_argument('--stress_models', type=str, nargs='+',
                    default=['fix2', 'v2', 'fix2_no_upi'],
                    help='noise* 数据集用的模型集（含无约束对照）')
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(20260810)

    # Re6400 训练统计（部署归一化约定）
    with h5py.File(MAIN_H5, 'r') as f:
        ti = f['train_inputs'][:]
    mean = ti.mean(axis=(0, 2, 3, 4))
    std = ti.std(axis=(0, 2, 3, 4))
    del ti
    print(f'device={device} norm std={std}', flush=True)

    results = {'norm_std': std.tolist(), 'datasets': {}}

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        models = args.stress_models if ds_name.startswith('noise') else args.models
        # 预载数据
        traj_list = []
        for h5p in cfg['h5']:
            with h5py.File(h5p, 'r') as f:
                traj_list.append(f['fields'][:].astype(np.float32))
        K = cfg['K']
        ds_out = {'K': K, 'noise': cfg['noise'], 'models': {}}

        for mname in models:
            model, mode, params = build_model(mname, device)
            allr = []
            t0 = time.time()
            for traj in traj_list:
                traj_norm = (traj - mean[:, None, None, None]) / std[:, None, None, None]
                for ic in range(cfg['ic_per_traj']):
                    if ic + 1 + K > traj.shape[0]:
                        break
                    r = rollout_one(model, mode, torch.tensor(traj_norm[ic]),
                                    traj_norm[ic + 1: ic + 1 + K], mean, std,
                                    device, K, cfg['noise'], rng)
                    allr.append(r)
            agg = lambda key: np.array([r[key] for r in allr])
            rL2a, Ea, ensa, saa = agg('rL2'), agg('E'), agg('enstrophy'), agg('spread_auroc')
            crashes = sum(int(r['crashed']) for r in allr)
            ks = sorted({k for k in (1, 10, 25, 50, 100, K) if k <= K})
            ds_out['models'][mname] = {
                'n_rollouts': len(allr),
                'crashes': crashes,
                'crash_steps': [r['crash_step'] for r in allr],
                'rL2': {f'@{k}': {'mean': float(rL2a[:, k - 1].mean()),
                                  'std': float(rL2a[:, k - 1].std())} for k in ks},
                'E_over_E0': {f'@{k}': float(Ea[:, k - 1].mean()) for k in ks},
                'enstrophy': {f'@{k}': float(ensa[:, k - 1].mean()) for k in ks},
                'spread_auroc_mean': float(np.nanmean(saa)),
                'trajectories': {
                    'rL2_mean': rL2a.mean(0).tolist(),
                    'E_mean': Ea.mean(0).tolist(),
                    'enstrophy_mean': ensa.mean(0).tolist(),
                    'spread_auroc_mean': np.nanmean(saa, 0).tolist(),
                },
                'wall_s': time.time() - t0,
            }
            print(f'[{ds_name}/{mname}] n={len(allr)} crashes={crashes} '
                  f'rL2@K={rL2a[:, -1].mean():.4f} E@K={Ea[:, -1].mean():.4f} '
                  f'AUROC={np.nanmean(saa):.3f} ({ds_out["models"][mname]["wall_s"]:.0f}s)',
                  flush=True)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        results['datasets'][ds_name] = ds_out

    out_path = Path(args.output)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()

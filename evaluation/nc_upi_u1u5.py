#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_upi_u1u5.py — NC 任务书第六节 U1-U5：UPI emergent error signal 完整检验

模型: v2（信号最活，主角）+ fix2（原生 UPI 对照）
数据: tgv(in-domain) / re3200 / re1000 / jhtdb（U4 三个 OOD）
输出:
  U1 空间相关: 逐步 Pearson r(score, |err|)
  U2 检测: 逐步 AUROC + AUPRC（top-10% 体素误差事件）
  U3 校准: (score, err) 水库采样 -> 分箱可靠性曲线 + 箱内失败概率
  U4: 以上在 OOD 重复
  U5: MC-dropout 基线（v2，8 次随机前向，预测方差图）同指标对比
  命名裁决输入: U1-U4 是否稳定

输出: results/paper_experiments/nc_upi_u1u5.json
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from nc_suite_eval import build_model, auroc_top_frac, pearson

MAIN_H5 = 'data_generated/tgv_re6400_N64_T20.0_fluidsim.h5'

DATASETS = {
    'tgv':    dict(h5=[MAIN_H5], K=100, ic=10),
    're3200': dict(h5=['data_generated/tgv_re3200_N64_T5.0_fluidsim.h5'], K=100, ic=10),
    're1000': dict(h5=['data_generated/tgv_re1000_N64_T2.0_fluidsim.h5'], K=30, ic=10),
    'jhtdb':  dict(h5=[f'data_generated/jhtdb_iso_re433_N64_traj{i:02d}.h5' for i in range(4)],
                   K=25, ic=5),
}
CAL_STEPS = (1, 5, 10, 25, 50, 100)   # 校准采样步（按各数据集 K 截断）
RESERVOIR = 60000
MC_PASSES = 8
MC_STEPS = (1, 10, 25, 50, 100)


def auprc_top_frac(score, target, frac=0.1):
    """Average Precision（top-frac 事件为正类）"""
    s = score.ravel()
    t = target.ravel()
    thr = np.quantile(t, 1.0 - frac)
    y = (t >= thr).astype(np.int64)
    order = np.argsort(-s, kind='mergesort')
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(tp[-1], 1)
    # AP = sum((rec_i - rec_{i-1}) * prec_i)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(((rec - rec_prev) * prec).sum())


class Reservoir:
    """等概率水库采样 (score_norm, err_norm) 对"""
    def __init__(self, cap, rng):
        self.cap = cap
        self.rng = rng
        self.s = []
        self.e = []
        self.n = 0

    def offer(self, s, e):
        m = len(s)
        self.n += m
        if len(self.s) < self.cap:
            take = min(self.cap - len(self.s), m)
            self.s.extend(s[:take])
            self.e.extend(e[:take])
        else:
            idx = self.rng.random(m) < (self.cap / self.n)
            rep = self.rng.integers(0, self.cap, int(idx.sum()))
            si = np.asarray(self.s)
            ei = np.asarray(self.e)
            si[rep] = s[idx]
            ei[rep] = e[idx]
            self.s = si.tolist()
            self.e = ei.tolist()


def enable_dropout_only(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout3d, nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def rollout_collect(model, mode, ic_norm, tgt_seq_norm, device, K, reservoir, rng,
                    mc_model=None):
    """单条 rollout：逐步 U1/U2 指标 + 校准采样 + 可选 MC-dropout"""
    u = ic_norm.unsqueeze(0).to(device).float()
    r = np.full(K, np.nan)
    au = np.full(K, np.nan)
    ap = np.full(K, np.nan)
    mc_au = {}
    mc_ap = {}

    for k in range(K):
        pred, comp = model(u, target=None, return_components=True, ablation_mode=mode)
        pred_np = pred[0].cpu().numpy()
        err = np.abs(pred_np - tgt_seq_norm[k]).mean(0)

        u_local = comp.get('u_local', None)
        if u_local is not None:
            score = np.sqrt((u_local[0].cpu().numpy() ** 2).sum(0))
            r[k] = pearson(score, err)
            au[k] = auroc_top_frac(score, err, 0.1)
            ap[k] = auprc_top_frac(score, err, 0.1)
            if (k + 1) in CAL_STEPS:
                # 逐快照归一化后入水库（消除跨步量纲漂移）
                sn = score / (score.mean() + 1e-12)
                en = err / (err.mean() + 1e-12)
                reservoir.offer(sn.ravel(), en.ravel())

        # MC-dropout（仅 v2 且本步在 MC_STEPS）
        if mc_model is not None and (k + 1) in MC_STEPS:
            preds = []
            enable_dropout_only(mc_model)
            for _ in range(MC_PASSES):
                p = mc_model(u, target=None, return_components=False, ablation_mode=mode)
                preds.append(p[0].cpu().numpy())
            mc_model.eval()
            var_map = np.var(np.stack(preds), axis=0).mean(0)
            mc_au[k + 1] = auroc_top_frac(var_map, err, 0.1)
            mc_ap[k + 1] = auprc_top_frac(var_map, err, 0.1)

        u = pred

    return {'pearson': r, 'auroc': au, 'auprc': ap, 'mc_auroc': mc_au, 'mc_auprc': mc_ap}


def calibration_curves(s_arr, e_arr, n_bins=20):
    """按 score 分位数分箱：mean e_norm vs mean s_norm + 箱内 P(top10% err)"""
    s = np.asarray(s_arr)
    e = np.asarray(e_arr)
    qs = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    bin_idx = np.digitize(s, qs) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    # top-10% err 事件按全局 err 定义（已逐快照归一化，~top10% 近似）
    e_thr = np.quantile(e, 0.9)
    xs, ys, ps, ns = [], [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() < 10:
            continue
        xs.append(float(s[m].mean()))
        ys.append(float(e[m].mean()))
        ps.append(float((e[m] > e_thr).mean()))
        ns.append(int(m.sum()))
    return {'bin_score': xs, 'bin_err': ys, 'bin_fail_prob': ps, 'bin_count': ns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, nargs='+',
                    default=['tgv', 're3200', 're1000', 'jhtdb'])
    ap.add_argument('--models', type=str, nargs='+', default=['v2', 'fix2'])
    ap.add_argument('--mc_model', type=str, default='v2')
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(20260811)

    with h5py.File(MAIN_H5, 'r') as f:
        ti = f['train_inputs'][:]
    mean = ti.mean(axis=(0, 2, 3, 4))
    std = ti.std(axis=(0, 2, 3, 4))
    del ti

    results = {'datasets': {}}
    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        K = cfg['K']
        trajs = []
        for p in cfg['h5']:
            with h5py.File(p, 'r') as f:
                trajs.append(f['fields'][:cfg['ic'] + K + 1].astype(np.float32))
        ds_out = {'K': K, 'models': {}}

        for mname in args.models:
            model, mode, _ = build_model(mname, device)
            mc_model = model if mname == args.mc_model else None
            reservoir = Reservoir(RESERVOIR, rng)
            allr = []
            t0 = time.time()
            for traj in trajs:
                traj_norm = (traj - mean[:, None, None, None]) / std[:, None, None, None]
                for ic in range(cfg['ic']):
                    if ic + 1 + K > traj.shape[0]:
                        break
                    r = rollout_collect(model, mode, torch.tensor(traj_norm[ic]),
                                        traj_norm[ic + 1: ic + 1 + K], device, K,
                                        reservoir, rng, mc_model=mc_model)
                    allr.append(r)
            agg = lambda key: np.array([r[key] for r in allr])
            cal = calibration_curves(np.array(reservoir.s), np.array(reservoir.e))
            mc_au = {}
            mc_ap = {}
            if mc_model is not None:
                for st in MC_STEPS:
                    vals_au = [r['mc_auroc'][st] for r in allr if st in r['mc_auroc']]
                    vals_ap = [r['mc_auprc'][st] for r in allr if st in r['mc_auprc']]
                    if vals_au:
                        mc_au[str(st)] = float(np.mean(vals_au))
                        mc_ap[str(st)] = float(np.mean(vals_ap))
            ds_out['models'][mname] = {
                'n_rollouts': len(allr),
                'U1_pearson_mean': float(np.nanmean(agg('pearson'))),
                'U2_auroc_mean': float(np.nanmean(agg('auroc'))),
                'U2_auprc_mean': float(np.nanmean(agg('auprc'))),
                'traj_pearson': np.nanmean(agg('pearson'), 0).tolist(),
                'traj_auroc': np.nanmean(agg('auroc'), 0).tolist(),
                'traj_auprc': np.nanmean(agg('auprc'), 0).tolist(),
                'U3_calibration': cal,
                'U5_mc_dropout': {'auroc': mc_au, 'auprc': mc_ap},
                'wall_s': time.time() - t0,
            }
            print(f'[{ds_name}/{mname}] n={len(allr)} r={ds_out["models"][mname]["U1_pearson_mean"]:.3f} '
                  f'AUROC={ds_out["models"][mname]["U2_auroc_mean"]:.3f} '
                  f'AUPRC={ds_out["models"][mname]["U2_auprc_mean"]:.3f} '
                  f'({ds_out["models"][mname]["wall_s"]:.0f}s)', flush=True)
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

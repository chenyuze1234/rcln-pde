#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_physics_stats.py — NC 任务书第五节「真实动力学」补充统计

对 7 个模型重跑 3 IC × K=200 rollout，在 k∈{50,100,200} 快照预测场，计算：
  - 速度分量 PDF + 偏度/峰度
  - 涡量模 |ω| PDF + 偏度/峰度
  - 纵向二阶结构函数 S2(r)
真值做同样统计作参考。ε(t) 与成本在主 JSON(nc_suite_k200.json)中已有，此处不重算。

输出: results/paper_experiments/nc_physics_stats.json
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from nc_suite_eval import build_model

SNAP_STEPS = (50, 100, 200)
VEL_BINS = np.linspace(-1.5, 1.5, 61)
VOR_BINS = np.linspace(0.0, 3.0, 61)
R_MAX = 32


def vorticity_mag(u):
    def d(f, ax):
        return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / 2
    wx = d(u[2], 1) - d(u[1], 2)
    wy = d(u[0], 2) - d(u[2], 0)
    wz = d(u[1], 0) - d(u[0], 1)
    return np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)


def moments(x):
    x = x.ravel().astype(np.float64)
    m = x.mean()
    c = x - m
    m2 = (c ** 2).mean()
    if m2 < 1e-20:
        return {'mean': float(m), 'std': 0.0, 'skew': 0.0, 'kurt': 0.0}
    m3 = (c ** 3).mean()
    m4 = (c ** 4).mean()
    return {'mean': float(m), 'std': float(np.sqrt(m2)),
            'skew': float(m3 / m2 ** 1.5), 'kurt': float(m4 / m2 ** 2 - 3.0)}


def pdf_hist(x, bins):
    h, _ = np.histogram(x.ravel(), bins=bins, density=True)
    return h


def structure_function_S2(u, r_max=R_MAX):
    """纵向 S2(r)：三方向平均，u (3,H,W,D) 物理单位"""
    S2 = np.zeros(r_max + 1)
    for a in range(3):
        comp = u[a]
        for r in range(1, r_max + 1):
            du = np.roll(comp, -r, axis=a) - comp
            S2[r] += float((du ** 2).mean())
    S2[1:] /= 3.0
    return S2[1:]


def field_stats(u_phys):
    vm = vorticity_mag(u_phys)
    return {
        'vel_pdf': pdf_hist(u_phys, VEL_BINS),
        'vel_mom': moments(u_phys),
        'vor_pdf': pdf_hist(vm, VOR_BINS),
        'vor_mom': moments(vm),
        'S2': structure_function_S2(u_phys),
    }


@torch.no_grad()
def rollout_collect(model, mode, ic_norm, mean, std, device, K, step_dts=None):
    u = ic_norm.unsqueeze(0).to(device).float()
    snaps = {}
    for k in range(K):
        if hasattr(model, 'set_step_dt'):
            if step_dts is None:
                raise RuntimeError('包络模型需要 step_dts')
            model.set_step_dt(float(step_dts[k]))
        pred = model(u, target=None, return_components=False, ablation_mode=mode)
        if (k + 1) in SNAP_STEPS:
            snaps[k + 1] = pred[0].cpu().numpy() * std[:, None, None, None] + mean[:, None, None, None]
        u = pred
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5_path', type=str,
                    default='data_generated/tgv_re6400_N64_T20.0_fluidsim.h5')
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--num_ic', type=int, default=3)
    ap.add_argument('--models', type=str, nargs='+',
                    default=['fix2', 'fix2_hc', 'fix2_no_upi', 'v2', 'v2gate70', 'v2gate90', 'v3'])
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with h5py.File(args.h5_path, 'r') as f:
        ti = f['train_inputs'][:]
        mean = ti.mean(axis=(0, 2, 3, 4))
        std = ti.std(axis=(0, 2, 3, 4))
        del ti
        fields = f['fields'][:args.num_ic + args.K + 1].astype(np.float32)
        times = f['times'][:args.num_ic + args.K + 1]
    fields_norm = (fields - mean[:, None, None, None]) / std[:, None, None, None]
    print(f'device={device} K={args.K} ICs={args.num_ic}', flush=True)

    results = {'K': args.K, 'num_ic': args.num_ic, 'snap_steps': list(SNAP_STEPS),
               'vel_bins': VEL_BINS.tolist(), 'vor_bins': VOR_BINS.tolist(),
               'models': {}}

    # 真值统计（逐 IC 逐快照，之后平均）
    truth = {str(s): {'vel_pdf': [], 'vor_pdf': [], 'S2': [], 'vel_mom': [], 'vor_mom': []}
             for s in SNAP_STEPS}
    for ic in range(args.num_ic):
        for s in SNAP_STEPS:
            st = field_stats(fields[ic + s])
            truth[str(s)]['vel_pdf'].append(st['vel_pdf'])
            truth[str(s)]['vor_pdf'].append(st['vor_pdf'])
            truth[str(s)]['S2'].append(st['S2'])
            truth[str(s)]['vel_mom'].append(st['vel_mom'])
            truth[str(s)]['vor_mom'].append(st['vor_mom'])
    results['truth'] = {}
    for s in SNAP_STEPS:
        key = str(s)
        results['truth'][key] = {
            'vel_pdf': np.mean(truth[key]['vel_pdf'], 0).tolist(),
            'vor_pdf': np.mean(truth[key]['vor_pdf'], 0).tolist(),
            'S2': np.mean(truth[key]['S2'], 0).tolist(),
            'vel_mom': {m: float(np.mean([x[m] for x in truth[key]['vel_mom']]))
                        for m in ('mean', 'std', 'skew', 'kurt')},
            'vor_mom': {m: float(np.mean([x[m] for x in truth[key]['vor_mom']]))
                        for m in ('mean', 'std', 'skew', 'kurt')},
        }
    print('truth done', flush=True)

    for name in args.models:
        model, mode, params = build_model(name, device, mean=mean, std=std)
        t0 = time.time()
        acc = {str(s): {'vel_pdf': [], 'vor_pdf': [], 'S2': [], 'vel_mom': [], 'vor_mom': []}
               for s in SNAP_STEPS}
        for ic in range(args.num_ic):
            ic_norm = torch.tensor(fields_norm[ic])
            step_dts = times[ic + 1: ic + 1 + args.K] - times[ic: ic + args.K]
            snaps = rollout_collect(model, mode, ic_norm, mean, std, device, args.K,
                                    step_dts=step_dts)
            for s, u_phys in snaps.items():
                st = field_stats(u_phys)
                acc[str(s)]['vel_pdf'].append(st['vel_pdf'])
                acc[str(s)]['vor_pdf'].append(st['vor_pdf'])
                acc[str(s)]['S2'].append(st['S2'])
                acc[str(s)]['vel_mom'].append(st['vel_mom'])
                acc[str(s)]['vor_mom'].append(st['vor_mom'])
            print(f'[{name}] IC{ic} done', flush=True)
        results['models'][name] = {}
        for s in SNAP_STEPS:
            key = str(s)
            results['models'][name][key] = {
                'vel_pdf': np.mean(acc[key]['vel_pdf'], 0).tolist(),
                'vor_pdf': np.mean(acc[key]['vor_pdf'], 0).tolist(),
                'S2': np.mean(acc[key]['S2'], 0).tolist(),
                'vel_mom': {m: float(np.mean([x[m] for x in acc[key]['vel_mom']]))
                            for m in ('mean', 'std', 'skew', 'kurt')},
                'vor_mom': {m: float(np.mean([x[m] for x in acc[key]['vor_mom']]))
                            for m in ('mean', 'std', 'skew', 'kurt')},
            }
        print(f'[{name}] done in {time.time()-t0:.0f}s', flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    out_path = Path(args.output)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()

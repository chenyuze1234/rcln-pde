#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_baselines_t20.py — NC 冲刺版 P0-5：T20 同协议基线训练器

所有外部基线在与 fix2 完全相同的协议下重训：
  数据: data_generated/tgv_re6400_N64_T20.0_fluidsim.h5 的 train_inputs/train_targets
  归一化: train_inputs 的逐通道 mean/std（与 nc_suite_eval.py 完全一致）
  训练: 100 epochs, batch 1, AdamW lr=1e-4, cosine, clip 1.0, AMP(可关),
        rollout=3 步 gamma=0.9 折扣 MSE（fix2 同款长程训练）
  验证: val_inputs->val_targets 单步 MSE，best 保存，patience 15 早停
  产出: checkpoints/t20_<model>/{best,final}.pt + config.json + history.json

用法:
  python -u train_baselines_t20.py --model fno_official
  python -u train_baselines_t20.py --model transolver --no_amp   # AMP 溢出时回退
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

H5 = 'data_generated/tgv_re6400_N64_T20.0_fluidsim.h5'


def build_baseline(name, device):
    if name in ('fno_official', 'fno_physreg'):
        from neuralop.models import FNO
        m = FNO(n_modes=(8, 8, 8), hidden_channels=72, in_channels=3,
                out_channels=3, n_layers=4)
    elif name == 'unet':
        from models.unet_3d import UNet3D
        m = UNet3D(in_ch=3, out_ch=3, base=34)
    elif name == 'dpot':
        from models.baselines_v6_ext import DPOTAdapter
        m = DPOTAdapter()
    elif name == 'transolver':
        from models.baselines_v6_ext import TransolverAdapter
        m = TransolverAdapter()
    else:
        raise ValueError(name)
    return m.to(device)


def div_central(u):
    """中心差分散度（归一化单位诊断用）u: (B,3,N,N,N)"""
    d = lambda f, ax: (torch.roll(f, -1, ax) - torch.roll(f, 1, ax)) / 2
    return d(u[:, 0], 1) + d(u[:, 1], 2) + d(u[:, 2], 3)


def phys_loss(pred, tgt, ld=0.01, le=0.05):
    """与 train_fno_physloss_64 完全相同的物理惩罚（PINO 风格臂）"""
    lmse = F.mse_loss(pred, tgt)
    ldiv = (div_central(pred) ** 2).mean()
    Ep = 0.5 * (pred ** 2).sum(dim=1).flatten(1).mean(1)
    Et = 0.5 * (tgt ** 2).sum(dim=1).flatten(1).mean(1)
    lene = F.mse_loss(Ep / (Et + 1e-8), torch.ones_like(Ep))
    return lmse + ld * ldiv + le * lene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['fno_official', 'unet', 'dpot', 'fno_physreg', 'transolver'])
    ap.add_argument('--h5_path', default=H5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--rollout_steps', type=int, default=3)
    ap.add_argument('--rollout_gamma', type=float, default=0.9)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or f'checkpoints/t20_{args.model}')
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with h5py.File(args.h5_path, 'r') as f:
        ti = f['train_inputs'][:].astype(np.float32)   # (482,3,64,64,64)
        vi = f['val_inputs'][:].astype(np.float32)
        vt = f['val_targets'][:].astype(np.float32)
    mean = ti.mean(axis=(0, 2, 3, 4))
    std = ti.std(axis=(0, 2, 3, 4))
    ti_n = (ti - mean[:, None, None, None]) / std[:, None, None, None]
    vi_n = (vi - mean[:, None, None, None]) / std[:, None, None, None]
    vt_n = (vt - mean[:, None, None, None]) / std[:, None, None, None]
    R = args.rollout_steps
    n_train = ti_n.shape[0] - R          # 479 个 rollout 窗口
    print(f'data: train_windows={n_train} val={vi_n.shape[0]} R={R}', flush=True)

    model = build_baseline(args.model, device)
    params = sum(p.numel() for p in model.parameters())
    use_amp = (not args.no_amp) and (not args.model.startswith('fno'))  # neuralop FNO ComplexFloat 不支持 GradScaler → fno 臂 fp32
    print(f'model={args.model} params={params/1e6:.2f}M amp={use_amp}',
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    cfg = vars(args)
    cfg.update({'params': params, 'norm': 'per-channel train_inputs',
                'mean': mean.tolist(), 'std': std.tolist(),
                'protocol': 'fix2-identical: T20 train split, 100ep, R=3 g=0.9, '
                            'AdamW 1e-4 cosine clip1.0'})
    json.dump(cfg, open(out_dir / 'config.json', 'w'), ensure_ascii=False, indent=1)

    best_val = float('inf')
    bad = 0
    hist = {'train_loss': [], 'val_loss': [], 'lr': []}
    ti_t = torch.tensor(ti_n)
    vi_t = torch.tensor(vi_n)
    vt_t = torch.tensor(vt_n)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n_train)
        tot, nb = 0.0, 0
        for bi, i in enumerate(perm):
            x = ti_t[i].unsqueeze(0).to(device)
            tgts = [ti_t[i + r].unsqueeze(0).to(device) for r in range(1, R + 1)]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = 0.0
                u = x
                for r in range(R):
                    u = model(u)
                    l = (phys_loss(u, tgts[r]) if args.model == 'fno_physreg'
                         else F.mse_loss(u, tgts[r]))
                    loss = loss + (args.rollout_gamma ** r) * l
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
            if (bi + 1) % 50 == 0:
                print(f'[{bi+1}/{n_train}] loss={tot/nb:.5f}', flush=True)
            if not np.isfinite(tot / nb):
                print(f'[FATAL] NaN loss at epoch {ep} batch {bi+1}', flush=True)
                return
        sched.step()

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for j in range(vi_t.shape[0]):
                x = vi_t[j].unsqueeze(0).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    p = model(x)
                vtot += F.mse_loss(p.float(), vt_t[j].unsqueeze(0).to(device)).item()
        val = vtot / vi_t.shape[0]
        tr = tot / nb
        hist['train_loss'].append(tr)
        hist['val_loss'].append(val)
        hist['lr'].append(sched.get_last_lr()[0])
        star = ''
        if val < best_val:
            best_val = val
            bad = 0
            star = '*'
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'best_val': best_val}, out_dir / 'best.pt')
        else:
            bad += 1
        print(f'Epoch {ep}/{args.epochs} | {time.time()-t0:.0f}s | '
              f'tr={tr:.5f} val={val:.5f}{star} | Patience {bad}/{args.patience}',
              flush=True)
        json.dump(hist, open(out_dir / 'history.json', 'w'))
        if bad >= args.patience:
            print('early stop', flush=True)
            break

    torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                'best_val': best_val}, out_dir / 'final.pt')
    print(f'DONE. Best val={best_val:.5f}', flush=True)


if __name__ == '__main__':
    main()

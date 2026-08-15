#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_fix2_soft.py — vNC 阶段 1：训练 fix2 残差软分支（fix2 全冻结）

数据: data_generated/tgv_re6400_N64_T20.0_fluidsim.h5 (482 train / 120 val)
损失: MSE(u_soft, (target - fix2(u)).detach())   —— v2 式直接监督，无增广无剪切
输出: checkpoints/v5_fix2_soft/v5_best.pt (soft_state_dict + config)
日志: results/logs/fix2_soft_train.log
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from models.rcln_fix2_soft import RCLN_Fix2Soft
from models.rcln_upi_v5_spec import spectral_lowpass

H5 = 'data_generated/tgv_re6400_N64_T20.0_fluidsim.h5'
CKPT_DIR = Path('checkpoints/v5_fix2_soft')
EPOCHS = 100
BS = 8
LR = 1e-3
WD = 1e-4
PATIENCE = 15


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5, 'r') as f:
        tr_x = f['train_inputs'][:].astype(np.float32)
        tr_y = f['train_targets'][:].astype(np.float32)
        va_x = f['val_inputs'][:].astype(np.float32)
        va_y = f['val_targets'][:].astype(np.float32)
    mean = tr_x.mean(axis=(0, 2, 3, 4))
    std = tr_x.std(axis=(0, 2, 3, 4))
    tr_x = (tr_x - mean[:, None, None, None]) / std[:, None, None, None]
    tr_y = (tr_y - mean[:, None, None, None]) / std[:, None, None, None]
    va_x = (va_x - mean[:, None, None, None]) / std[:, None, None, None]
    va_y = (va_y - mean[:, None, None, None]) / std[:, None, None, None]
    print(f'train {tr_x.shape} val {va_x.shape} std={std}', flush=True)

    model = RCLN_Fix2Soft(device=device)
    n_soft = sum(p.numel() for p in model.soft.parameters())
    print(f'soft params: {n_soft/1e3:.1f}K (fix2 frozen)', flush=True)

    opt = torch.optim.AdamW(model.soft.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    tr_x_t = torch.tensor(tr_x)
    tr_y_t = torch.tensor(tr_y)
    va_x_t = torch.tensor(va_x).to(device)
    va_y_t = torch.tensor(va_y).to(device)

    best_val = float('inf')
    best_epoch = -1
    bad = 0
    history = []
    n_train = tr_x_t.shape[0]

    for ep in range(1, EPOCHS + 1):
        model.soft.train()
        perm = torch.randperm(n_train)
        tr_loss = 0.0
        t0 = time.time()
        for i in range(0, n_train, BS):
            idx = perm[i:i + BS]
            x = tr_x_t[idx].to(device)
            y = tr_y_t[idx].to(device)
            with torch.no_grad():
                fix2_out = model.fix2(x, target=None, return_components=False,
                                      ablation_mode='full')
                resid = y - fix2_out
                u_high = x - spectral_lowpass(x, model.k_cut)
                soft_in = torch.cat([u_high, fix2_out], dim=1)
            u_soft = model.soft(soft_in)
            loss = F.mse_loss(u_soft, resid)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.soft.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(idx)
        tr_loss /= n_train
        sched.step()

        model.soft.eval()
        va_loss = 0.0
        with torch.no_grad():
            for i in range(0, va_x_t.shape[0], BS):
                x = va_x_t[i:i + BS]
                y = va_y_t[i:i + BS]
                fix2_out = model.fix2(x, target=None, return_components=False,
                                      ablation_mode='full')
                resid = y - fix2_out
                u_high = x - spectral_lowpass(x, model.k_cut)
                u_soft = model.soft(torch.cat([u_high, fix2_out], dim=1))
                va_loss += F.mse_loss(u_soft, resid).item() * x.shape[0]
        va_loss /= va_x_t.shape[0]

        history.append({'epoch': ep, 'train': tr_loss, 'val': va_loss})
        improved = va_loss < best_val - 1e-6
        if improved:
            best_val = va_loss
            best_epoch = ep
            bad = 0
            torch.save({'soft_state_dict': model.soft.state_dict(),
                        'config': {'k_cut': model.k_cut, 'soft_base': 16,
                                   'in_channels': 6},
                        'best_epoch': ep, 'best_val': best_val},
                       CKPT_DIR / 'v5_best.pt')
        else:
            bad += 1
        print(f'E{ep:03d} train={tr_loss:.5f} val={va_loss:.5f} '
              f'best=E{best_epoch}({best_val:.5f}) {time.time()-t0:.1f}s', flush=True)
        if bad >= PATIENCE:
            print(f'Early stopping at epoch {ep}, best E{best_epoch}', flush=True)
            break

    with open(CKPT_DIR / 'history.json', 'w') as f:
        json.dump({'history': history, 'best_epoch': best_epoch, 'best_val': best_val}, f)
    print(f'done. best E{best_epoch} val={best_val:.5f}', flush=True)


if __name__ == '__main__':
    main()

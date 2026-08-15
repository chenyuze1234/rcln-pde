"""
System B resolution OOD —— 64^2 训练模型 zero-shot 上 128^2（NC 任务书 Step 6 收尾）
====================================================================
FNO 论文式 zero-shot super-resolution 协议:
  三臂（anchor / loss_level / none）在 64^2 Kolmogorov 2D NS 上训练完毕
  （checkpoints/system_b_fix2/{arm}_best.pt, commit ecef4d8），
  本脚本不做任何再训练，直接在 128^2 真值上 rollout 评估:
    - FNO2d 谱卷积 modes 截断不变，天然支持任意分辨率；
    - anchor 的零参数谱 hard core 用 128^2 波数网格重建
      （lambda(k) 与 PhysicsSpec 从 h5 attrs 读，不硬编码）；
    - 能量包络也在 128^2 上谱算；c_d/c_p 沿用 64^2 训练标定值
      （从 generality_system_b_fix2.json envelope_meta 读 —— 本实验同时是
       "标定跨分辨率迁移" 检验点，报告包络命中率）；
    - 归一化 mean/std 用 128^2 文件自身统计（与 64^2 OOD 场景协议一致:
      被评数据自身统计）。
  评估第四臂 anchor_noenv = anchor 关包络/均值投影（零成本消融）。

指标/口径与主实验 run_system_b_fix2.py 完全一致（fair 口径）:
  物理空间 rL2=|err|/|tgt|、E/E0、Z/Z0（Z 仅诊断）；
  crash: 非有限 或 rL2>5 或 E/E0>5，崩溃后冻结末值填到 K；
  K in {10,50,100,200}；3 IC（3 条轨迹各 1 个）；step_time_ms。

输出: results/paper_experiments/generality_system_b_res_ood.json
      results/logs/system_b_res_ood.log
运行: cd D:/AxiomOS_Project 后
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" run_system_b_res_ood.py
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, '.')
from models.rcln_2d_fix2 import (PhysicsSpec, FNO2dUV, RCLN2DFix2, NSPostproc2D,
                                 ns_lambda)

SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
KS_LIST = [10, 50, 100, 200]
K_MAX = max(KS_LIST)

H5_RES128 = 'data_generated/ns2d_kolmogorov_res128.h5'
CKPT_DIR = 'checkpoints/system_b_fix2'
FIX2_JSON = 'results/paper_experiments/generality_system_b_fix2.json'
OUT_JSON = 'results/paper_experiments/generality_system_b_res_ood.json'

os.makedirs('results/paper_experiments', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)

print('=' * 70)
print(f'  SYSTEM B resolution OOD: 64^2-trained arms zero-shot @128^2 | device={DEV}')
print('=' * 70)

# --------------------------------------------------------------------
# 128^2 数据与 PhysicsSpec（全部从 attrs 读，不硬编码）
# --------------------------------------------------------------------
with h5py.File(H5_RES128, 'r') as f:
    res_fields = f['fields'][:].astype(np.float32)     # [3, 260, 128, 128]
    res_attrs = dict(f.attrs)
SPEC = PhysicsSpec.from_h5_attrs(res_attrs)
N = SPEC.N
DT = SPEC.dt
assert N == 128, f'expect N=128, got {N}'
print(f'res128: {res_fields.shape} attrs: nu={SPEC.nu} A={SPEC.A} kf={SPEC.kf} '
      f'alpha={SPEC.alpha} dt={DT} Re_box={res_attrs["Re_box"]:.0f} '
      f'tau_let={res_attrs["tau_let"]:.2f} K200={res_attrs["K200_cover_tau"]:.2f}tau')

# 归一化: 被评数据（128^2 文件）自身统计 —— 与 64^2 OOD 场景协议一致
MEAN = float(res_fields.mean())
STD = float(res_fields.std()) + 1e-8
print(f'z-score(res128 self): mean={MEAN:.6f} std={STD:.4f}')

# 128^2 谱网格（numpy, 供指标计算）
KXnp = np.fft.fftfreq(N, d=1.0 / N)[:, None]
KYnp = np.fft.rfftfreq(N, d=1.0 / N)[None, :]
K2np = KXnp**2 + KYnp**2
K2inv_np = np.where(K2np > 0, 1.0 / np.maximum(K2np, 1e-30), 0.0)

# --------------------------------------------------------------------
# 包络标定迁移: c_d/c_p 从 64^2 主实验 JSON 读（不重新标定 —— 检验点）
# --------------------------------------------------------------------
fix2 = json.load(open(FIX2_JSON))
C_D = float(fix2['envelope_meta']['c_d'])
C_P = float(fix2['envelope_meta']['c_p'])
print(f'\n[envelope] 沿用 64^2 训练标定: c_d={C_D:.4f} c_p={C_P:.4f} '
      f'(source: {fix2["envelope_meta"]["calibration"]["source"]})')

# --------------------------------------------------------------------
# 模型: 在 N=128 构建（谱 hard core / UV 通道 / 包络全部用 128^2 网格重建），
#       再从 64^2 checkpoint 加载可迁移参数（形状匹配才加载 —— buffer 除外）
# --------------------------------------------------------------------
def build_arm(name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if name == 'anchor':
        return RCLN2DFix2(SPEC, MEAN, STD, modes=12, width=32, layers=4)
    return FNO2dUV(N, modes=12, width=32, layers=4)


def load_cross_resolution(model, ckpt_path, arm):
    """只加载形状匹配的键（nn.Parameter 全部匹配；KX/KY/K2inv/prop/fterm 等
    分辨率相关 buffer 已在 N=128 构建时正确重建，跳过）。"""
    sd = torch.load(ckpt_path, map_location=DEV)['model_state_dict']
    msd = model.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            msd[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(msd)
    # 完整性断言: 所有可训练参数必须被加载
    param_keys = {k for k, _ in model.named_parameters()}
    assert param_keys.issubset(set(loaded)), \
        f'[{arm}] 有参数未加载: {param_keys - set(loaded)}'
    print(f'  [{arm}] loaded {len(loaded)} tensors, rebuilt @128^2 buffers: '
          f'{sorted(skipped)}')
    return model.eval()


models = {}
for arm in ['anchor', 'loss_level', 'none']:
    m = build_arm(arm).to(DEV)
    load_cross_resolution(m, f'{CKPT_DIR}/{arm}_best.pt', arm)
    models[arm] = m
n_params = sum(p.numel() for p in models['anchor'].parameters())
print(f'params/arm = {n_params:,} (与 64^2 完全一致 —— 谱卷积参数与分辨率无关)')

# --------------------------------------------------------------------
# hard core 128^2 重建核验: prop 应等于生成器线性算子 exp(lam*dt_s)^ns
# --------------------------------------------------------------------
lam_ref, _ = ns_lambda(SPEC)
ns = int(res_attrs['sample_every'])
dt_s = float(res_attrs['dt_solver'])
prop_ref = np.exp(lam_ref * dt_s) ** ns
prop_model = models['anchor'].hard.prop.cpu().numpy().astype(np.float64)
prop_err = float(np.max(np.abs(prop_model - prop_ref)))
# 强迫项对照（几何级数 vs phi1 闭式）
geo = dt_s * (np.exp(lam_ref * dt_s * ns) - 1.0) / np.where(
    np.abs(lam_ref) > 1e-14, np.expm1(lam_ref * dt_s), 1.0)
geo = np.where(np.abs(lam_ref) > 1e-14, geo, ns * dt_s)
fhat_phys = np.fft.rfft2((-SPEC.A * SPEC.kf *
                          np.cos(SPEC.kf * np.arange(N) * (2 * np.pi / N)))[None, :]
                         * np.ones((N, 1)))
fterm_ref = geo * fhat_phys / STD / N
fterm_model = models['anchor'].hard.fterm.cpu().numpy()
fterm_err = float(np.max(np.abs(fterm_model - fterm_ref)) /
                  (np.max(np.abs(fterm_ref)) + 1e-30))
print(f'\n[sanity] hard-core @128^2 prop vs generator Lop: max|err|={prop_err:.3e}')
print(f'[sanity] forcing term vs solver geometric series: rel max err={fterm_err:.3e}')
if prop_err > 1e-6 or fterm_err > 1e-3:
    print('[sanity] *** 失败：128^2 hard core 重建与生成器不一致！ ***')
    sys.exit(1)
print('[sanity] OK: 128^2 hard core 零参数重建正确（lambda 从 attrs 读）')

# --------------------------------------------------------------------
# 评估（fair 口径，与 run_system_b_fix2.py 逐行同构，仅 N 参数化）
# --------------------------------------------------------------------
def rollout_eval(model, trajs, mean, std, n_ic, arm_name, scn_name):
    n_traj = len(trajs)
    base, rem = divmod(n_ic, n_traj)
    ic_counts = [base + (1 if j < rem else 0) for j in range(n_traj)]
    per_ic = []
    step_times = []
    with torch.no_grad():
        for j in range(n_traj):
            fields = trajs[j]
            nj = ic_counts[j]
            if nj == 0:
                continue
            sp = max(1, (len(fields) - K_MAX - 2) // nj)
            for ic in range(nj):
                idx = min(ic * sp, len(fields) - K_MAX - 2)
                ic_phys = fields[idx].astype(np.float64)
                what0 = np.fft.rfft2(ic_phys)
                psi0 = -what0 * K2inv_np
                u0p = np.fft.irfft2(-1j * KYnp * psi0, s=(N, N))
                v0p = np.fft.irfft2(1j * KXnp * psi0, s=(N, N))
                E0 = 0.5 * np.mean(u0p**2 + v0p**2) + 1e-12
                Z0 = 0.5 * np.mean(ic_phys**2) + 1e-12
                u = torch.from_numpy(((fields[idx] - mean) / std)
                                     .astype(np.float32)).view(1, 1, N, N).to(DEV)
                recs = []
                crash_step = None
                last = (5.0, 5.0, 5.0)
                for step in range(K_MAX):
                    if DEV == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pred = model(u)
                    if DEV == 'cuda':
                        torch.cuda.synchronize()
                    step_times.append((time.perf_counter() - t0) * 1e3)
                    p_phys = (pred.cpu().numpy().reshape(N, N).astype(np.float64)
                              * std + mean)
                    t_phys = fields[idx + step + 1].astype(np.float64)
                    rl2 = float(np.linalg.norm(p_phys - t_phys)
                                / (np.linalg.norm(t_phys) + 1e-12))
                    whatp = np.fft.rfft2(p_phys)
                    psip = -whatp * K2inv_np
                    up = np.fft.irfft2(-1j * KYnp * psip, s=(N, N))
                    vp = np.fft.irfft2(1j * KXnp * psip, s=(N, N))
                    er = float(0.5 * np.mean(up**2 + vp**2) / E0)
                    zr = float(0.5 * np.mean(p_phys**2) / Z0)
                    crashed = (not np.isfinite(rl2)) or (not np.isfinite(er)) \
                        or (not np.isfinite(zr)) or rl2 > 5.0 or er > 5.0
                    if crashed:
                        crash_step = step + 1
                        if np.isfinite(rl2) and np.isfinite(er) and np.isfinite(zr):
                            last = (rl2, er, zr)
                        recs.append(last)
                        break
                    last = (rl2, er, zr)
                    recs.append(last)
                    u = pred
                while len(recs) < K_MAX:
                    recs.append(last)
                per_ic.append({'recs': recs, 'crash_step': crash_step})
                tag = f'crash@{crash_step}' if crash_step else 'ok'
                print(f'    [{scn_name}/{arm_name}] traj{j} IC{ic} idx={idx}: {tag} '
                      f'rL2@10={recs[9][0]:.3f} rL2@100={recs[99][0]:.3f} '
                      f'E@100={recs[99][1]:.3f} Z@100={recs[99][2]:.3f}', flush=True)
    out = {}
    for K in KS_LIST:
        rl2s = np.array([p['recs'][K - 1][0] for p in per_ic])
        ers = np.array([p['recs'][K - 1][1] for p in per_ic])
        zrs = np.array([p['recs'][K - 1][2] for p in per_ic])
        crashes = [p['crash_step'] for p in per_ic
                   if p['crash_step'] is not None and p['crash_step'] <= K]
        out[str(K)] = {
            'rL2_mean': float(rl2s.mean()), 'rL2_std': float(rl2s.std()),
            'E_ratio_mean': float(ers.mean()), 'E_ratio_std': float(ers.std()),
            'Z_ratio_mean': float(zrs.mean()), 'Z_ratio_std': float(zrs.std()),
            'crash_rate': len(crashes) / max(1, len(per_ic)),
            'first_crash_steps': [p['crash_step'] for p in per_ic],
            'n_ic': len(per_ic),
            'step_time_ms': float(np.mean(step_times)),
        }
    return out


anchor_env = NSPostproc2D(models['anchor'], SPEC, MEAN, STD,
                          c_d=C_D, c_p=C_P,
                          mean_project=True, envelope=True).to(DEV)

print(f'\n--- 评估: res_ood_128 (trajs=3, n_ic=3, zero-shot, '
      f'mean={MEAN:.4f} std={STD:.4f}, spec={SPEC}) ---')
anchor_env.reset_stats()
results = {}
for arm_name, model in [('anchor', anchor_env),
                        ('anchor_noenv', models['anchor']),
                        ('loss_level', models['loss_level']),
                        ('none', models['none'])]:
    results[arm_name] = rollout_eval(model, res_fields, MEAN, STD, 3,
                                     arm_name, 'res_ood_128')
env_stats = anchor_env.env_stats()
print(f'  [envelope/res_ood_128] floor_hit={env_stats["floor_hit_rate"]:.3f} '
      f'ceil_hit={env_stats["ceil_hit_rate"]:.3f} '
      f'mean|corr|={env_stats["mean_abs_corr"]:.4f} n={env_stats["n_steps"]}')

# --------------------------------------------------------------------
# JSON 输出（同主实验结构 + 64^2 in_domain 对照块）
# --------------------------------------------------------------------
out_json = {
    'protocol': {
        'experiment': ('NC System B Step6: resolution OOD —— 64^2-trained arms '
                       'zero-shot rollout on 128^2 2D Navier-Stokes '
                       '(FNO-style super-resolution, no retraining)'),
        'equation': ('w_t + J(psi,w) = nu*Lap(w) + f_w - alpha*Lap^-1(w); '
                     'psi^=-w^/k^2; u=-psi_y,v=psi_x; '
                     'f=A*sin(k_f*y)*x-hat (contour-ETDRK4 verified)'),
        'data': {'res128': H5_RES128,
                 'res128_attrs': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                  for k, v in res_attrs.items()},
                 'train_checkpoints': f'{CKPT_DIR}/{{arm}}_best.pt (64^2, commit ecef4d8)'},
        'normalization': ('res128 file self-statistics z-score (same protocol as '
                          '64^2 OOD scenarios: evaluated-data statistics)'),
        'cross_resolution_mechanics': {
            'fno': 'SpectralConv2d modes=12 truncation unchanged; '
                   'all weights resolution-independent, loaded verbatim',
            'hard_core': 'rebuilt on 128^2 wavenumber grid; lambda(k) & PhysicsSpec '
                         'from h5 attrs, no hardcoding; prop/forcing sanity-checked '
                         'against 128^2 generator linear operator',
            'envelope': 'c_d/c_p carried over from 64^2 training calibration '
                        '(calibration transfer across resolution is part of the test); '
                        'envelope computed spectrally on 128^2 grid',
            'loaded_vs_rebuilt': 'all nn.Parameters loaded from 64^2 ckpt; '
                                 'resolution-dependent buffers (KX/KY/K2inv/prop/fterm) '
                                 'rebuilt at N=128',
        },
        'eval': ('physical-space rL2 & E/E0 & Z/Z0 (Z diagnostic only, not enveloped); '
                 'crash: non-finite or rL2>5 or E/E0>5; frozen after crash; '
                 'K in {10,50,100,200}; 3 ICs (1 per trajectory); fair protocol '
                 'identical to run_system_b_fix2.py'),
        'arms': {'params_per_arm': n_params,
                 'source_64_training': fix2['protocol']['arms']},
        'hardcore_sanity_128': {'prop_max_err': prop_err,
                                'forcing_rel_err': fterm_err},
        'device': DEV,
    },
    'envelope_meta': {
        'c_d': C_D, 'c_p': C_P,
        'calibration': dict(fix2['envelope_meta']['calibration'],
                            note='carried over from 64^2 training; NOT recalibrated '
                                 'on 128^2 (calibration-transfer test)'),
        'per_scenario': {'res_ood_128': env_stats},
    },
    'res_ood_128': results,
    'baseline_64_in_domain': {
        arm: fix2['in_domain'][arm] for arm in
        ['anchor', 'anchor_noenv', 'loss_level', 'none']
    },
}
with open(OUT_JSON, 'w') as f:
    json.dump(out_json, f, indent=2)
print(f'\nSaved: {OUT_JSON}')

# --------------------------------------------------------------------
# 汇总表（128^2 zero-shot vs 64^2 in-domain 并列）
# --------------------------------------------------------------------
print('\n' + '=' * 96)
print('  SUMMARY: 128^2 zero-shot (res_ood_128) vs 64^2 in-domain | K=200 | K=10')
print('=' * 96)
for scn, block in [('res_ood_128 (zero-shot)', results),
                   ('64^2 in_domain (baseline)', fix2['in_domain'])]:
    print(f'\n[{scn}]')
    print(f'{"arm":<14s} {"rL2@200":>16s} {"E/E0@200":>16s} {"Z/Z0@200":>16s} '
          f'{"crash@200":>9s} {"rL2@10":>16s} {"ms/step":>8s}')
    for arm in ['anchor', 'anchor_noenv', 'loss_level', 'none']:
        d200 = block[arm]['200']
        d10 = block[arm]['10']
        print(f'{arm:<14s} '
              f'{d200["rL2_mean"]:>7.3f}+-{d200["rL2_std"]:<7.3f} '
              f'{d200["E_ratio_mean"]:>7.3f}+-{d200["E_ratio_std"]:<7.3f} '
              f'{d200["Z_ratio_mean"]:>7.3f}+-{d200["Z_ratio_std"]:<7.3f} '
              f'{d200["crash_rate"]:>9.2f} '
              f'{d10["rL2_mean"]:>7.3f}+-{d10["rL2_std"]:<7.3f} '
              f'{d200["step_time_ms"]:>8.2f}')
print('\nALL DONE.')

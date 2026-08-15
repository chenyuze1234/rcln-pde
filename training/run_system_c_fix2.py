"""
System C fix2 —— KS 1D 跨系统 Go/No-Go（NC 任务书 System C，fair 口径）
================================================================
三臂（同一 FNO1d: modes=16, width=64, 4 层；参数量对齐 287,361）：
  anchor     : pred = LinearExact(u) + FNO1d(u)，端到端训练；推理路径加
               均值投影 + 双边能量包络（c_d/c_p 由 L32 训练段真值标定）。
  loss_level : 纯 FNO1d + 物理残差损失 lambda_phys=0.1（谱 RHS，2/3 去混叠）。
  none       : 纯 FNO1d，纯 MSE。
评估第四臂 anchor_noenv = anchor 关包络/均值投影（零训练成本消融）。

数据（全部标准 KS，轮廓积分 ETDRK4 生成，一步一致性 relerr~3e-8）：
  训练+in_domain : data_generated/ks_L32_N64_T30.0_dt0.01.h5（前80%训练/后20%验证）
  length OOD     : ks_L44_N64_T40.0_dt0.01.h5 / ks_L22_N64_T30.0_dt0.01.h5
                   （各自用自身 mean/std 重归一化；hard core/包络的 L 同步切换）

训练协议（三臂一致）：z-score(训练段统计)，AdamW lr=1e-3 + cosine，
  50 epochs（smoke 可改），batch 64，grad clip 1.0，seed 42，
  rollout R=3, gamma=0.9: loss = sum gamma^k * mse(pred_k, target_k)。

评估（fair 口径，对齐 fix2 套件）：物理空间指标；rL2=|err|/|tgt|；
  E/E0 相对 IC；crash: 非有限 或 rL2>5 或 E/E0>5，崩溃后冻结末值填到 K。
  K ∈ {10,50,100,200}；IC 等间隔 sp=(len-Kmax-1)//n_ic。
输出：results/paper_experiments/generality_system_c_fix2.json
      checkpoints/system_c_fix2/{arm}_best.pt + {arm}_history.json
运行：cd D:/AxiomOS_Project 后
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" run_system_c_fix2.py [--epochs 50]
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from models.rcln_1d_fix2 import (FNO1d, RCLN1DFix2, KSPostproc1D,
                                 LinearExactKS, rhs_ks_torch)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch', type=int, default=64)
args = parser.parse_args()

SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 64
L_TRAIN = 32.0
DT = 0.01
R_ROLL = 3
GAMMA = 0.9
LAMBDA_PHYS = 0.1
KS_LIST = [10, 50, 100, 200]
K_MAX = max(KS_LIST)

H5_L32 = 'data_generated/ks_L32_N64_T30.0_dt0.01.h5'
H5_L44 = 'data_generated/ks_L44_N64_T40.0_dt0.01.h5'
H5_L22 = 'data_generated/ks_L22_N64_T30.0_dt0.01.h5'

CKPT_DIR = 'checkpoints/system_c_fix2'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs('results/paper_experiments', exist_ok=True)

print('=' * 70)
print(f'  SYSTEM C fix2: KS Go/No-Go | device={DEV} epochs={args.epochs}')
print('=' * 70)

# --------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------
with h5py.File(H5_L32, 'r') as f:
    ks32 = f['fields'][:].astype(np.float32)
n_frames = len(ks32)
n_train_f = int(n_frames * 0.8)
train_seg = ks32[:n_train_f]
val_seg = ks32[n_train_f:]
MEAN = float(train_seg.mean())
STD = float(train_seg.std()) + 1e-8
print(f'L32: {ks32.shape} train_frames={n_train_f} val_frames={n_frames-n_train_f} '
      f'mean={MEAN:.4f} std={STD:.4f}')

def norm(a):
    return ((a - MEAN) / STD).astype(np.float32)

tr_n = norm(train_seg)
va_n = norm(val_seg)

def make_roll(seg):
    """相邻帧 rollout R=3 数据集: u0, t1, t2, t3"""
    return (torch.from_numpy(seg[:-3]).unsqueeze(1),
            torch.from_numpy(seg[1:-2]).unsqueeze(1),
            torch.from_numpy(seg[2:-1]).unsqueeze(1),
            torch.from_numpy(seg[3:]).unsqueeze(1))

TR = make_roll(tr_n)
VA = make_roll(va_n)
print(f'rollout pairs: train={len(TR[0])} val={len(VA[0])}')

# --------------------------------------------------------------------
# Hard core 符号/尺度验证
# 注意：在统计平稳的混沌吸引子上 |du/dt| 远小于 |线性项|~|非线性项|
# （两者近似相消），"hard-only 优于恒等映射" 在该 regime 不可能成立——
# 高阶模态被 NL 强迫为准稳态（slaved），恒等映射本来就极强。真正的风险是
# lambda 的符号/尺度实现错误，故直接对照生成器的线性算子：
#   生成器 Lop = -k^2 - k^4 (k=2j*pi*fftfreq)，求解器 dt=0.0005，
#   帧 dt=0.01 = 20 个求解步 -> prop 应等于 E_solver^20（rfft 布局）。
# --------------------------------------------------------------------
hard_check = LinearExactKS(N, L_TRAIN, DT).to(DEV)
k_gen = 2j * np.pi * np.fft.fftfreq(N, d=L_TRAIN / N)
Lop_gen = -k_gen**2 - k_gen**4
E_solver = np.exp(0.0005 * Lop_gen) ** 20          # 帧级线性传播子（fft 布局）
prop_ref = E_solver[:N // 2 + 1].real              # rfft 布局（Lop 为实数）
prop_model = hard_check.prop.cpu().numpy()
prop_err = float(np.max(np.abs(prop_model - prop_ref)))
with torch.no_grad():
    u0 = VA[0].to(DEV); t1 = VA[1].to(DEV)
    mse_hard = F.mse_loss(hard_check(u0), t1).item()
    mse_id = F.mse_loss(u0, t1).item()
print(f'\n[sanity] hard-core prop vs generator Lop: max|err|={prop_err:.3e}')
print(f'[sanity] (参考) hard-only 1-step MSE={mse_hard:.3e} vs identity={mse_id:.3e} '
      f'— 平稳吸引子上 NL/线性项相消，hard-only 不预期优于恒等；'
      f'Soft 学习的是非线性增量，端到端 MSE 会驱动二者配合')
if prop_err > 1e-6:   # 阈值放宽至 float32 精度之外（buffer 为 float32）
    print('[sanity] *** 失败：hard core 传播子与生成器线性算子不一致！ ***')
    sys.exit(1)
print('[sanity] OK: hard core 传播子与生成器线性部分完全一致（符号/尺度正确）')

# --------------------------------------------------------------------
# 能量包络标定（L32 训练段真值）
# --------------------------------------------------------------------
print('\n[envelope] 标定 c_d / c_p（L32 训练段真值，p99.5 x 1.2）')
tr_phys = train_seg.astype(np.float64)
kap = 2.0 * np.pi * np.fft.rfftfreq(N, d=L_TRAIN / N)
uh = np.fft.rfft(tr_phys, axis=1)
u_x = np.fft.irfft(1j * kap[None, :] * uh, n=N, axis=1)
u_xx = np.fft.irfft(-kap[None, :]**2 * uh, n=N, axis=1)
P_t = (u_x**2).mean(axis=1)        # 产生率
D_t = (u_xx**2).mean(axis=1)       # 耗散率
E_t = 0.5 * (tr_phys**2).mean(axis=1)
dE = np.diff(E_t)
pos = dE > 0
ratio_p = dE[pos] / (P_t[:-1][pos] * DT)
ratio_d = -dE[~pos] / (D_t[:-1][~pos] * DT)
C_P = float(1.2 * np.percentile(ratio_p, 99.5))
C_D = float(1.2 * np.percentile(ratio_d, 99.5))
print(f'  dE>0 占比={pos.mean():.3f}  p99.5(dE/(P dt))={np.percentile(ratio_p,99.5):.4f} -> c_p={C_P:.4f}')
print(f'  dE<0 占比={(~pos).mean():.3f}  p99.5(-dE/(D dt))={np.percentile(ratio_d,99.5):.4f} -> c_d={C_D:.4f}')

# --------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------
def build_arm(name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if name == 'anchor':
        return RCLN1DFix2(N, L_TRAIN, DT, modes=16, width=64, layers=4)
    return FNO1d(modes=16, width=64, layers=4)

def arm_loss(model, u0, t1, t2, t3, phys):
    p1 = model(u0)
    p2 = model(p1)
    p3 = model(p2)
    loss = F.mse_loss(p1, t1) + GAMMA * F.mse_loss(p2, t2) + GAMMA**2 * F.mse_loss(p3, t3)
    if phys:
        def res(pred, u_in):
            u_phys = u_in * STD + MEAN
            return (pred - u_in) / DT - rhs_ks_torch(u_phys, L_TRAIN) / STD
        loss = loss + LAMBDA_PHYS * (
            (res(p1, u0)**2).mean()
            + GAMMA * (res(p2, p1)**2).mean()
            + GAMMA**2 * (res(p3, p2)**2).mean())
    return loss

def train_arm(name, phys):
    model = build_arm(name).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    hist = {'train_loss': [], 'val_loss': [], 'lr': []}
    best = float('inf')
    t_start = time.time()
    n_tr = len(TR[0])
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_tr)
        tl = 0.0
        for i in range(0, n_tr, args.batch):
            bi = perm[i:i + args.batch]
            u0, t1, t2, t3 = [x[bi].to(DEV) for x in TR]
            opt.zero_grad()
            loss = arm_loss(model, u0, t1, t2, t3, phys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(bi)
        tl /= n_tr
        sch.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for i in range(0, len(VA[0]), args.batch):
                u0, t1, t2, t3 = [x[i:i + args.batch].to(DEV) for x in VA]
                vl += arm_loss(model, u0, t1, t2, t3, phys).item() * len(u0)
        vl /= len(VA[0])
        hist['train_loss'].append(tl)
        hist['val_loss'].append(vl)
        hist['lr'].append(sch.get_last_lr()[0])
        star = ''
        if vl < best:
            best = vl
            star = '*'
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': vl,
                        'arm': name, 'epoch': ep + 1},
                       f'{CKPT_DIR}/{name}_best.pt')
        print(f'[SYS-C] {name:10s} ep{ep+1:3d}: tr={tl:.6f} val={vl:.6f}{star} '
              f'lr={sch.get_last_lr()[0]:.2e} {time.time()-t_start:.0f}s', flush=True)
    train_time = time.time() - t_start
    with open(f'{CKPT_DIR}/{name}_history.json', 'w') as f:
        json.dump(hist, f)
    print(f'[SYS-C] {name} DONE. best val={best:.6f} params={n_params:,} '
          f'train_time={train_time:.0f}s', flush=True)
    # 重载 best
    model.load_state_dict(torch.load(f'{CKPT_DIR}/{name}_best.pt',
                                     map_location=DEV)['model_state_dict'])
    model.eval()
    return model, n_params, train_time, best

print('\n--- 训练三臂 ---')
t_all = time.time()
models = {}
arm_meta = {}
models['anchor'], p_a, tt_a, bv_a = train_arm('anchor', phys=False)
arm_meta['anchor'] = {'params': p_a, 'train_time_s': tt_a, 'best_val': bv_a}
models['loss_level'], p_l, tt_l, bv_l = train_arm('loss_level', phys=True)
arm_meta['loss_level'] = {'params': p_l, 'train_time_s': tt_l, 'best_val': bv_l}
models['none'], p_n, tt_n, bv_n = train_arm('none', phys=False)
arm_meta['none'] = {'params': p_n, 'train_time_s': tt_n, 'best_val': bv_n}
print(f'--- 训练全部完成，总耗时 {time.time()-t_all:.0f}s ---')

# --------------------------------------------------------------------
# 评估（fair 口径）
# --------------------------------------------------------------------
def rollout_eval(model, fields, L, mean, std, n_ic, arm_name, scn_name):
    """物理空间指标 + crash 冻结。返回 per-K 聚合。"""
    sp = max(1, (len(fields) - K_MAX - 1) // n_ic)
    per_ic = []
    step_times = []
    with torch.no_grad():
        for ic in range(n_ic):
            idx = min(ic * sp, len(fields) - K_MAX - 2)
            ic_phys = fields[idx].astype(np.float64)
            E0 = 0.5 * (ic_phys**2).mean() + 1e-12
            u = torch.from_numpy(((fields[idx] - mean) / std)
                                 .astype(np.float32)).view(1, 1, N).to(DEV)
            recs = []          # 每步 (rL2, E_ratio)
            crash_step = None
            last_rl2, last_er = 5.0, 5.0
            for step in range(K_MAX):
                if DEV == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = model(u)
                if DEV == 'cuda':
                    torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t0) * 1e3)
                p_phys = (pred.cpu().numpy().reshape(-1).astype(np.float64)
                          * std + mean)
                t_phys = fields[idx + step + 1].astype(np.float64)
                rl2 = float(np.linalg.norm(p_phys - t_phys)
                            / (np.linalg.norm(t_phys) + 1e-12))
                er = float(0.5 * (p_phys**2).mean() / E0)
                crashed = (not np.isfinite(rl2)) or (not np.isfinite(er)) \
                    or rl2 > 5.0 or er > 5.0
                if crashed:
                    crash_step = step + 1
                    if np.isfinite(rl2) and np.isfinite(er):
                        last_rl2, last_er = rl2, er
                    recs.append((last_rl2, last_er))
                    break
                last_rl2, last_er = rl2, er
                recs.append((rl2, er))
                u = pred
            # 冻结末值填到 K_MAX
            while len(recs) < K_MAX:
                recs.append((last_rl2, last_er))
            per_ic.append({'recs': recs, 'crash_step': crash_step})
            tag = f'crash@{crash_step}' if crash_step else 'ok'
            print(f'    [{scn_name}/{arm_name}] IC{ic} idx={idx}: {tag} '
                  f'rL2@10={recs[9][0]:.3f} rL2@100={recs[99][0]:.3f} '
                  f'E@100={recs[99][1]:.3f}', flush=True)
    out = {}
    for K in KS_LIST:
        rl2s = np.array([p['recs'][K - 1][0] for p in per_ic])
        ers = np.array([p['recs'][K - 1][1] for p in per_ic])
        crashes = [p['crash_step'] for p in per_ic
                   if p['crash_step'] is not None and p['crash_step'] <= K]
        out[str(K)] = {
            'rL2_mean': float(rl2s.mean()), 'rL2_std': float(rl2s.std()),
            'E_ratio_mean': float(ers.mean()), 'E_ratio_std': float(ers.std()),
            'crash_rate': len(crashes) / n_ic,
            'first_crash_steps': [p['crash_step'] for p in per_ic],
            'n_ic': n_ic,
            'step_time_ms': float(np.mean(step_times)),
        }
    return out

# postproc wrapper（anchor 专用；envelope 统计按场景记录）
anchor_env = KSPostproc1D(models['anchor'], MEAN, STD, N, L_TRAIN, DT,
                          c_d=C_D, c_p=C_P,
                          mean_project=True, envelope=True).to(DEV)

def eval_scenario(scn_name, fields, L, n_ic, mean, std):
    print(f'\n--- 评估: {scn_name} (L={L}, frames={len(fields)}, n_ic={n_ic}, '
          f'mean={mean:.4f} std={std:.4f}) ---')
    anchor_env.set_L(L)
    anchor_env.set_norm(mean, std)
    anchor_env.reset_stats()
    arms = [('anchor', anchor_env),
            ('anchor_noenv', models['anchor']),
            ('loss_level', models['loss_level']),
            ('none', models['none'])]
    res = {}
    for arm_name, model in arms:
        res[arm_name] = rollout_eval(model, fields, L, mean, std, n_ic,
                                     arm_name, scn_name)
    env_stats = anchor_env.env_stats()
    print(f'  [envelope/{scn_name}] floor_hit={env_stats["floor_hit_rate"]:.3f} '
          f'ceil_hit={env_stats["ceil_hit_rate"]:.3f} '
          f'mean|corr|={env_stats["mean_abs_corr"]:.4f} '
          f'n={env_stats["n_steps"]}')
    return res, env_stats

results = {}
envelope_meta = {
    'c_d': C_D, 'c_p': C_P,
    'calibration': {
        'source': 'L32 train-segment truth, p99.5 x 1.2',
        'p99_5_dE_over_Pdt': float(np.percentile(ratio_p, 99.5)),
        'p99_5_negdE_over_Ddt': float(np.percentile(ratio_d, 99.5)),
        'frac_dE_positive': float(pos.mean()),
        'dt': DT,
    },
    'per_scenario': {},
}

# in_domain: L32 验证段，用训练段归一化统计
res, es = eval_scenario('in_domain', val_seg, 32.0, 5, MEAN, STD)
results['in_domain'] = res
envelope_meta['per_scenario']['in_domain'] = es

# length OOD: L44 / L22，各自重算 mean/std
with h5py.File(H5_L44, 'r') as f:
    ks44 = f['fields'][:].astype(np.float32)
m44, s44 = float(ks44.mean()), float(ks44.std()) + 1e-8
res, es = eval_scenario('length_ood_L44', ks44, 44.0, 3, m44, s44)
results['length_ood_L44'] = res
envelope_meta['per_scenario']['length_ood_L44'] = es

with h5py.File(H5_L22, 'r') as f:
    ks22 = f['fields'][:].astype(np.float32)
m22, s22 = float(ks22.mean()), float(ks22.std()) + 1e-8
res, es = eval_scenario('length_ood_L22', ks22, 22.0, 3, m22, s22)
results['length_ood_L22'] = res
envelope_meta['per_scenario']['length_ood_L22'] = es

# --------------------------------------------------------------------
# JSON 输出
# --------------------------------------------------------------------
out_json = {
    'protocol': {
        'experiment': 'NC System C: KS 1D cross-system Go/No-Go (fix2)',
        'equation': 'u_t = -u*u_x - u_xx - u_xxxx (standard KS, contour-ETDRK4 verified)',
        'data': {'train': H5_L32, 'ood_L44': H5_L44, 'ood_L22': H5_L22},
        'train_split': 'L32 first 80% frames train / last 20% val, z-score(train stats)',
        'optimizer': f'AdamW lr={args.lr} cosine, epochs={args.epochs}, '
                     f'batch={args.batch}, grad_clip=1.0, seed={SEED}',
        'rollout_training': f'R={R_ROLL}, gamma={GAMMA}, '
                            'loss=sum gamma^k mse(pred_k,target_k) autoregressive',
        'lambda_phys': LAMBDA_PHYS,
        'eval': 'physical-space rL2 & E/E0; crash: non-finite or rL2>5 or E/E0>5; '
                'frozen after crash; K in {10,50,100,200}; '
                'IC spacing sp=(len-Kmax-1)//n_ic',
        'arms': arm_meta,
        'hardcore_sanity': {'mse_hard_1step': mse_hard, 'mse_identity': mse_id},
        'device': DEV,
    },
    'envelope_meta': envelope_meta,
    'in_domain': results['in_domain'],
    'length_ood_L44': results['length_ood_L44'],
    'length_ood_L22': results['length_ood_L22'],
}
out_path = 'results/paper_experiments/generality_system_c_fix2.json'
with open(out_path, 'w') as f:
    json.dump(out_json, f, indent=2)
print(f'\nSaved: {out_path}')

# --------------------------------------------------------------------
# 汇总表
# --------------------------------------------------------------------
print('\n' + '=' * 78)
print('  SUMMARY (K=200 | K=10)')
print('=' * 78)
for scn in ['in_domain', 'length_ood_L44', 'length_ood_L22']:
    print(f'\n[{scn}]')
    print(f'{"arm":<14s} {"rL2@200":>16s} {"E/E0@200":>16s} {"crash@200":>9s} '
          f'{"rL2@10":>16s} {"ms/step":>8s}')
    for arm in ['anchor', 'anchor_noenv', 'loss_level', 'none']:
        d200 = results[scn][arm]['200']
        d10 = results[scn][arm]['10']
        print(f'{arm:<14s} '
              f'{d200["rL2_mean"]:>7.3f}±{d200["rL2_std"]:<7.3f} '
              f'{d200["E_ratio_mean"]:>7.3f}±{d200["E_ratio_std"]:<7.3f} '
              f'{d200["crash_rate"]:>9.2f} '
              f'{d10["rL2_mean"]:>7.3f}±{d10["rL2_std"]:<7.3f} '
              f'{d200["step_time_ms"]:>8.2f}')
print('\nALL DONE.')

"""
System B fix2 —— 2D Navier-Stokes 跨系统 Go/No-Go（NC 任务书 System B，fair 口径）
================================================================
与 run_system_c_fix2.py (KS / System C) 严格同构，便于跨系统比较。
三臂（同一 FNO2d: modes=12, width=32, 4 层, 3 输入通道 (w,u,v)；参数量对齐）:
  anchor     : pred = LinearExact(w) + FNO2d(w,u,v)，端到端训练；推理路径加
               均值投影 + 双边能量包络（c_d/c_p 由训练真值 p99.5x1.2 标定）。
               hard core: w^_{t+1} = e^{lam*dt} w^_t + phi1(lam*dt) f^/std,
               lam=-(nu*k^2+alpha/k^2)；包络 E∈[E-c_d·D·dt, E+c_p·P_f·dt],
               D=2nu*Z+alpha<psi^2>, P_f=<u·f>（解析强迫）。
  loss_level : 纯 FNO2d + 物理残差损失 lambda_phys=0.1（谱 RHS，2/3 去混叠）。
  none       : 纯 FNO2d，纯 MSE。
评估第四臂 anchor_noenv = anchor 关包络/均值投影（零训练成本消融）。

数据（轮廓积分 ETDRK4 生成，验证门见 gen_2d_ns_data.log）:
  训练/验证/测试 : data_generated/ns2d_kolmogorov_main.h5（14 条独立 IC 轨迹，
                   10 训 / 2 验 / 2 测，各 260 帧）
  Re OOD         : ns2d_kolmogorov_re_ood.h5（nu 减半，Re x2，3 条）
  decaying OOD   : ns2d_decaying_hit.h5（f=0, alpha=0，3 条；
                   anchor 的 PhysicsSpec 同步切换 —— 同架构原则不同 PhysicsSpec）
OOD 场景各自用自身 mean/std 重归一化；hard core/包络参数从 h5 attrs 读，不硬编码。

训练协议（三臂一致）：z-score(训练轨迹统计)，AdamW lr=1e-3 + cosine，
  50 epochs，batch 16，grad clip 1.0，seed 42，rollout R=3 gamma=0.9。

评估（fair 口径，对齐 fix2 套件）：物理空间指标；rL2=|err|/|tgt|；
  E/E0、Z/Z0 相对 IC（Z 仅诊断，不包络）；
  crash: 非有限 或 rL2>5 或 E/E0>5，崩溃后冻结末值填到 K。
  K ∈ {10,50,100,200}；in_domain 5 IC（2 条测试轨迹），OOD 各 3 IC。
输出：results/paper_experiments/generality_system_b_fix2.json
      checkpoints/system_b_fix2/{arm}_best.pt + {arm}_history.json
      results/logs/system_b_fix2.log
运行：cd D:/AxiomOS_Project 后
  "C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe" run_system_b_fix2.py [--epochs 50]
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from models.rcln_2d_fix2 import (PhysicsSpec, FNO2dUV, RCLN2DFix2, NSPostproc2D,
                                 NSRHS2D, ns_lambda, ns_wavenumbers)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch', type=int, default=16)
args = parser.parse_args()

SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 64
R_ROLL = 3
GAMMA = 0.9
LAMBDA_PHYS = 0.1
KS_LIST = [10, 50, 100, 200]
K_MAX = max(KS_LIST)

H5_MAIN = 'data_generated/ns2d_kolmogorov_main.h5'
H5_OOD = 'data_generated/ns2d_kolmogorov_re_ood.h5'
H5_DEC = 'data_generated/ns2d_decaying_hit.h5'

CKPT_DIR = 'checkpoints/system_b_fix2'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs('results/paper_experiments', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)

print('=' * 70)
print(f'  SYSTEM B fix2: 2D NS Go/No-Go | device={DEV} epochs={args.epochs}')
print('=' * 70)

# --------------------------------------------------------------------
# 数据（PhysicsSpec 从 attrs 读，不硬编码）
# --------------------------------------------------------------------
with h5py.File(H5_MAIN, 'r') as f:
    main_fields = f['fields'][:].astype(np.float32)      # [14, 260, 64, 64]
    main_attrs = dict(f.attrs)
SPEC = PhysicsSpec.from_h5_attrs(main_attrs)
DT = SPEC.dt
print(f'main: {main_fields.shape} attrs: nu={SPEC.nu} A={SPEC.A} kf={SPEC.kf} '
      f'alpha={SPEC.alpha} dt={DT} Re_box={main_attrs["Re_box"]:.0f} '
      f'tau_let={main_attrs["tau_let"]:.2f} K200={main_attrs["K200_cover_tau"]:.2f}tau')

train_traj = main_fields[:10]
val_traj = main_fields[10:12]
test_traj = main_fields[12:14]
MEAN = float(train_traj.mean())
STD = float(train_traj.std()) + 1e-8
print(f'z-score(train): mean={MEAN:.6f} std={STD:.4f}  '
      f'(w 零均值, mean~0 符合预期)')

def norm(a):
    return ((a - MEAN) / STD).astype(np.float32)

def make_roll(trajs):
    """所有轨迹相邻帧 rollout R=3 数据集: u0, t1, t2, t3"""
    u0, t1, t2, t3 = [], [], [], []
    for tr in trajs:
        u0.append(tr[:-3]); t1.append(tr[1:-2]); t2.append(tr[2:-1]); t3.append(tr[3:])
    return [torch.from_numpy(np.concatenate(x)).unsqueeze(1) for x in (u0, t1, t2, t3)]

TR = make_roll(norm(train_traj))
VA = make_roll(norm(val_traj))
print(f'rollout pairs: train={len(TR[0])} val={len(VA[0])}')

# --------------------------------------------------------------------
# Hard core 符号/尺度验证（与生成器线性算子逐项对照）
# 注意（沿用 KS 结论）：平稳吸引子上 NL/线性项近似相消，"hard-only 优于
# 恒等映射"判据不成立，只做传播子逐项对照。
#   生成器 lam = -nu*k^2 - alpha/k^2，求解器 dt=5e-3，
#   帧 dt = sample_every 个求解步 -> prop 应等于 exp(lam*dt_s)^ns（float64）。
#   强迫项应与几何级数 sum_j e^{lam*dt_s*j}*dt_s*f^ 一致到 O(dt_s)。
# --------------------------------------------------------------------
lam_ref, _ = ns_lambda(SPEC)
ns = int(main_attrs['sample_every'])
dt_s = float(main_attrs['dt_solver'])
prop_ref = np.exp(lam_ref * dt_s) ** ns
hard_check = RCLN2DFix2(SPEC, MEAN, STD).hard
prop_model = hard_check.prop.cpu().numpy().astype(np.float64)
prop_err = float(np.max(np.abs(prop_model - prop_ref)))
# 强迫项对照（几何级数 vs phi1 闭式，O(dt_s) 相对容差）
geo = dt_s * (np.exp(lam_ref * dt_s * ns) - 1.0) / np.where(
    np.abs(lam_ref) > 1e-14, np.expm1(lam_ref * dt_s), 1.0)
geo = np.where(np.abs(lam_ref) > 1e-14, geo, ns * dt_s)
fhat_phys = np.fft.rfft2((-SPEC.A * SPEC.kf *
                          np.cos(SPEC.kf * np.arange(N) * (2 * np.pi / N)))[None, :]
                         * np.ones((N, 1)))
fterm_ref = geo * fhat_phys / STD / N       # ortho 约定 + 归一化空间
fterm_model = hard_check.fterm.cpu().numpy()
fterm_err = float(np.max(np.abs(fterm_model - fterm_ref)) /
                  (np.max(np.abs(fterm_ref)) + 1e-30))
print(f'\n[sanity] hard-core prop vs generator Lop: max|err|={prop_err:.3e}')
print(f'[sanity] forcing term vs solver geometric series: rel max err={fterm_err:.3e}')
if prop_err > 1e-6:   # buffer 为 float32，阈值与 KS 实验一致
    print('[sanity] *** 失败：hard core 传播子与生成器线性算子不一致！ ***')
    sys.exit(1)
if fterm_err > 1e-3:
    print('[sanity] *** 失败：强迫项与求解器等效积分不一致！ ***')
    sys.exit(1)
print('[sanity] OK: hard core 传播子/强迫项与生成器线性部分一致（符号/尺度正确）')

# --------------------------------------------------------------------
# 能量包络标定（训练轨迹真值, p99.5 x 1.2）
#   dE/dt = P_f - D;  dE<0 -> 与 D*dt 比, dE>0 -> 与 P_f*dt 比
# --------------------------------------------------------------------
print('\n[envelope] 标定 c_d / c_p（训练轨迹真值，p99.5 x 1.2）')
KXnp, KYnp = np.fft.fftfreq(N, d=1.0 / N)[:, None], np.fft.rfftfreq(N, d=1.0 / N)[None, :]
K2np = KXnp**2 + KYnp**2
K2inv_np = np.where(K2np > 0, 1.0 / np.maximum(K2np, 1e-30), 0.0)
ygrid = np.arange(N) * (2 * np.pi / N)
fx_grid = np.broadcast_to((SPEC.A * np.sin(SPEC.kf * ygrid))[None, :], (N, N))

E_list, D_list, P_list = [], [], []
for tr in train_traj.astype(np.float64):
    what = np.fft.rfft2(tr, axes=(1, 2))
    psi = -what * K2inv_np[None]
    u = np.fft.irfft2(-1j * KYnp[None] * psi, s=(N, N), axes=(1, 2))
    v = np.fft.irfft2(1j * KXnp[None] * psi, s=(N, N), axes=(1, 2))
    psi_p = np.fft.irfft2(psi, s=(N, N), axes=(1, 2))
    E = 0.5 * (u**2 + v**2).mean(axis=(1, 2))
    Z = 0.5 * (tr**2).mean(axis=(1, 2))
    D = 2.0 * SPEC.nu * Z + SPEC.alpha * (psi_p**2).mean(axis=(1, 2))
    P = (u * fx_grid[None]).mean(axis=(1, 2))
    E_list.append(E); D_list.append(D); P_list.append(P)
E_t = np.concatenate(E_list)
# 逐轨迹 diff（不跨轨迹边界）
dE = np.concatenate([np.diff(e) for e in E_list])
D_t = np.concatenate([d[:-1] for d in D_list])
P_t = np.concatenate([p[:-1] for p in P_list])
pos = dE > 0
ratio_p = dE[pos] / (P_t[pos] * DT + 1e-30)
ratio_d = -dE[~pos] / (D_t[~pos] * DT + 1e-30)
C_P = float(1.2 * np.percentile(ratio_p, 99.5))
C_D = float(1.2 * np.percentile(ratio_d, 99.5))
print(f'  dE>0 占比={pos.mean():.3f}  p99.5(dE/(P dt))={np.percentile(ratio_p,99.5):.4f} -> c_p={C_P:.4f}')
print(f'  dE<0 占比={(~pos).mean():.3f}  p99.5(-dE/(D dt))={np.percentile(ratio_d,99.5):.4f} -> c_d={C_D:.4f}')

# --------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------
RHS_MAIN = NSRHS2D(SPEC).to(DEV)

def build_arm(name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if name == 'anchor':
        return RCLN2DFix2(SPEC, MEAN, STD, modes=12, width=32, layers=4)
    return FNO2dUV(N, modes=12, width=32, layers=4)

def arm_loss(model, u0, t1, t2, t3, phys):
    p1 = model(u0)
    p2 = model(p1)
    p3 = model(p2)
    loss = F.mse_loss(p1, t1) + GAMMA * F.mse_loss(p2, t2) + GAMMA**2 * F.mse_loss(p3, t3)
    if phys:
        def res(pred, u_in):
            u_phys = u_in * STD + MEAN
            return (pred - u_in) / DT - RHS_MAIN(u_phys) / STD
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
        print(f'[SYS-B] {name:10s} ep{ep+1:3d}: tr={tl:.6f} val={vl:.6f}{star} '
              f'lr={sch.get_last_lr()[0]:.2e} {time.time()-t_start:.0f}s', flush=True)
        # 健康门：前 2 epoch 出现 NaN 直接中止
        if ep < 2 and not np.isfinite(vl):
            print('[health] *** NaN val loss in first 2 epochs, abort ***')
            sys.exit(1)
    train_time = time.time() - t_start
    with open(f'{CKPT_DIR}/{name}_history.json', 'w') as f:
        json.dump(hist, f)
    print(f'[SYS-B] {name} DONE. best val={best:.6f} params={n_params:,} '
          f'train_time={train_time:.0f}s', flush=True)
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

# 健康门：val 全程有下降
for nm, mk in [('anchor', bv_a), ('loss_level', bv_l), ('none', bv_n)]:
    h = json.load(open(f'{CKPT_DIR}/{nm}_history.json'))
    assert h['val_loss'][-1] < h['val_loss'][0] or mk < h['val_loss'][0], \
        f'[health] {nm} val 未下降'
print('[health] OK: 三臂 val loss 均下降，前 2 epoch 无 NaN')

# --------------------------------------------------------------------
# 评估（fair 口径）
# --------------------------------------------------------------------
def rollout_eval(model, trajs, mean, std, n_ic, arm_name, scn_name):
    """物理空间指标 + crash 冻结。trajs: [n_traj, 260, N, N]。
    IC 分配: 尽量均分到各轨迹，轨迹内等间隔。"""
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

# postproc wrapper（anchor 专用；envelope 统计按场景记录）
anchor_env = NSPostproc2D(models['anchor'], SPEC, MEAN, STD,
                          c_d=C_D, c_p=C_P,
                          mean_project=True, envelope=True).to(DEV)

def eval_scenario(scn_name, trajs, spec_scn, n_ic, mean, std):
    print(f'\n--- 评估: {scn_name} (trajs={len(trajs)}, n_ic={n_ic}, '
          f'mean={mean:.4f} std={std:.4f}, spec={spec_scn}) ---')
    anchor_env.set_spec(spec_scn, mean, std)
    anchor_env.reset_stats()
    arms = [('anchor', anchor_env),
            ('anchor_noenv', models['anchor']),
            ('loss_level', models['loss_level']),
            ('none', models['none'])]
    res = {}
    for arm_name, model in arms:
        res[arm_name] = rollout_eval(model, trajs, mean, std, n_ic,
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
        'source': 'main train-trajectory truth, p99.5 x 1.2',
        'p99_5_dE_over_Pdt': float(np.percentile(ratio_p, 99.5)),
        'p99_5_negdE_over_Ddt': float(np.percentile(ratio_d, 99.5)),
        'frac_dE_positive': float(pos.mean()),
        'dt': DT,
    },
    'per_scenario': {},
}

# in_domain: 2 条测试轨迹，用训练归一化统计
res, es = eval_scenario('in_domain', test_traj, SPEC, 5, MEAN, STD)
results['in_domain'] = res
envelope_meta['per_scenario']['in_domain'] = es

# Re OOD: nu' 从 attrs 读，自身 mean/std 重归一化
with h5py.File(H5_OOD, 'r') as f:
    ood_fields = f['fields'][:].astype(np.float32)
    SPEC_OOD = PhysicsSpec.from_h5_attrs(dict(f.attrs))
m_o, s_o = float(ood_fields.mean()), float(ood_fields.std()) + 1e-8
res, es = eval_scenario('re_ood', ood_fields, SPEC_OOD, 3, m_o, s_o)
results['re_ood'] = res
envelope_meta['per_scenario']['re_ood'] = es

# decaying OOD: PhysicsSpec 切换 f=0/alpha=0（同架构原则、不同 PhysicsSpec）
with h5py.File(H5_DEC, 'r') as f:
    dec_fields = f['fields'][:].astype(np.float32)
    SPEC_DEC = PhysicsSpec.from_h5_attrs(dict(f.attrs))
m_d, s_d = float(dec_fields.mean()), float(dec_fields.std()) + 1e-8
res, es = eval_scenario('decaying_hit', dec_fields, SPEC_DEC, 3, m_d, s_d)
results['decaying_hit'] = res
envelope_meta['per_scenario']['decaying_hit'] = es

# --------------------------------------------------------------------
# JSON 输出
# --------------------------------------------------------------------
out_json = {
    'protocol': {
        'experiment': 'NC System B: 2D Navier-Stokes cross-system Go/No-Go (fix2)',
        'equation': ('w_t + J(psi,w) = nu*Lap(w) + f_w - alpha*Lap^-1(w); '
                     'psi^=-w^/k^2; u=-psi_y,v=psi_x; '
                     'f=A*sin(k_f*y)*x-hat (contour-ETDRK4 verified)'),
        'data': {'main': H5_MAIN, 're_ood': H5_OOD, 'decaying': H5_DEC,
                 'main_attrs': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                for k, v in main_attrs.items()}},
        'train_split': 'main 14 independent-IC trajectories: 10 train / 2 val / 2 test, '
                       'z-score(train stats); OOD scenarios re-normalized with own stats',
        'optimizer': f'AdamW lr={args.lr} cosine, epochs={args.epochs}, '
                     f'batch={args.batch}, grad_clip=1.0, seed={SEED}',
        'rollout_training': f'R={R_ROLL}, gamma={GAMMA}, '
                            'loss=sum gamma^k mse(pred_k,target_k) autoregressive',
        'lambda_phys': LAMBDA_PHYS,
        'eval': 'physical-space rL2 & E/E0 & Z/Z0 (Z diagnostic only, not enveloped); '
                'crash: non-finite or rL2>5 or E/E0>5; frozen after crash; '
                'K in {10,50,100,200}; ICs spread across trajectories',
        'arms': arm_meta,
        'hardcore_sanity': {'prop_max_err': prop_err, 'forcing_rel_err': fterm_err},
        'device': DEV,
    },
    'envelope_meta': envelope_meta,
    'in_domain': results['in_domain'],
    're_ood': results['re_ood'],
    'decaying_hit': results['decaying_hit'],
}
out_path = 'results/paper_experiments/generality_system_b_fix2.json'
with open(out_path, 'w') as f:
    json.dump(out_json, f, indent=2)
print(f'\nSaved: {out_path}')

# --------------------------------------------------------------------
# 汇总表
# --------------------------------------------------------------------
print('\n' + '=' * 86)
print('  SUMMARY (K=200 | K=10)')
print('=' * 86)
for scn in ['in_domain', 're_ood', 'decaying_hit']:
    print(f'\n[{scn}]')
    print(f'{"arm":<14s} {"rL2@200":>16s} {"E/E0@200":>16s} {"Z/Z0@200":>16s} '
          f'{"crash@200":>9s} {"rL2@10":>16s} {"ms/step":>8s}')
    for arm in ['anchor', 'anchor_noenv', 'loss_level', 'none']:
        d200 = results[scn][arm]['200']
        d10 = results[scn][arm]['10']
        print(f'{arm:<14s} '
              f'{d200["rL2_mean"]:>7.3f}±{d200["rL2_std"]:<7.3f} '
              f'{d200["E_ratio_mean"]:>7.3f}±{d200["E_ratio_std"]:<7.3f} '
              f'{d200["Z_ratio_mean"]:>7.3f}±{d200["Z_ratio_std"]:<7.3f} '
              f'{d200["crash_rate"]:>9.2f} '
              f'{d10["rL2_mean"]:>7.3f}±{d10["rL2_std"]:<7.3f} '
              f'{d200["step_time_ms"]:>8.2f}')
print('\nALL DONE.')

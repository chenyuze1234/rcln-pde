"""
UPI Emergent Error Signal Verification: U1-U5 — fusion_w variant (post-fix)
================================================================================
Same protocol as compute_upi_u1_u5_cal.py, auditing THREE signals in parallel
against true per-voxel error |pred-true| (channel-mean), at every rollout step:

  a. fusion_w   : TRUE per-voxel fusion weight (NEW, exposed by commit 3da585d
                  via comp['fusion_w']). Semantics (from UPIv4Fusion source):
                      u_blend = w*u_local + (1-w)*u_global
                  w -> 1 means "trust the soft shell (u_local)"; for the
                  calibrated gate w = e_g^b / (e_g^b + e_l^b), w -> 1 when the
                  LOCAL branch is predicted accurate. Prior expectation:
                  HIGH w should coincide with LOW error  =>  NEGATIVE
                  correlation expected. We therefore report r(w) and AUROC(w)
                  AND the symmetric r(1-w) / AUROC(1-w); the naming rule is
                  applied to whichever orientation carries the signal.
  b. e_fused    : calibrated fused-error predictor  ê = fw*ê_l + (1-fw)*ê_g
                  (err_pred channels; positive control). NOTE: the reference
                  script compute_upi_u1_u5_cal.py used comp['w'] (which is the
                  STIM MAP, not the gate — the naming misalignment fixed in
                  3da585d). We use the TRUE gate fusion_w here; the legacy
                  (stim_map-weighted) variant is also recorded for comparison.
  c. stim_map   : geometric stimulus sigmoid(||u_l-u_g||/||u_g|| - 0.5)
                  (comp['w'] / comp['stim_map']; negative/baseline control,
                  reference: U1 r=-0.03±0.09, U2 AUROC=0.49).

Protocol (identical to reference): TGV Re=6400, IC 0, K=100 autoregressive
steps, error = |pred - true| channel-mean; U4 = TGV Re=800 (truncate at first
non-finite frame); U5 = MC Dropout N=10 on step-1.

Naming rule: Pearson>0.05 and AUROC>0.55 -> "discrepancy indicator";
r>0.7 and AUROC>0.8 -> "calibrated per-voxel error predictor".
"""
import torch, numpy as np, h5py, json, sys, os
sys.path.insert(0, '.'); from rollout_eval_v5 import load_model

DEV = 'cuda'; CKPT = 'checkpoints/v5_cal/v5_best.pt'
H5 = 'data_generated/tgv_re6400_N64_T20.0_fluidsim.h5'
OUT = 'results/paper_experiments'; os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

with h5py.File(H5, 'r') as f: fields = f['fields'][:]
mean = float(np.mean(fields)); std = float(np.std(fields)) + 1e-8
print(f'TGV Re=6400: {fields.shape}')

# ---------- metrics (identical implementations to reference) ----------
def pearson(x, y):
    xf = x.flatten(); yf = y.flatten()
    mx = xf.mean(); my = yf.mean()
    num = ((xf - mx) * (yf - my)).sum()
    den = np.sqrt(((xf - mx) ** 2).sum() * ((yf - my) ** 2).sum()) + 1e-10
    return float(num / den)

def spearman(x, y):
    from scipy.stats import spearmanr
    r, _ = spearmanr(x.flatten()[:20000], y.flatten()[:20000])
    return float(r)

def auroc_auprc(stim_map, true_err, q=0.10):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        return float('nan'), float('nan')
    s = stim_map.flatten(); e = true_err.flatten()
    thresh = np.percentile(e, 100 * (1 - q))
    labels = (e >= thresh).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan'), float('nan')
    try:
        return float(roc_auc_score(labels, s)), float(average_precision_score(labels, s))
    except Exception:
        return float('nan'), float('nan')

def calibration_curve(stim_map, true_err, n_bins=10):
    s = stim_map.flatten(); e = true_err.flatten()
    bins = np.percentile(s, np.linspace(0, 100, n_bins + 1))
    bin_centers = []; bin_errors = []
    for i in range(n_bins):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if mask.sum() > 10:
            bin_centers.append(s[mask].mean()); bin_errors.append(e[mask].mean())
    return np.array(bin_centers), np.array(bin_errors)

def n_monotone(cal_list):
    """count steps whose binned error sequence is non-decreasing (U3)"""
    n = 0; tot = 0
    for c in cal_list:
        er = np.array(c['errors'])
        if len(er) >= 2:
            tot += 1
            if np.all(np.diff(er) >= -1e-12): n += 1
    return n, tot

# ---------- signal extraction ----------
def extract_signals(comp):
    """returns dict of [H,W,D] numpy signal maps"""
    fw = comp.get('fusion_w', None)
    stim = comp['w'].cpu().numpy()[0, 0]                 # stim_map (legacy key)
    ep = comp.get('err_pred', None)
    out = {'stim_map': stim}
    if fw is not None:
        fw_np = fw.detach().cpu().numpy()[0, 0]
        out['fusion_w'] = fw_np
        out['one_minus_w'] = 1.0 - fw_np
        if ep is not None:
            ep_np = ep.detach().cpu().numpy()[0]         # [2,H,W,D]
            out['e_fused'] = fw_np * ep_np[1] + (1.0 - fw_np) * ep_np[0]   # true gate
            out['e_fused_legacy'] = stim * ep_np[1] + (1.0 - stim) * ep_np[0]  # reference-script variant
    return out

SIGNALS = ['fusion_w', 'one_minus_w', 'e_fused', 'e_fused_legacy', 'stim_map']

def eval_step_metrics(signals, err_mag):
    m = {}
    for name, sm in signals.items():
        pc = pearson(sm, err_mag)
        try: sp = spearman(sm, err_mag)
        except Exception: sp = float('nan')
        au, ap = auroc_auprc(sm, err_mag, q=0.10)
        bc, be = calibration_curve(sm, err_mag)
        m[name] = {'pearson': pc, 'spearman': sp, 'auroc': au, 'auprc': ap,
                   'calibration': {'bins': bc.tolist(), 'errors': be.tolist()},
                   'sig_mean': float(sm.mean())}
    return m

def new_results():
    return {s: {'pearson': [], 'spearman': [], 'auroc': [], 'auprc': [],
                'calibration': [], 'sig_mean': []} for s in SIGNALS}

def append_results(res, m):
    for s in SIGNALS:
        if s in m:
            for k in ('pearson', 'spearman', 'auroc', 'auprc', 'calibration', 'sig_mean'):
                res[s][k].append(m[s][k])

def summarize(res, keys=('fusion_w', 'one_minus_w', 'e_fused', 'e_fused_legacy', 'stim_map')):
    out = {}
    for s in keys:
        pr = np.array([v for v in res[s]['pearson'] if np.isfinite(v)])
        sp = np.array([v for v in res[s]['spearman'] if np.isfinite(v)])
        au = np.array([v for v in res[s]['auroc'] if np.isfinite(v)])
        ap = np.array([v for v in res[s]['auprc'] if np.isfinite(v)])
        nm, nt = n_monotone(res[s]['calibration'])
        out[s] = {
            'n_steps': int(len(pr)),
            'pearson_mean': float(pr.mean()) if len(pr) else float('nan'),
            'pearson_std': float(pr.std()) if len(pr) else float('nan'),
            'spearman_mean': float(sp.mean()) if len(sp) else float('nan'),
            'spearman_std': float(sp.std()) if len(sp) else float('nan'),
            'auroc_mean': float(au.mean()) if len(au) else float('nan'),
            'auprc_mean': float(ap.mean()) if len(ap) else float('nan'),
            'u3_monotone_steps': f'{nm}/{nt}',
        }
    return out

# ============================================================
# Health check: strict-load + fusion_w sanity
# ============================================================
print(f'\n{"=" * 60}')
print('  HEALTH CHECK')
print(f'{"=" * 60}')
model = load_model(CKPT, torch.device(DEV), ablation_mode='full', use_calibrated_gate=True)
ck = torch.load(CKPT, map_location='cpu')
state = ck.get('model_state_dict', ck)
mkeys = set(model.state_dict().keys())
missing = sorted(mkeys - set(state.keys())); unexpected = sorted(set(state.keys()) - mkeys)
print(f'  strict-load missing={missing} unexpected={unexpected}')
assert not missing and not unexpected, 'checkpoint key mismatch!'

init_raw = fields[0]; init = (init_raw - mean) / std
u = torch.from_numpy(init).float().unsqueeze(0).to(DEV)
with torch.no_grad():
    pred, comp = model(u, target=None, return_components=True)
fw = comp.get('fusion_w')
assert fw is not None, 'fusion_w is None — fixed model not active?'
fw_np = fw.detach().cpu().numpy()
print(f'  fusion_w: shape={tuple(fw.shape)} min={fw_np.min():.4f} max={fw_np.max():.4f} '
      f'mean={fw_np.mean():.4f} std={fw_np.std():.4f}')
assert fw_np.std() > 1e-6, 'fusion_w looks constant!'
print('  err_pred:', None if comp.get('err_pred') is None else tuple(comp['err_pred'].shape))
print('  HEALTH CHECK OK')

# ============================================================
# U1-U3: Main system (TGV Re=6400)
# ============================================================
print(f'\n{"=" * 60}')
print('  U1-U3: TGV Re=6400 (IN-DOMAIN) — fusion_w audit')
print(f'{"=" * 60}')

K_U = 100; IC_IDX = 0
init_raw = fields[IC_IDX * 10]
init = (init_raw - mean) / std
u = torch.from_numpy(init).float().unsqueeze(0).to(DEV)

results = new_results(); true_err_log = []

with torch.no_grad():
    for step in range(K_U):
        pred, comp = model(u, target=None, return_components=True)
        p_np = pred.cpu().numpy()[0]
        t_np = (fields[IC_IDX * 10 + step + 1] - mean) / std
        true_err = np.abs(p_np - t_np)
        err_mag = true_err.mean(axis=0)
        signals = extract_signals(comp)
        m = eval_step_metrics(signals, err_mag)
        append_results(results, m)
        true_err_log.append(float(err_mag.mean()))
        if step % 20 == 0:
            print(f"  K={step + 1:3d}: fw_r={m['fusion_w']['pearson']:+.4f} "
                  f"ef_r={m['e_fused']['pearson']:+.4f} stim_r={m['stim_map']['pearson']:+.4f} "
                  f"fw_AUC={m['fusion_w']['auroc']:.4f}")
        if float(err_mag.max()) > 50 or not np.isfinite(p_np).all(): break
        u = pred

summary_main = summarize(results)
print('\n--- U1/U2 summary (in-domain) ---')
for s, v in summary_main.items():
    print(f"  {s:14s}: pearson={v['pearson_mean']:+.4f}±{v['pearson_std']:.4f} "
          f"spearman={v['spearman_mean']:+.4f} AUROC={v['auroc_mean']:.4f} "
          f"AUPRC={v['auprc_mean']:.4f} monotone={v['u3_monotone_steps']}")

with open(f'{OUT}/upi_u1_u3_fusionw.json', 'w') as f:
    json.dump({'per_step': results, 'summary': summary_main,
               'true_err_mean': true_err_log}, f, indent=2)
print(f'Saved: {OUT}/upi_u1_u3_fusionw.json')

# ============================================================
# U4: OOD calibration (TGV Re=800)
# ============================================================
print(f'\n{"=" * 60}')
print('  U4: OOD — TGV Re=800')
print(f'{"=" * 60}')

H5_OOD = 'data_generated/tgv_re800_N64_T5.0_dt0.01.h5'
summary_ood = {}
try:
    with h5py.File(H5_OOD, 'r') as f:
        ood_fields = f['fields'][:]
    finite_mask = np.isfinite(ood_fields).all(axis=(1, 2, 3, 4))
    n_ok = int(np.argmin(finite_mask)) if not finite_mask.all() else len(ood_fields)
    ood_fields = ood_fields[:n_ok]
    print(f'U4: {n_ok} finite frames usable')
    ood_mean = float(np.mean(ood_fields)); ood_std = float(np.std(ood_fields)) + 1e-8
    ood_init = (ood_fields[0] - ood_mean) / ood_std
    u_ood = torch.from_numpy(ood_init).float().unsqueeze(0).to(DEV)

    ood_results = new_results()
    with torch.no_grad():
        for step in range(min(100, len(ood_fields) - 1)):
            pred, comp = model(u_ood, target=None, return_components=True)
            p_np = pred.cpu().numpy()[0]
            t_np = (ood_fields[step + 1] - ood_mean) / ood_std
            err_mag = np.abs(p_np - t_np).mean(axis=0)
            signals = extract_signals(comp)
            m = eval_step_metrics(signals, err_mag)
            append_results(ood_results, m)
            if step % 20 == 0:
                print(f"  K={step + 1:3d}: fw_r={m['fusion_w']['pearson']:+.4f} "
                      f"ef_r={m['e_fused']['pearson']:+.4f} stim_r={m['stim_map']['pearson']:+.4f}")
            u_ood = pred
    summary_ood = summarize(ood_results)
    print('\n--- U4 summary (OOD) ---')
    for s, v in summary_ood.items():
        print(f"  {s:14s}: pearson={v['pearson_mean']:+.4f}±{v['pearson_std']:.4f} "
              f"AUROC={v['auroc_mean']:.4f} monotone={v['u3_monotone_steps']}")
    with open(f'{OUT}/upi_u4_ood_fusionw.json', 'w') as f:
        json.dump({'per_step': ood_results, 'summary': summary_ood}, f, indent=2)
    print(f'Saved: {OUT}/upi_u4_ood_fusionw.json')
except Exception as e:
    print(f'U4 SKIPPED: {e}')

# ============================================================
# U5: MC Dropout baseline (same protocol as reference)
# ============================================================
print(f'\n{"=" * 60}')
print('  U5: MC Dropout UQ baseline')
print(f'{"=" * 60}')

model.train()
N_MC = 10
init_mc = (fields[0] - mean) / std
u_mc = torch.from_numpy(init_mc).float().unsqueeze(0).to(DEV)
preds = []
with torch.no_grad():
    for _ in range(N_MC):
        p = model(u_mc)
        if isinstance(p, tuple): p = p[0]
        preds.append(p.cpu().numpy()[0])
preds = np.stack(preds)
mc_std_mag = preds.std(axis=0).mean(axis=0)
t_np = (fields[1] - mean) / std
true_err_mc = np.abs(preds.mean(axis=0) - t_np).mean(axis=0)
mc_pearson = pearson(mc_std_mag, true_err_mc)
mc_auroc, mc_auprc = auroc_auprc(mc_std_mag, true_err_mc)
print(f'MC Dropout: Pearson={mc_pearson:+.4f} AUROC={mc_auroc:.4f} '
      f'(mc_std mean={mc_std_mag.mean():.6f} — no dropout layers in CNN encoder, expect ~0)')

model.eval()
with torch.no_grad():
    _, comp = model(u_mc, target=None, return_components=True)
sigs_mc = extract_signals(comp)
u5 = {'mc_dropout': {'pearson': mc_pearson, 'auroc': mc_auroc, 'auprc': mc_auprc,
                     'n_samples': N_MC, 'mc_std_mean': float(mc_std_mag.mean())}}
for s in ('fusion_w', 'one_minus_w', 'e_fused', 'e_fused_legacy', 'stim_map'):
    if s in sigs_mc:
        u5[s] = {'pearson': pearson(sigs_mc[s], true_err_mc),
                 'auroc_auprc': auroc_auprc(sigs_mc[s], true_err_mc)}
        print(f"UPI {s:14s}: Pearson={u5[s]['pearson']:+.4f} AUROC={u5[s]['auroc_auprc'][0]:.4f}")
with open(f'{OUT}/upi_u5_uq_fusionw.json', 'w') as f:
    json.dump(u5, f, indent=2)
print(f'Saved: {OUT}/upi_u5_uq_fusionw.json')
model.eval()

# ============================================================
# FINAL SUMMARY + NAMING RULE CHECK
# ============================================================
print(f'\n{"=" * 70}')
print('  U1-U5 VERIFICATION SUMMARY — fusion_w (post 3da585d)')
print(f'{"=" * 70}')
print(f'{"signal":14s} {"U1 pearson":>16s} {"U1 spearman":>12s} {"U2 AUROC":>9s} {"U2 AUPRC":>9s} {"U3 mono":>9s} {"U4 pearson":>11s} {"U4 AUROC":>9s}')
for s in ('fusion_w', 'one_minus_w', 'e_fused', 'e_fused_legacy', 'stim_map'):
    m = summary_main[s]; o = summary_ood.get(s, {})
    print(f"{s:14s} {m['pearson_mean']:+.4f}±{m['pearson_std']:.4f}   "
          f"{m['spearman_mean']:+.4f}     {m['auroc_mean']:.4f}  {m['auprc_mean']:.4f}  "
          f"{m['u3_monotone_steps']:>9s} {o.get('pearson_mean', float('nan')):+.4f}    {o.get('auroc_mean', float('nan')):.4f}")

print('\nReference (old gate/stim_map, same protocol): U1 r=-0.03±0.09, U2 AUROC=0.49')
print('Reference (v5-cal ê_fused via legacy w):      U1 r=+0.256±0.041, U2 AUROC=0.740')
print('\nNAMING RULE CHECK (per signal, in-domain):')
for s in ('fusion_w', 'one_minus_w', 'e_fused', 'stim_map'):
    m = summary_main[s]
    ok = abs(m['pearson_mean']) > 0.05 and m['auroc_mean'] > 0.55
    cal = m['pearson_mean'] > 0.7 and m['auroc_mean'] > 0.8
    verdict = ('CALIBRATED ERROR PREDICTOR' if cal else
               ('discrepancy indicator' if ok else 'NO DETECTABLE SIGNAL'))
    print(f"  {s:14s}: {verdict}")

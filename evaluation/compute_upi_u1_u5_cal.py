"""
UPI Emergent Error Signal Verification: U1-U5 — v5-cal calibrated-gate variant
=================================================================================
Identical protocol to compute_upi_u1_u5.py, with two changes:
  1. Checkpoint: checkpoints/v5_cal/v5_best.pt (calibrated gate model)
  2. Signal: calibrated fused-error predictor
        ê_fused = w·ê_l + (1−w)·ê_g
     from ErrorCalibHead (comp['err_pred']) and the zero-param gate w,
     instead of the unsupervised geometric stim_map.

U1: Spatial correlation  corr(ê(x), |pred-true|(x)) per-step
U2: Error detection     AUROC/AUPRC for top-q% voxel error
U3: Calibration         Binned ê vs empirical error
U4: OOD calibration     Repeat on unseen Re=800
U5: UQ baseline         MC Dropout comparison

Reference (old gate, same protocol): U1 r=-0.03±0.09, U2 AUROC=0.49.
Naming rule: Pearson>0.05 and AUROC>0.55 -> "discrepancy indicator";
for a calibrated error predictor we additionally expect U3 monotone bins.
"""
import torch, numpy as np, h5py, json, sys, os
sys.path.insert(0,'.'); from rollout_eval_v5 import load_model

DEV='cuda'; CKPT='checkpoints/v5_cal/v5_best.pt'; H5='data_generated/tgv_re6400_N64_T20.0_fluidsim.h5'
OUT='results/paper_experiments'; os.makedirs(OUT,exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

with h5py.File(H5,'r') as f: fields=f['fields'][:]
mean=float(np.mean(fields)); std=float(np.std(fields))+1e-8
print(f'TGV Re=6400: {fields.shape}')

def pearson(x,y):
    xf=x.flatten(); yf=y.flatten()
    mx=xf.mean(); my=yf.mean()
    num=((xf-mx)*(yf-my)).sum()
    den=np.sqrt(((xf-mx)**2).sum()*((yf-my)**2).sum())+1e-10
    return float(num/den)

def spearman(x,y):
    from scipy.stats import spearmanr
    r,_=spearmanr(x.flatten()[:20000],y.flatten()[:20000])
    return float(r)

def auroc_auprc(stim_map, true_err, q=0.10):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        return float('nan'), float('nan')
    s=stim_map.flatten(); e=true_err.flatten()
    thresh=np.percentile(e,100*(1-q))
    labels=(e>=thresh).astype(int)
    if labels.sum()==0 or labels.sum()==len(labels):
        return float('nan'), float('nan')
    try:
        return float(roc_auc_score(labels,s)), float(average_precision_score(labels,s))
    except: return float('nan'), float('nan')

def calibration_curve(stim_map, true_err, n_bins=10):
    s=stim_map.flatten(); e=true_err.flatten()
    bins=np.percentile(s,np.linspace(0,100,n_bins+1))
    bin_centers=[]; bin_errors=[]
    for i in range(n_bins):
        mask=(s>=bins[i])&(s<bins[i+1])
        if mask.sum()>10:
            bin_centers.append(s[mask].mean())
            bin_errors.append(e[mask].mean())
    return np.array(bin_centers), np.array(bin_errors)

def calib_signal(comp):
    """ê_fused = w·ê_l + (1−w)·ê_g  -> [H,W,D] numpy"""
    ep = comp['err_pred']                       # [B,2,H,W,D]
    w = comp['w']                               # [B,1,H,W,D]
    sig = w * ep[:, 1:2] + (1.0 - w) * ep[:, 0:1]
    return sig.cpu().numpy()[0, 0]

# ============================================================
# U1-U3: Main system (TGV Re=6400)
# ============================================================
print(f'\n{"="*60}')
print('  U1-U3: TGV Re=6400 (IN-DOMAIN) — v5-cal')
print(f'{"="*60}')

model=load_model(CKPT,torch.device(DEV),ablation_mode='full',use_calibrated_gate=True)

K_U=100; IC_IDX=0
init_raw=fields[IC_IDX*10]
init=(init_raw-mean)/std
u=torch.from_numpy(init).float().unsqueeze(0).to(DEV)

results={'step':[],'pearson':[],'spearman':[],'auroc':[],'auprc':[],
         'calibration':[],'stim_mean':[],'true_err_mean':[]}

with torch.no_grad():
    for step in range(K_U):
        pred,comp=model(u,target=None,return_components=True)
        p_np=pred.cpu().numpy()[0]
        t_np=(fields[IC_IDX*10+step+1]-mean)/std
        true_err=np.abs(p_np-t_np)
        sm=calib_signal(comp)
        err_mag=true_err.mean(axis=0)
        pc=pearson(sm,err_mag)
        try: sp=spearman(sm,err_mag)
        except: sp=float('nan')
        au,ap=auroc_auprc(sm,err_mag,q=0.10)
        bc,be=calibration_curve(sm,err_mag)

        results['step'].append(step+1)
        results['pearson'].append(pc)
        results['spearman'].append(sp)
        results['auroc'].append(au)
        results['auprc'].append(ap)
        results['calibration'].append({'bins':bc.tolist(),'errors':be.tolist()})
        results['stim_mean'].append(float(sm.mean()))
        results['true_err_mean'].append(float(err_mag.mean()))

        if step%20==0:
            print(f'  K={step+1:3d}: pearson={pc:+.4f} spearman={sp:+.4f} AUROC={au:.4f}')

        if float(err_mag.max())>50 or not np.isfinite(p_np).all(): break
        u=pred

pearson_arr=np.array([v for v in results['pearson'] if np.isfinite(v)])
auroc_arr=np.array([v for v in results['auroc'] if np.isfinite(v)])
print(f'\nU1: mean Pearson={pearson_arr.mean():+.4f} +/- {pearson_arr.std():.4f}')
print(f'U2: mean AUROC={auroc_arr.mean():.4f} (AUPRC baseline ~0.10)')

u1u3_data={'U1_pearson':results['pearson'],'U1_spearman':results['spearman'],
           'U2_auroc':results['auroc'],'U2_auprc':results['auprc'],
           'U3_calibration':results['calibration'],
           'summary':{'mean_pearson':float(pearson_arr.mean()),'std_pearson':float(pearson_arr.std()),
                      'mean_auroc':float(auroc_arr.mean()),'mean_auprc':float(np.nanmean(results['auprc']))}}
with open(f'{OUT}/upi_u1_u3_cal.json','w') as f: json.dump(u1u3_data,f,indent=2)
print(f'Saved: {OUT}/upi_u1_u3_cal.json')

# ============================================================
# U4: OOD calibration (TGV Re=800)
# ============================================================
print(f'\n{"="*60}')
print('  U4: OOD — TGV Re=800')
print(f'{"="*60}')

H5_OOD='data_generated/tgv_re800_N64_T5.0_dt0.01.h5'
try:
    with h5py.File(H5_OOD,'r') as f:
        ood_fields=f['fields'][:]
    # data hygiene: solver crashed at frame 222 — trailing frames are NaN.
    # Truncate at first non-finite frame or stats become NaN and poison the input.
    finite_mask=np.isfinite(ood_fields).all(axis=(1,2,3,4))
    n_ok=int(np.argmin(finite_mask)) if not finite_mask.all() else len(ood_fields)
    ood_fields=ood_fields[:n_ok]
    print(f'U4: {n_ok} finite frames usable')
    ood_mean=float(np.mean(ood_fields)); ood_std=float(np.std(ood_fields))+1e-8
    ood_init=(ood_fields[0]-ood_mean)/ood_std
    u_ood=torch.from_numpy(ood_init).float().unsqueeze(0).to(DEV)

    ood_results={'step':[],'pearson':[],'auroc':[]}
    with torch.no_grad():
        for step in range(min(100,len(ood_fields)-1)):
            pred,comp=model(u_ood,target=None,return_components=True)
            p_np=pred.cpu().numpy()[0]; t_np=(ood_fields[step+1]-ood_mean)/ood_std
            err_mag=np.abs(p_np-t_np).mean(axis=0)
            sm=calib_signal(comp)
            pc=pearson(sm,err_mag); au,ap=auroc_auprc(sm,err_mag)
            ood_results['step'].append(step+1)
            ood_results['pearson'].append(pc); ood_results['auroc'].append(au)
            if step%20==0:
                print(f'  K={step+1:3d}: pearson={pc:+.4f} AUROC={au:.4f}')
            u_ood=pred

    ood_pearson=np.array([v for v in ood_results['pearson'] if np.isfinite(v)])
    ood_auroc=np.array([v for v in ood_results['auroc'] if np.isfinite(v)])
    print(f'U4 OOD: mean Pearson={ood_pearson.mean():+.4f} AUROC={ood_auroc.mean():.4f}')
    with open(f'{OUT}/upi_u4_ood_cal.json','w') as f: json.dump(ood_results,f,indent=2)
except Exception as e:
    print(f'U4 SKIPPED: {e}')
    ood_pearson=np.array([]); ood_auroc=np.array([])

# ============================================================
# U5: MC Dropout baseline
# ============================================================
print(f'\n{"="*60}')
print('  U5: MC Dropout UQ baseline')
print(f'{"="*60}')

model.train()
N_MC=10
init_mc=(fields[0]-mean)/std
u_mc=torch.from_numpy(init_mc).float().unsqueeze(0).to(DEV)
preds=[]
with torch.no_grad():
    for _ in range(N_MC):
        p=model(u_mc)
        if isinstance(p,tuple): p=p[0]
        preds.append(p.cpu().numpy()[0])
preds=np.stack(preds)
mc_std=preds.std(axis=0)
mc_std_mag=mc_std.mean(axis=0)

t_np=(fields[1]-mean)/std
true_err_mc=np.abs(preds.mean(axis=0)-t_np).mean(axis=0)

mc_pearson=pearson(mc_std_mag,true_err_mc)
mc_auroc,mc_auprc=auroc_auprc(mc_std_mag,true_err_mc)
print(f'MC Dropout: Pearson={mc_pearson:+.4f} AUROC={mc_auroc:.4f}')

model.eval()
with torch.no_grad():
    _,comp=model(u_mc,target=None,return_components=True)
upi_sm=calib_signal(comp)
upi_pearson=pearson(upi_sm,true_err_mc)
upi_auroc,upi_auprc=auroc_auprc(upi_sm,true_err_mc)
print(f'UPI calibrated: Pearson={upi_pearson:+.4f} AUROC={upi_auroc:.4f}')

u5_data={'mc_dropout':{'pearson':mc_pearson,'auroc':mc_auroc,'auprc':mc_auprc,'n_samples':N_MC},
         'upi_calibrated':{'pearson':upi_pearson,'auroc':upi_auroc,'auprc':upi_auprc}}
with open(f'{OUT}/upi_u5_uq_cal.json','w') as f: json.dump(u5_data,f,indent=2)
print(f'Saved: {OUT}/upi_u5_uq_cal.json')

model.eval()

# ============================================================
# FINAL SUMMARY + NAMING RULE CHECK
# ============================================================
print(f'\n{"="*65}')
print('  U1-U5 VERIFICATION SUMMARY — v5-cal')
print(f'{"="*65}')
print(f'U1: Spatial correlation   Pearson={pearson_arr.mean():+.4f} +/- {pearson_arr.std():.4f}')
print(f'U2: Error detection       AUROC={auroc_arr.mean():.4f} (chance=0.50)')
print(f'U3: Calibration           Binned ê vs error monotonic? (see json)')
print(f'U4: OOD calibration      Pearson={ood_pearson.mean():+.4f} AUROC={ood_auroc.mean():.4f}' if len(ood_pearson)>0 else 'U4: SKIPPED')
print(f'U5: UQ baseline          MC Dropout Pearson={mc_pearson:+.4f} vs UPI={upi_pearson:+.4f}')
print()
print('NAMING RULE CHECK:')
if abs(pearson_arr.mean())>0.05 and auroc_arr.mean()>0.55:
    print('  => signal has predictive value (Pearson>0.05, AUROC>0.55)')
    if pearson_arr.mean()>0.7 and auroc_arr.mean()>0.8:
        print('  => CALIBRATED ERROR PREDICTOR thresholds met (r>0.7, AUROC>0.8)')
        print('     "calibrated per-voxel error predictor" language is justified.')
    else:
        print('  => "geometry-driven discrepancy indicator" only.')
else:
    print('  => No detectable signal — remove from paper or note as null result')

# RCLN-PDE: Architecture-embedded Physical Constraints for Long-horizon Neural PDE Prediction

Companion code and results for the manuscript:

> **"Architecture-embedded physical constraints enable stable long-horizon neural PDE prediction"**
> (submitted to *Nature Communications*)

This repository contains the core custom code, configurations, per-initial-condition result JSONs,
and figure-source data behind the paper. It is a curated subset of the authors' working repository,
covering exactly what the Data/Code Availability statement promises.

## The RCLN architecture

RCLN (Residual-Coupled Link Neuron) embeds physics directly into the network architecture
through three components:

1. **Physical anchor (hard core)** — an analytic/spectral physical propagator that provides an
   exact, constraint-satisfying baseline prediction (`F_hard`).
2. **Dissipation envelope** — a learned monotone envelope that bounds the energy-decay rate of the
   predicted trajectory, preventing unphysical energy growth in long rollouts.
3. **Geometric gating (soft shell)** — a residual soft branch `F_soft` coupled to the anchor through
   a gate, so the model learns `F_soft + lambda_res * P[F_hard]` with the gate modulating how much
   learned correction enters each prediction.

The Unified Physics Interface (UPI) verifies physical contracts (energy, enstrophy, divergence,
spectral slope) of every rollout.

## Three benchmark systems

| System | Data generation | Training entry | Evaluation |
|---|---|---|---|
| 3D Taylor–Green vortex (TGV) | `data_generation/generate_tgv_fluidsim.py`, `data_generation/generate_3d_*.py`, `data_generation/taylor_green_*.py` | `training/train_baselines_t20.py` (+ `run_t20_baselines_retry.sh`) | `evaluation/nc_suite_eval.py`, `evaluation/nc_ood_eval.py` |
| 2D Kolmogorov / Navier–Stokes flow | `data_generation/generate_2d_*.py`, `data_generation/generate_kolmogorov.py` | `training/train_system_b_2dns.py`, `training/run_system_b_fix2.py`, `training/run_system_b_res_ood.py` | `evaluation/nc_suite_eval.py` (system B arms) |
| 1D Kuramoto–Sivashinsky (KS) | `data_generation/generate_ks_data.py`, `data_generation/generate_ks_l22_dt001.py`, `data_generation/generate_ks_l32_l44_dt001.py` | `training/train_system_c_ks.py`, `training/run_system_c_fix2.py` | `evaluation/nc_suite_eval.py` (system C arms) |

## Repository layout

```
models/              # only the architectures used in the paper:
                     #   rcln_upi_v5.py          - main 3D TGV model (fix2 / fix2+envelope arms)
                     #   rcln_upi_v5_spec.py     - spectral-split variant (soft-branch high-freq)
                     #   rcln_fix2_soft.py       - fix2 + soft branch, v2/v2-transplant loaders
                     #   memory_module_v5.py     - FiLM memory module (used by the two files above)
                     #   rcln_2d_fix2.py         - 2D Kolmogorov/NS arm (+ built-in FNO2d baseline)
                     #   rcln_2d_ns.py           - 2D Navier-Stokes arm (train_system_b_2dns)
                     #   rcln_1d_fix2.py         - 1D KS fix2 arm (+ built-in FNO1d baseline)
                     #   rcln_1d_ks.py           - 1D Kuramoto-Sivashinsky arm
                     #   dissipation_envelope.py - learned dissipation envelope wrapper
                     #   physics_correct.py      - PhysicsCorrect baseline
                     #   unet_3d.py              - U-Net baseline
                     #   baselines_v6_ext.py     - DPOT / Transolver adapters
                     # (FNO baselines come from the external `neuralop` package)
training/            # training entry points and runner scripts for the three systems
evaluation/          # NC evaluation suite, OOD evaluation, physics statistics, mechanism analyses (M1-M3), UPI metrics
data_generation/     # PDE solvers / data generators (3D TGV, 2D NS/Kolmogorov, 1D KS) and JHTDB fetch utilities
plotting/            # figure production scripts (Fig.1-Fig.7) and suite-merging helper
results/paper_experiments/  # per-initial-condition result JSONs consumed by the figure scripts
paper/               # manuscript source: manuscript_nc_en.tex, si_nc_en.tex, cover_letter_en.md
```

## Reproduction

All scripts were developed on Windows + PyTorch (CUDA). Install dependencies first:

```bash
pip install torch numpy scipy matplotlib h5py scikit-learn pandas
# optional, only for specific generators/baselines:
pip install fluidsim neuralop
```

Scripts use flat imports between sibling directories. Run from the repository root with the
sub-directories on `PYTHONPATH`, e.g. (git-bash / Linux):

```bash
export PYTHONPATH=".:models:training:evaluation"
# Windows cmd:
# set PYTHONPATH=.;models;training;evaluation
```

Typical pipeline:

```bash
# 1. Generate data (example: 3D TGV via fluidsim; 1D KS via the built-in solver)
python data_generation/generate_tgv_fluidsim.py
python data_generation/generate_ks_data.py

# 2. Train
python training/train_baselines_t20.py          # T=20 baselines (FNO, U-Net, DPOT, FNO+PhysReg)
python training/train_system_b_2dns.py          # 2D Kolmogorov system
python training/train_system_c_ks.py            # 1D KS system
python training/train_fix2_soft.py              # RCLN fix2 + soft branch

# 3. Evaluate (writes JSONs into results/paper_experiments/)
python evaluation/nc_suite_eval.py              # main k=200 rollout suite
python evaluation/nc_ood_eval.py                # out-of-distribution evaluation
python evaluation/nc_physics_stats.py           # physics statistics
python evaluation/nc_mechanism_m1_m3.py         # mechanism analyses M1-M3
python evaluation/nc_upi_u1u5.py                # UPI U1-U5 metrics

# 4. Figures (reads results/paper_experiments/*.json, writes paper_v6/figures/)
python plotting/make_figures_nc.py
```

Note: `plotting/make_figures_nc.py` also reads the two figure-source arrays shipped in
`results/paper_experiments/` (`upi_maps.npz` for Fig.6, `physics_diagnostics.npz` for Fig.4),
so all figures can be regenerated from the shipped data alone. Datasets and checkpoints are not
distributed with this repository; regenerate them with the scripts above.

## Result JSONs

`results/paper_experiments/` contains the per-initial-condition evaluation outputs referenced by
the paper figures, including:

- `nc_suite_k200.json` — main k=200 rollout suite (all models, truth curves, spectra, divergence);
- `nc_suite_t20_baselines_k200.json`, `nc_suite_fix2env_k200.json`, `nc_suite_fix2soft_k200.json`,
  `nc_suite_v2trans_k200.json` — baseline / ablation suites;
- `nc_physics_stats*.json` — physics-consistency statistics;
- `nc_ood_eval.json`, `generality_system_*_fix2.json`, `generality_system_b_res_ood.json` — OOD/generality;
- `mechanism_m1_m3.json` and related `mechanism_*.json` — mechanism dissection;
- `nc_upi_u1u5.json`, `upi_*.json` — UPI contract metrics (U1-U5);
- `anchor_ablation.json`, `cross_re.json`, `noise_robustness.json`, `rollout_*.json` — ablations and robustness.

Each JSON stores per-IC means/stds so that every panel in the paper can be regenerated without
re-running GPU rollouts.

## Paper source

`paper/` contains the English manuscript (`manuscript_nc_en.tex`), the Supplementary Information
(`si_nc_en.tex`), and the cover letter (`cover_letter_en.md`).

## License

This code is released under the [Apache License 2.0](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@article{rcln_pde_nc,
  title   = {Architecture-embedded physical constraints enable stable long-horizon neural PDE prediction},
  author  = {Chen, Yuze and collaborators},
  journal = {Nature Communications (under review)},
  year    = {2026},
  note    = {Code available at https://github.com/chenyuze1234/rcln-pde}
}
```

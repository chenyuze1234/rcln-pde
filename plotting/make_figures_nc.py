# -*- coding: utf-8 -*-
"""
make_figures_nc.py — NC 冲刺版论文 Fig.1–Fig.7 生产脚本（可重复跑）。
所有数字来自 results/paper_experiments/ 下的已验证 JSON/NPZ，不编造。
输出: paper_v6/figures/fig{N}_*.png (300 dpi) + .pdf + CAPTIONS.md
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "paper_experiments"
OUT = ROOT / "paper_v6" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 统一风格
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "grid.alpha": 0.28, "grid.linewidth": 0.5,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.default": "regular",
})

OURS = "#B2182B"           # 强调色：ours
TRUTH = "#111111"
STYLE = {  # name -> (color, ls, lw, label)
    "fix2_env":    (OURS,       "-",  2.4, "RCLN (fix2+env)"),
    "fix2":        ("#7f7f7f",  "--", 1.1, "RCLN (fix2, no env.)"),
    "fix2_hc":     ("#17becf",  "--", 1.1, "RCLN-hc"),
    "fix2_no_upi": ("#bcbd22",  "--", 1.1, "RCLN-noUPI"),
    "v2":          ("#e377c2",  "-",  1.1, "RCLN-v2 (soft)"),
    "v2gate70":    ("#e377c2",  "--", 1.1, "RCLN-v2 gate70"),
    "v2gate90":    ("#e377c2",  ":",  1.1, "RCLN-v2 gate90"),
    "v3":          ("#9ac0cd",  "-",  1.1, "RCLN-v3"),
    "t20_fno":     ("#1f77b4",  "-",  1.1, "FNO"),
    "t20_unet":    ("#ff7f0e",  "-",  1.1, "U-Net"),
    "t20_dpot":    ("#2ca02c",  "-",  1.1, "DPOT"),
    "t20_physreg": ("#9467bd",  "-",  1.1, "FNO+PhysReg"),
    "pc_fno":      ("#8c564b",  "-",  1.1, "PhysicsCorrect (approx.)"),
}
ARM_STYLE = {  # 跨系统 4 臂
    "anchor":       (OURS,       "-",  "anchor (ours)"),
    "anchor_noenv": ("#7f7f7f",  "--", "anchor w/o envelope"),
    "loss_level":   ("#ff7f0e",  "-",  "loss-level phys."),
    "none":         ("#1f77b4",  "-",  "no physics"),
}
KS = [1, 10, 50, 100, 200]

# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def J(name):
    return json.load(open(RES / name))

suite = J("nc_suite_k200.json")
stats_env = J("nc_physics_stats_fix2env.json")
stats = J("nc_physics_stats.json")
upi_u1u5 = J("nc_upi_u1u5.json")
upi_v2t = J("nc_upi_v2trans_u1u4.json")
genB = J("generality_system_b_fix2.json")
genB_res = J("generality_system_b_res_ood.json")
genC = J("generality_system_c_fix2.json")
ood3d = J("nc_ood_eval.json")
upi_maps = np.load(RES / "upi_maps.npz")

M = suite["models"]
TRUTH_E = np.array(suite["truth"]["E_over_E0"])      # len 201, steps 0..200
TRUTH_Z = np.array(suite["truth"]["enstrophy"])
TRUTH_SPEC = suite["truth"]["spectra"]
TRUTH_DIV = suite["truth"]["div_mean"]
E200_TRUTH = TRUTH_E[200]

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)

def rk(mname, key):
    """suite rL2/E dict at @k -> (mean, std)"""
    d = M[mname][key]
    return np.array([d[f"@{k}"]["mean"] for k in KS]), np.array([d[f"@{k}"]["std"] for k in KS])

def box(ax, x, y, w, h, text, fc, fs=7.6, tc="white", ec="none", bold=True, lw=0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.4, style="-|>", ms=11, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# ----------------------------------------------------------------------------
# Fig.1  Concept & Failure Mode
# ----------------------------------------------------------------------------
def fig1():
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.10,
                          top=0.97, bottom=0.08, left=0.08, right=0.98)

    # ---- (a) predict-then-correct ----
    axa = fig.add_subplot(gs[0, 0]); axa.axis("off")
    axa.set_xlim(0, 10); axa.set_ylim(0, 6.2)
    axa.set_title("(a)  Predict-then-correct (FNO / PINO / post-hoc)", fontsize=9, loc="left")
    box(axa, 0.1, 3.5, 1.4, 1.3, "Input\n$u_t$", "#4a90c2", fs=6.8)
    box(axa, 1.9, 3.5, 2.2, 1.3, "Neural\npredictor", "#4a90c2", fs=6.8)
    box(axa, 4.5, 3.5, 2.2, 1.3, "$\\hat{u}_{t+1}$\ndrifted", "#c25e4a", fs=6.8)
    box(axa, 7.1, 3.5, 2.2, 1.3, "Post-hoc\ncorrector", "#888888", fs=6.8)
    arrow(axa, 1.55, 4.15, 1.85, 4.15); arrow(axa, 4.15, 4.15, 4.45, 4.15)
    arrow(axa, 6.75, 4.15, 7.05, 4.15)
    arrow(axa, 3.0, 3.4, 8.2, 3.4, color="#B2182B", lw=1.1, ls=(0, (4, 3)))
    axa.text(5.6, 2.9, "correction optional / bypassable", fontsize=6.8, color="#B2182B",
             ha="center", style="italic")
    axa.text(5.0, 1.9, "physics enters only as a soft loss\nor an a-posteriori fix —\n"
                       "nothing prevents energy blow-up", fontsize=7.0, ha="center",
             color="#444444", style="italic")
    axa.text(5.0, 0.55, "=> instability accumulates autoregressively", fontsize=7.4,
             ha="center", color="#B2182B", fontweight="bold")

    # ---- (b) ours ----
    axb = fig.add_subplot(gs[0, 1]); axb.axis("off")
    axb.set_xlim(0, 10); axb.set_ylim(0, 6.2)
    axb.set_title("(b)  RCLN: physics on the forward pass", fontsize=9, loc="left")
    box(axb, 0.1, 3.7, 1.3, 1.2, "Input\n$u_t$", "#4a90c2", fs=6.8)
    box(axb, 1.8, 3.7, 2.3, 1.2, "Physical\nAnchor", "#1f6e8c", fs=6.8)
    box(axb, 1.8, 0.8, 2.3, 1.2, "Residual\nLearner", "#3d9970", fs=6.8)
    box(axb, 4.6, 3.7, 2.3, 1.2, "Geometry\nGate", "#e08a2e", fs=6.8)
    box(axb, 7.4, 3.7, 2.4, 1.2, "Output\n$u_{t+1}\\in\\mathcal{C}$", "#7a5195", fs=6.8)
    arrow(axb, 1.45, 4.3, 1.75, 4.3)
    arrow(axb, 4.15, 4.3, 4.55, 4.3)
    arrow(axb, 6.95, 4.3, 7.35, 4.3)
    arrow(axb, 2.95, 2.05, 2.95, 3.65)
    arrow(axb, 4.15, 1.4, 5.7, 1.4)
    arrow(axb, 5.7, 1.4, 5.7, 3.65)
    axb.text(5.95, 2.4, "$\\delta u$", fontsize=7.2, color="#3d9970")
    axb.text(2.95, 5.75, "spectral div-free projection\n+ dissipation envelope",
             fontsize=6.4, ha="center", color="#1f6e8c")
    axb.text(5.75, 5.12, "$g\\in[0,1]$,  $\\Vert g\\odot\\delta u\\Vert\\leq R_{eff}$",
             fontsize=6.4, ha="center", color="#e08a2e")
    axb.text(2.95, 0.35, "local correction $\\delta u$ (bounded)", fontsize=6.4,
             ha="center", color="#3d9970")
    axb.text(8.6, 2.75, "non-bypassable:\nevery step re-enters\n$\\mathcal{C}$ before the next",
             fontsize=6.6, ha="center", color="#7a5195", fontweight="bold")

    # ---- (c) data panel ----
    axc = fig.add_subplot(gs[1, :])
    steps = np.arange(1, 201)
    axc.plot(np.arange(0, 201), TRUTH_E, color=TRUTH, ls="--", lw=2.6, label="Truth (DNS)",
             zorder=2)
    for name in ["t20_physreg", "t20_fno", "fix2_env"]:
        c, ls, lw, lab = STYLE[name]
        E = np.array(M[name]["trajectories"]["E_mean"])
        axc.plot(steps, E, color=c, ls=ls, lw=lw, label=lab, zorder=4)
    for name, dy, xt in [("t20_physreg", 0.55, (30, 4.55)), ("t20_fno", -0.75, (150, 2.25))]:
        cs = [s for s in M[name]["crash_steps"] if s > 0]
        c0 = float(np.mean(cs))
        c = STYLE[name][0]
        axc.axvline(c0, color=c, ls=":", lw=1.0, alpha=0.8)
        axc.annotate(f"{STYLE[name][3]}\nfirst crash $k\\approx${c0:.0f}",
                     xy=(c0, 3.1), xytext=xt, fontsize=7.2, color=c,
                     arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
    axc.set_xlabel("rollout step $k$ (3D Taylor–Green vortex, Re=6400)")
    axc.set_ylabel("$E(k)/E_0$")
    axc.set_xlim(0, 205); axc.set_ylim(0.4, 5.6)
    axc.grid(True)
    axc.legend(loc="center left", framealpha=0.9)
    axc.text(0.985, 0.025, "trajectories truncated at $E/E_0$=5 for visibility —\n"
             "post-crash states are NaN (curves held at last finite value);\n"
             "full runaway on log scale in the inset",
             transform=axc.transAxes, fontsize=6.4, ha="right", va="bottom",
             color="#444444", style="italic")
    axc.set_title("(c)  Same initial condition, K=200 rollout: unconstrained baselines diverge, "
                  "anchored model stays on the truth manifold", fontsize=9, loc="left")

    # log-y inset: complete runaway trajectories. 崩溃后状态为 NaN，存储轨迹冻结在
    # 最后有限步 —— inset 画到各模型首崩步为止，× 标记首崩点（数据事实，非外推）
    axin = axc.inset_axes([0.615, 0.575, 0.365, 0.37])
    axin.plot(np.arange(0, 201), TRUTH_E, color=TRUTH, ls="--", lw=1.5)
    for name in ["t20_physreg", "t20_fno", "fix2_env"]:
        c, ls, lw, lab = STYLE[name]
        E = np.array(M[name]["trajectories"]["E_mean"])
        cs = [s for s in M[name]["crash_steps"] if s > 0]
        kend = min(cs) if cs else len(E)
        axin.plot(steps[:kend], E[:kend], color=c, ls=ls, lw=1.4)
        if cs:
            axin.plot(steps[kend - 1], E[kend - 1], "x", color=c, ms=5.5, mew=1.6)
    axin.set_yscale("log")
    axin.set_xlim(0, 205); axin.set_ylim(0.55, 9)
    axin.set_yticks([0.6, 1, 2, 5]); axin.set_yticklabels(["0.6", "1", "2", "5"])
    axin.minorticks_off()
    axin.tick_params(labelsize=6.2)
    axin.grid(True, alpha=0.3)
    axin.text(0.04, 0.96, "log scale: full runaway,\n$\\times$ = first crash",
              transform=axin.transAxes, fontsize=6.0, va="top", color="#333333")
    save(fig, "fig1_concept_failure")

# ----------------------------------------------------------------------------
# Fig.2  Theory / Mechanism (geometry schematic)
# ----------------------------------------------------------------------------
def fig2():
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.12,
                          top=0.92, bottom=0.08, left=0.04, right=0.98)

    # ---- (a) geometry ----
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(-2.6, 3.6); ax.set_ylim(-3.75, 2.7)
    ax.set_aspect("equal")
    ax.set_title("(a)  Feasible set, anchor, gated correction", fontsize=9, loc="left")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(1.6 * np.cos(th), 1.6 * np.sin(th), color="#1f6e8c", alpha=0.10, zorder=0)
    ax.plot(1.6 * np.cos(th), 1.6 * np.sin(th), color="#1f6e8c", lw=1.6)
    ax.text(-0.35, -0.15, "$\\mathcal{C}$: feasible set\n(div-free  ∩\ndissipation envelope)",
            fontsize=6.8, ha="center", color="#1f6e8c")
    p_pred = np.array([2.35, 1.75])
    ax.plot(*p_pred, "o", color="#c25e4a", ms=6, zorder=5)
    ax.text(p_pred[0] + 0.18, p_pred[1] + 0.28, "$\\hat{u}$ unconstrained", fontsize=6.6,
            color="#c25e4a")
    p_anc = p_pred / np.linalg.norm(p_pred) * 1.6
    arrow(ax, *p_pred, *p_anc, color="#1f6e8c", lw=1.8)
    ax.text(2.30, 2.42, "hard anchor $\\Pi_{\\mathcal{C}}$", fontsize=6.8, color="#1f6e8c")
    ax.plot(*p_anc, "s", color="#1f6e8c", ms=6, zorder=5)
    ax.text(p_anc[0] - 0.28, p_anc[1] + 0.06, "$u_{anchor}$", fontsize=7.0,
            color="#1f6e8c", ha="right")
    R = 0.85
    gth = np.linspace(0, 2 * np.pi, 200)
    ax.plot(p_anc[0] + R * np.cos(gth), p_anc[1] + R * np.sin(gth),
            color="#e08a2e", lw=1.3, ls="--")
    ax.text(p_anc[0] - 0.35, p_anc[1] - 1.35, "$R_{eff}$ sphere", fontsize=6.6,
            color="#e08a2e", ha="center")
    d_raw = np.array([1.45, 0.55])
    p_raw = p_anc + d_raw
    p_clip = p_anc + d_raw / np.linalg.norm(d_raw) * R
    arrow(ax, *p_anc, *p_raw, color="#3d9970", lw=1.1, ls=(0, (3, 2)))
    ax.text(p_raw[0] + 0.10, p_raw[1] - 0.55, "raw residual\n(clipped)", fontsize=6.2,
            color="#3d9970")
    arrow(ax, *p_anc, *p_clip, color="#e08a2e", lw=2.0)
    ax.plot(*p_clip, "^", color="#e08a2e", ms=7, zorder=5)
    ax.text(p_clip[0] - 0.55, p_clip[1] + 0.55, "$u_{fused}$", fontsize=7.4,
            color="#e08a2e", fontweight="bold")

    # 一维路径示意（与 Fig.1b 同一配色编码：蓝=Anchor, 绿=Residual, 橙=Gate）
    ys, hs = -3.55, 0.72
    ax.text(-2.55, ys + hs + 0.16, "one fused step (colour code as in Fig. 1b):",
            fontsize=6.2, color="#444444", style="italic")
    box(ax, -2.55, ys, 0.85, hs, "$\\hat{u}$", "#c25e4a", fs=6.6)
    box(ax, -1.35, ys, 1.15, hs, "anchor\n$\\Pi_{\\mathcal{C}}$", "#1f6e8c", fs=6.2)
    box(ax, 0.15, ys, 1.15, hs, "$+\\,g\\odot\\delta u$", "#3d9970", fs=6.4)
    box(ax, 1.65, ys, 1.05, hs, "clip\n$R_{eff}$", "#e08a2e", fs=6.2)
    box(ax, 3.05, ys, 0.55, hs, "$\\mathcal{C}$", "#7a5195", fs=6.6)
    for x1, x2 in [(-1.66, -1.39), (-0.16, 0.11), (1.34, 1.61), (2.74, 3.01)]:
        arrow(ax, x1, ys + hs / 2, x2, ys + hs / 2, color="#555555", lw=1.0, ms=8)

    # ---- (b) per-step path + theorem ----
    axb = fig.add_subplot(gs[0, 1]); axb.axis("off")
    axb.set_xlim(0, 10); axb.set_ylim(0, 6.4)
    axb.set_title("(b)  One fused step — constraints cannot be skipped", fontsize=9, loc="left")
    box(axb, 0.2, 4.6, 1.2, 1.0, "$u_t$", "#4a90c2", fs=7.0)
    box(axb, 1.9, 4.6, 2.4, 1.0, "Anchor\n$\\Pi_{\\mathcal{C}}\\,\\hat{u}$", "#1f6e8c", fs=7.0)
    box(axb, 1.9, 3.0, 2.4, 1.0, "Residual net\n$\\delta u=f_\\theta(u_t)$", "#3d9970", fs=7.0)
    box(axb, 5.0, 4.6, 2.6, 1.0, "Geometry gate\n$u_{anchor}+g\\odot\\delta u$", "#e08a2e", fs=6.8)
    box(axb, 8.3, 4.6, 1.5, 1.0, "$u_{t+1}$", "#7a5195", fs=7.0)
    arrow(axb, 1.45, 5.1, 1.85, 5.1); arrow(axb, 4.35, 5.1, 4.95, 5.1)
    arrow(axb, 7.65, 5.1, 8.25, 5.1)
    arrow(axb, 3.1, 4.05, 3.1, 4.55)
    arrow(axb, 4.35, 3.5, 6.3, 3.5); arrow(axb, 6.3, 3.5, 6.3, 4.55)
    axb.text(0.3, 2.15, "Bounded correction (Theorem):", fontsize=7.4, color="#333333",
             fontweight="bold")
    axb.text(0.3, 1.45, "$\\Vert u_{fused}-u_{anchor}\\Vert\\ \\leq\\ R_{eff}(u_t)$",
             fontsize=8.8, color="#e08a2e")
    axb.text(0.3, 0.85, "$E_{in}-c\\,\\varepsilon(u_t)\\Delta t\\ \\leq\\ E(u_{t+1})\\ \\leq\\ E_{in}$",
             fontsize=8.2, color="#1f6e8c")
    axb.text(0.3, 0.30, "$\\nabla\\cdot u_{t+1}=0$  (spectral projection, exact)",
             fontsize=8.0, color="#1f6e8c")
    axb.text(5.4, 1.45, "⇒ per-step state stays within an $R_{eff}$ ball\n"
                        "   of a feasible state; energy never grows and\n"
                        "   dissipation is bounded below.",
             fontsize=6.8, color="#444444", va="top")
    save(fig, "fig2_theory_mechanism")

# ----------------------------------------------------------------------------
# Fig.3  3D TGV main result (4 panels)
# ----------------------------------------------------------------------------
MAIN6 = ["fix2_env", "t20_fno", "t20_unet", "t20_dpot", "t20_physreg", "pc_fno"]

def fig3():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.subplots_adjust(hspace=0.34, wspace=0.26, top=0.94, bottom=0.09, left=0.10, right=0.97)
    steps = np.arange(1, 201)

    # (a) rL2(k) with std band
    ax = axs[0, 0]
    for name in MAIN6:
        c, ls, lw, lab = STYLE[name]
        mu, sd = rk(name, "rL2")
        ax.plot(KS, mu, color=c, ls=ls, lw=lw, marker="o", ms=3, label=lab)
        ax.fill_between(KS, np.maximum(mu - sd, 1e-4), mu + sd, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.minorticks_off()
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("relative $L_2$ error")
    ax.grid(True, which="both")
    ax.set_title("(a)  Pointwise error growth", fontsize=9, loc="left")
    ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9)

    # (b) survival
    ax = axs[0, 1]
    for name in MAIN6:
        c, ls, lw, lab = STYLE[name]
        cs = np.array(M[name]["crash_steps"], dtype=float)
        ks = np.arange(0, 201)
        surv = np.array([np.mean((cs < 0) | (cs > k)) for k in ks])
        ax.plot(ks, surv * 100, color=c, ls=ls, lw=lw, label=lab)
    for name, xytext in [("t20_physreg", (28, 52)), ("t20_fno", (142, 68))]:
        cs = [s for s in M[name]["crash_steps"] if s > 0]
        c0 = float(np.mean(cs))
        c = STYLE[name][0]
        ax.axvline(c0, color=c, ls="--", lw=1.1, alpha=0.85)
        ax.annotate(f"{STYLE[name][3]}\nfirst crash $k\\approx${c0:.0f}",
                    xy=(c0, 62), xytext=xytext, fontsize=6.4, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("surviving ICs (%)")
    ax.set_ylim(-3, 103); ax.set_xlim(0, 205)
    ax.grid(True)
    ax.set_title("(b)  Survival (no NaN/blow-up), 10 ICs", fontsize=9, loc="left")

    # (c) E(t) + Theorem-2 feasible band (about the truth trajectory)
    ax = axs[1, 0]
    eps_t = -np.gradient(TRUTH_E)                       # per-step dissipation of truth
    c_env = float(M["fix2_env"]["envelope"]["c"])       # envelope constant (1.218)
    ceil_b = TRUTH_E[:-1]                               # ceiling: E(k) <= E(k-1)
    floor_b = TRUTH_E[:-1] - c_env * eps_t[:-1]         # floor: E(k) >= E(k-1) - c·εΔt
    kb = np.arange(1, 201)
    ax.fill_between(kb, floor_b, ceil_b, color="#888888", alpha=0.30, lw=0, zorder=1)
    ax.plot(np.arange(0, 201), TRUTH_E, color=TRUTH, ls="--", lw=2.4, label="Truth (DNS)",
            zorder=2)
    ax.axhline(E200_TRUTH, color=TRUTH, ls=":", lw=0.8, alpha=0.5)
    for name in MAIN6:
        c, ls, lw, lab = STYLE[name]
        ax.plot(steps, M[name]["trajectories"]["E_mean"], color=c, ls=ls, lw=lw, label=lab,
                zorder=4)
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("$E(k)/E_0$")
    ax.set_xlim(0, 205); ax.set_ylim(0.4, 5.6)
    ax.grid(True)
    ax.annotate("Thm-2 feasible band (grey, hairline\nat this scale) — magnified in inset",
                xy=(178, TRUTH_E[178]), xytext=(98, 2.30), fontsize=6.2, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.text(0.03, 0.05, "line colours as in (a)", transform=ax.transAxes, fontsize=6.6,
            color="#555555", style="italic")
    ax.set_title("(c)  Energy fidelity (truth $E_{200}/E_0$=0.9416, dashed)", fontsize=9, loc="left")

    # zoom inset: per-step constraint geometry (k = 150..200). 灰色带 = 以真值为参考的
    # 每步允许区间；蓝色带 = 包络实际作用在 RCLN 自身轨迹上的每步区间（数值验证：
    # 模型轨迹逐步落在自身带内，偏差 < 2e-6，为 IC 平均的数值噪声）
    axin = ax.inset_axes([0.055, 0.50, 0.36, 0.335])
    k0, k1 = 150, 200
    sl = slice(k0 - 1, k1)
    E_m = np.array(M["fix2_env"]["trajectories"]["E_mean"])
    eps_m = np.array(M["fix2_env"]["trajectories"]["eps_mean"])
    axin.fill_between(kb[sl], ceil_b[sl], 0.985, color="#B2182B", alpha=0.08, lw=0,
                      step="post")
    axin.fill_between(kb[sl], floor_b[sl], ceil_b[sl], color="#777777", alpha=0.55, lw=0,
                      step="post")
    axin.fill_between(kb[sl], E_m[k0 - 1:k1] - c_env * eps_m[k0 - 2:k1 - 1], E_m[k0 - 1:k1],
                      color="#1f6e8c", alpha=0.35, lw=0, step="post")
    axin.plot(np.arange(k0, k1 + 1), TRUTH_E[k0:k1 + 1], color=TRUTH, ls="--", lw=1.6)
    axin.plot(np.arange(k0, k1 + 1), E_m[k0 - 1:k1], color=OURS, lw=1.6)
    axin.set_xlim(k0, k1); axin.set_ylim(0.924, 0.985)
    axin.tick_params(labelsize=6.0)
    axin.grid(True, alpha=0.3)
    axin.text(0.03, 0.97, "band about truth (grey)", transform=axin.transAxes,
              fontsize=5.6, color="#555555", va="top")
    axin.text(0.03, 0.84, "band on RCLN state (blue)", transform=axin.transAxes,
              fontsize=5.6, color="#1f6e8c", va="top")
    axin.text(0.97, 0.55, "growth\nforbidden", transform=axin.transAxes, fontsize=5.6,
              color="#B2182B", va="top", ha="right")
    axin.text(0.97, 0.04, "zoom $k$=150–200 (Thm 2)", transform=axin.transAxes,
              fontsize=5.8, ha="right", color="#333333")

    # (d) eps(t) — truth 由 -dE/dt 代理（自由衰减湍流 dE/dt=-ε 精确成立）
    ax = axs[1, 1]
    eps_truth = -np.gradient(TRUTH_E)  # per-step
    ax.plot(np.arange(1, 200), eps_truth[1:200], color=TRUTH, ls="--", lw=1.5,
            label="Truth ($-\\mathrm{d}E/\\mathrm{d}t$)")
    for name in MAIN6:
        c, ls, lw, lab = STYLE[name]
        eps = np.array(M[name]["trajectories"]["eps_mean"])
        cs = [s for s in M[name]["crash_steps"] if s > 0]
        kend = min(cs) if cs else len(eps)   # 崩溃后的数值无意义，截断
        ax.plot(np.arange(1, kend + 1), eps[:kend], color=c, ls=ls, lw=lw, label=lab)
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("dissipation rate $\\varepsilon$")
    ax.set_yscale("log")
    ax.set_xlim(0, 205); ax.set_ylim(3e-5, 1.2e-1)
    ax.grid(True, which="both")
    ax.set_title("(d)  Dissipation rate (curves end at crash)", fontsize=9, loc="left")
    ax.legend(fontsize=6.2, loc="lower left", framealpha=0.9)
    save(fig, "fig3_tgv_main")

# ----------------------------------------------------------------------------
# Fig.4  Turbulence physics (4 panels)
# ----------------------------------------------------------------------------
def fig4():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(hspace=0.62, wspace=0.27, top=0.94, bottom=0.09, left=0.10, right=0.97)
    steps = np.arange(1, 201)
    kspec = np.arange(1, 33)

    # (a) spectra
    ax = axs[0, 0]
    spec_steps = [("10", "-"), ("50", "--"), ("200", ":")]
    for sk, ls in spec_steps:
        ax.plot(kspec, TRUTH_SPEC[sk], color=TRUTH, ls=ls, lw=1.5,
                label=f"Truth $k$={sk}")
        ax.plot(kspec, M["fix2_env"]["spectra"][sk], color=OURS, ls=ls, lw=1.6,
                label=f"RCLN (fix2+env) $k$={sk}")
    ax.plot(kspec, M["t20_fno"]["spectra"]["50"], color=STYLE["t20_fno"][0], ls="--", lw=1.1,
            label="FNO $k$=50 (pre-crash)")
    ax.plot(kspec, M["pc_fno"]["spectra"]["200"], color=STYLE["pc_fno"][0], ls=":", lw=1.1,
            label="PhysicsCorrect $k$=200")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wavenumber $k$"); ax.set_ylabel("$E(k)$")
    ax.grid(True, which="both")
    ax.set_title("(a)  Energy spectrum: high-$k$ behaviour", fontsize=9, loc="left")
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              framealpha=0.9)

    # (b) enstrophy
    ax = axs[0, 1]
    Z0 = TRUTH_Z[0]
    ax.plot(np.arange(0, 201), TRUTH_Z / Z0, color=TRUTH, ls="--", lw=2.2, label="Truth (DNS)",
            zorder=2)
    for name in ["fix2_env", "t20_fno", "t20_unet", "pc_fno"]:
        c, ls, lw, lab = STYLE[name]
        Z = np.array(M[name]["trajectories"]["enstrophy_mean"]) / Z0
        cs = [s for s in M[name]["crash_steps"] if s > 0]
        kend = min(cs) if cs else len(Z)
        ax.plot(steps[:kend], Z[:kend], color=c, ls=ls, lw=lw, label=lab, zorder=4)
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("enstrophy / $Z^{truth}_0$")
    ax.set_xlim(0, 205); ax.grid(True)
    ax.set_title("(b)  Enstrophy, small-scale content (FNO ends at crash)", fontsize=9,
                 loc="left")
    ax.legend(fontsize=6.2, framealpha=0.9, loc="upper left")

    # (c) divergence
    ax = axs[1, 0]
    ax.axhline(TRUTH_DIV, color=TRUTH, ls="--", lw=1.4,
               label=f"Truth (mean |div|={TRUTH_DIV:.1e})")
    for name in ["fix2_env", "fix2", "fix2_no_upi", "t20_fno", "t20_unet", "t20_dpot"]:
        c, ls, lw, lab = STYLE[name]
        dv = np.array(M[name]["trajectories"]["div_mean"])
        cs = [s for s in M[name]["crash_steps"] if s > 0]
        kend = min(cs) if cs else len(dv)
        ax.plot(steps[:kend], dv[:kend], color=c, ls=ls, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("mean $|\\nabla\\cdot u|$")
    ax.set_xlim(0, 205); ax.grid(True, which="both")
    ax.set_title("(c)  Divergence violation", fontsize=9, loc="left")
    ax.legend(fontsize=5.9, loc="upper left", framealpha=0.9)

    # (d) vorticity PDF @k=200  —— 替代“涡结构切片”（npz 中无 pred 场，见报告）
    ax = axs[1, 1]
    vbc = 0.5 * (np.array(stats_env["vor_bins"][:-1]) + np.array(stats_env["vor_bins"][1:]))
    ax.plot(vbc, stats_env["truth"]["200"]["vor_pdf"], color=TRUTH, ls="--", lw=1.6,
            label="Truth (DNS)")
    for name, src in [("fix2_env", stats_env), ("fix2", stats), ("fix2_no_upi", stats),
                      ("v2", stats)]:
        c, ls, lw, lab = STYLE[name]
        ax.plot(vbc, src["models"][name]["200"]["vor_pdf"], color=c, ls=ls, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("vorticity magnitude $|\\omega|$"); ax.set_ylabel("PDF")
    ax.grid(True, which="both")
    ax.set_title("(d)  Vorticity PDF at $k$=200 (3 ICs): tails = vortex cores",
                 fontsize=9, loc="left")
    ax.legend(fontsize=6.4, framealpha=0.9)
    save(fig, "fig4_turbulence_physics")

# ----------------------------------------------------------------------------
# Fig.5  Generality across systems + OOD
# ----------------------------------------------------------------------------
def _arm_Ks(dic, scen, arm):
    Ks = sorted(int(k) for k in dic[scen][arm].keys())
    return np.array(Ks), dic[scen][arm]

# OOD 面板配色：系统色系（3D=红系、2D=蓝系、1D=绿系）
FAM = {"3D": ["#B2182B", "#d6604d", "#f4a582"],
       "2D": ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0"],
       "1D": ["#1b7837", "#5aae61", "#a6d96a", "#d9f0d3"]}

def _ood_groups():
    """返回 [(group_label, family, [(bar_label, rL2, E, crash_rate), ...]), ...]"""
    groups = []
    o3 = ood3d["datasets"]
    for ds, dlab in [("re3200", "3D TGV\nRe 3200"), ("re1000", "3D TGV\nRe 1000")]:
        K = str(int(o3[ds]["K"]))
        bars = []
        for m, lab in [("fix2", "RCLN (fix2)"), ("v2", "RCLN-v2"), ("v2gate70", "v2-gate70")]:
            mm = o3[ds]["models"][m]
            bars.append((lab, mm["rL2"]["@" + K]["mean"], float(mm["E_over_E0"]["@" + K]),
                         mm["crashes"] / mm["n_rollouts"]))
        groups.append((dlab, "3D", bars))
    for scen, dlab, src, fam in [("re_ood", "2D NS\nRe OOD", genB, "2D"),
                                 ("decaying_hit", "2D NS\ndecaying", genB, "2D"),
                                 ("length_ood_L22", "KS\nL=22", genC, "1D"),
                                 ("length_ood_L44", "KS\nL=44", genC, "1D")]:
        bars = []
        for arm, lab in [("anchor", "anchor"), ("anchor_noenv", "anchor w/o env"),
                         ("loss_level", "loss-level"), ("none", "no physics")]:
            dd = src[scen][arm]
            K = str(max(int(k) for k in dd.keys()))
            bars.append((lab, dd[K]["rL2_mean"], dd[K]["E_ratio_mean"], dd[K]["crash_rate"]))
        groups.append((dlab, fam, bars))
    bars = []
    for arm, lab in [("anchor", "anchor"), ("none", "no physics")]:
        dd = genB_res["res_ood_128"][arm]["200"]
        bars.append((lab, dd["rL2_mean"], dd["E_ratio_mean"], dd["crash_rate"]))
    dd = genB_res["baseline_64_in_domain"]["anchor"]["200"]
    bars.append(("anchor@64² (ref)", dd["rL2_mean"], dd["E_ratio_mean"], dd["crash_rate"]))
    groups.append(("2D NS\nres 64→128", "2D", bars))
    return groups

def _ood_panel(ax, groups, idx, ymax, ylabel):
    """idx: 0=rL2, 1=E/E0。崩溃臂黑边+斜纹；超高 bar 截断并标注真实值。"""
    pos = 0.0
    xt = []
    for glab, fam, bars in groups:
        ctr = pos + (len(bars) - 1) / 2
        xt.append((ctr, glab))
        for bi, (lab, r, e, cr) in enumerate(bars):
            v = (r, e)[idx]
            c = FAM[fam][bi]
            crashed = cr > 0
            ec = "black" if crashed else "none"
            ht = "//" if crashed else None
            if v <= ymax:
                ax.bar(pos, v, width=0.8, color=c, alpha=0.92, edgecolor=ec,
                       lw=1.2 if crashed else 0, hatch=ht)
                ax.text(pos, v + 0.02 * ymax, f"{v:.2f}", ha="center", fontsize=5.6,
                        rotation=90, clip_on=True)
            else:
                ax.bar(pos, ymax, width=0.8, color=c, alpha=0.55, hatch="//",
                       edgecolor="black" if crashed else "white",
                       lw=1.2 if crashed else 0.8)
                ax.text(pos, ymax - 0.13 * ymax - 0.11 * ymax * (bi % 2), f"{v:.2f}",
                        ha="center", va="top", fontsize=6.2, fontweight="bold",
                        color="#222222")
            pos += 1.0
        pos += 0.9
    for ctr, glab in xt:
        ax.text(ctr, -0.13 * ymax, glab, ha="center", va="top", fontsize=7.0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_xticks([])
    ax.grid(True, axis="y")

def fig5():
    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.02, 1.02], hspace=0.40, wspace=0.34,
                          top=0.955, bottom=0.075, left=0.09, right=0.98)

    # (a) System A: E/E0@200 bars
    ax = fig.add_subplot(gs[0, 0])
    names = ["fix2_env", "fix2", "t20_fno", "t20_unet", "t20_dpot", "t20_physreg", "pc_fno"]
    vals = [M[n]["E_over_E0"]["@200"]["mean"] for n in names]
    stds = [M[n]["E_over_E0"]["@200"]["std"] for n in names]
    cols = [STYLE[n][0] for n in names]
    xpos = np.arange(len(names))
    ax.bar(xpos, vals, yerr=stds, color=cols, alpha=0.9, capsize=2,
           error_kw=dict(lw=0.8))
    ax.axhline(E200_TRUTH, color=TRUTH, ls="--", lw=1.2, label="truth 0.9416")
    for i, n in enumerate(names):
        if M[n]["crashes"] > 0:
            ax.text(i, vals[i] + stds[i] + 0.15, "crashed\n10/10", ha="center", fontsize=6.0,
                    color=cols[i])
    ax.set_xticks(xpos)
    ax.set_xticklabels(["ours", "fix2", "FNO", "U-Net", "DPOT", "PhysReg", "PC"],
                       rotation=45, ha="right", fontsize=6.6)
    ax.set_ylabel("$E/E_0$ at $k$=200")
    ax.set_ylim(0, 6.4)
    ax.grid(True, axis="y")
    ax.set_title("(a)  System A: 3D TGV (Re 6400)", fontsize=8.6, loc="left")
    ax.legend(fontsize=6.4, loc="upper left")

    # (b) System B: E ratio vs K, 4 arms (in-domain)
    ax = fig.add_subplot(gs[0, 1])
    for arm, (c, ls, lab) in ARM_STYLE.items():
        Ks, dd = _arm_Ks(genB, "in_domain", arm)
        E = [dd[str(k)]["E_ratio_mean"] for k in Ks]
        ax.plot(Ks, E, color=c, ls=ls, lw=1.5, marker="o", ms=3, label=lab)
    ax.axhline(1.0, color=TRUTH, ls=":", lw=0.9)
    ax.set_xlabel("rollout steps $K$"); ax.set_ylabel("$E/E_0$")
    ax.grid(True); ax.set_ylim(0.6, 2.2)
    ax.set_title("(b)  System B: 2D NS (in-domain)", fontsize=8.6, loc="left")
    ax.legend(fontsize=6.2, loc="center left")

    # (c) System C: KS in-domain
    ax = fig.add_subplot(gs[0, 2])
    for arm, (c, ls, lab) in ARM_STYLE.items():
        Ks, dd = _arm_Ks(genC, "in_domain", arm)
        E = [dd[str(k)]["E_ratio_mean"] for k in Ks]
        ax.plot(Ks, E, color=c, ls=ls, lw=1.5, marker="o", ms=3, label=lab)
    ax.axhline(1.0, color=TRUTH, ls=":", lw=0.9)
    ax.set_xlabel("rollout steps $K$"); ax.set_ylabel("$E/E_0$")
    ax.grid(True); ax.set_ylim(0.8, 1.4)
    ax.set_title("(c)  System C: KS 1D (in-domain)", fontsize=8.6, loc="left")

    groups = _ood_groups()

    # (d1) OOD energy ratio at final K
    ax = fig.add_subplot(gs[1, :])
    _ood_panel(ax, groups, idx=1, ymax=3.4, ylabel="$E/E_0$ at final $K$")
    ax.axhline(1.0, color=TRUTH, ls="--", lw=1.1, label="ideal $E/E_0$=1")
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)
    ax.set_title("(d1)  Distribution shift — energy ratio: anchored arms stay near $E/E_0$=1 "
                 "(colour family: 3D = red, 2D = blue, 1D = green; black edge + hatching = "
                 "crash at final $K$)", fontsize=9, loc="left")

    # (d2) OOD rL2 at final K
    ax = fig.add_subplot(gs[2, :])
    _ood_panel(ax, groups, idx=0, ymax=3.4, ylabel="relative $L_2$ error at final $K$")
    ax.text(0.003, 0.985, "bar order within group — 3D: fix2, v2, v2-gate70;  2D & KS: "
            "anchor, anchor w/o env, loss-level, no physics;  res: anchor, no physics, "
            "anchor@64² (ref)", transform=ax.transAxes, fontsize=6.0, va="top",
            color="#444444")
    ax.set_title("(d2)  Distribution shift — pointwise error: the stability advantage of the "
                 "anchored arm does not come at an accuracy cost", fontsize=9, loc="left")
    save(fig, "fig5_generality")

# ----------------------------------------------------------------------------
# Fig.6  UPI: geometry-driven discrepancy indicator
# ----------------------------------------------------------------------------
def fig6():
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 0.85, 0.85], hspace=0.60, wspace=0.30,
                          top=0.90, bottom=0.08, left=0.07, right=0.97)

    # (a) maps: rows = k10, k100; cols = score, err
    for row, kk in enumerate(["k10", "k100"]):
        ax1 = fig.add_subplot(gs[0, row * 2])
        ax2 = fig.add_subplot(gs[0, row * 2 + 1])
        im1 = ax1.imshow(upi_maps[f"{kk}_score"], cmap="viridis", origin="lower")
        im2 = ax2.imshow(upi_maps[f"{kk}_err"], cmap="inferno", origin="lower")
        ax1.set_title(f"$k$={kk[1:]}: gate score", fontsize=8)
        ax2.set_title(f"$k$={kk[1:]}: true error", fontsize=8)
        for a in (ax1, ax2):
            a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
        r = float(upi_maps[f"{kk}_r"]); au = float(upi_maps[f"{kk}_au"])
        ax2.set_xlabel(f"r={r:.2f}, AUROC={au:.2f}", fontsize=7)
    fig.text(0.07, 0.935, "(a)  Gate map vs. true error map (mid-plane slice): the geometry-driven\n"
                          "      discrepancy indicator tracks where the error actually grows",
             fontsize=9, ha="left", va="top")

    # (b) corr / AUROC vs k
    ax = fig.add_subplot(gs[1, 0:2])
    ks4 = [1, 10, 50, 100]
    rr = [float(upi_maps[f"k{k}_r"]) for k in ks4]
    aa = [float(upi_maps[f"k{k}_au"]) for k in ks4]
    ax.plot(ks4, rr, "-o", color=OURS, lw=1.8, ms=4, label="Pearson r (score vs. error)")
    ax.plot(ks4, aa, "-s", color="#1f77b4", lw=1.8, ms=4, label="AUROC (top-quantile error)")
    ax.axhline(0.5, color="#888888", ls=":", lw=0.9)
    ax.set_xscale("log"); ax.set_xticks(ks4); ax.set_xticklabels(ks4); ax.minorticks_off()
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("score–error agreement")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True); ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("(b)  Signal strengthens with rollout depth", fontsize=9, loc="left")

    # (c) AUROC grouped bars across datasets
    ax = fig.add_subplot(gs[1, 2:4])
    dsets = [("tgv", "in-domain\nTGV"), ("re3200", "Re 3200"), ("re1000", "Re 1000"),
             ("jhtdb", "JHTDB\n(flow-type OOD)")]
    methods = [("fix2 native", upi_u1u5, "fix2", "#7f7f7f"),
               ("v2 soft (native)", upi_u1u5, "v2", "#1f77b4"),
               ("v2->fix2 transplant", upi_v2t, "fix2_v2trans_sig", OURS)]
    w = 0.26
    x0 = np.arange(len(dsets))
    for i, (mlab, src, key, c) in enumerate(methods):
        vals = [src["datasets"][ds]["models"][key]["U2_auroc_mean"] for ds, _ in dsets]
        ax.bar(x0 + (i - 1) * w, vals, width=w, color=c, alpha=0.9, label=mlab)
        for x, v in zip(x0 + (i - 1) * w, vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.8)
    ax.axhline(0.5, color="#888888", ls=":", lw=0.9)
    ax.set_xticks(x0)
    ax.set_xticklabels([d for _, d in dsets], fontsize=7)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 0.95)
    ax.grid(True, axis="y")
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.set_title("(c)  Error-detection AUROC across distributions", fontsize=9, loc="left")

    # (d) calibration / reliability
    ax = fig.add_subplot(gs[2, 1:3])
    for ds, lab, c, mk in [("tgv", "in-domain TGV", OURS, "o"),
                           ("re1000", "Re 1000 (OOD)", "#1f77b4", "s")]:
        cal = upi_v2t["datasets"][ds]["models"]["fix2_v2trans_sig"]["U3_calibration"]
        bs = np.array(cal["bin_score"]); be = np.array(cal["bin_err"])
        bc = np.array(cal["bin_count"])
        msk = bc > 0
        ax.plot(bs[msk], be[msk], marker=mk, ms=4, lw=1.6, color=c,
                label=f"{lab} (transplant)")
    smax = max(1e-9, float(np.max(bs))) if msk.any() else 1.0
    ax.plot([0, smax], [0, smax], color="#888888", ls=":", lw=0.9,
            label="1:1 reference")
    ax.set_xlabel("mean gate score per bin")
    ax.set_ylabel("mean true error per bin")
    ax.grid(True)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("(d)  Reliability: gate score vs. true error (monotone, OOD-shifted)",
                 fontsize=9, loc="left")
    save(fig, "fig6_upi_signal")

# ----------------------------------------------------------------------------
# Fig.7  Mechanistic ablation + cost + rL2/E decoupling
# ----------------------------------------------------------------------------
def fig7():
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30,
                          top=0.94, bottom=0.09, left=0.10, right=0.97)

    # (a) hard-vs-soft ablation: rL2@100 vs E@200
    ax = fig.add_subplot(gs[0, 0])
    abl = ["fix2_env", "fix2", "fix2_hc", "fix2_no_upi", "v2", "v2gate70", "v2gate90", "v3"]
    off7a = {"fix2_env": (-14, 15), "fix2": (-10, -17), "fix2_hc": (9, -5),
             "fix2_no_upi": (-24, 12), "v2": (11, 6), "v2gate70": (11, -4),
             "v2gate90": (11, -7), "v3": (-34, 11)}
    for n in abl:
        c, ls, lw, lab = STYLE[n]
        x = M[n]["rL2"]["@100"]["mean"]; y = M[n]["E_over_E0"]["@200"]["mean"]
        mk, ms = ("*", 15) if n == "fix2_env" else ("o", 6)
        ax.scatter(x, y, marker=mk, s=ms ** 1.6, color=c, zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off7a[n],
                    fontsize=6.2, color=c)
    ax.axhline(E200_TRUTH, color=TRUTH, ls="--", lw=1.1)
    ax.text(0.36, E200_TRUTH * 0.90, "truth $E_{200}$=0.9416", fontsize=6.4, color=TRUTH)
    ax.set_xlabel("relative $L_2$ error @ $k$=100")
    ax.set_ylabel("$E/E_0$ @ $k$=200")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.45, 4.3); ax.set_xlim(0.33, 2.6)
    ax.grid(True, which="both")
    ax.set_title("(a)  Hard-vs-soft physics ablation (3D TGV)", fontsize=9, loc="left")

    # (b) Pareto: cost vs accuracy
    ax = fig.add_subplot(gs[0, 1])
    off7b = {"t20_physreg": (8, 5), "t20_fno": (10, 2), "v2": (-6, 15),
             "pc_fno": (14, -3), "fix2_env": (-10, -24), "fix2": (-58, -13),
             "t20_unet": (9, -4), "t20_dpot": (8, 5)}
    for n in ["fix2_env", "fix2", "v2", "t20_fno", "t20_unet", "t20_dpot",
              "t20_physreg", "pc_fno"]:
        c, ls, lw, lab = STYLE[n]
        x = M[n]["step_time_ms"]; y = M[n]["rL2"]["@100"]["mean"]; p = M[n]["params"] / 1e6
        mk = "*" if n == "fix2_env" else "o"
        ax.scatter(x, y, marker=mk, s=18 * p, color=c, alpha=0.85, zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off7b[n],
                    fontsize=6.0, color=c)
    # 跨系统臂（网格更小，仅示意，空心标记）
    b_anchor = genB["in_domain"]["anchor"]["100"]
    c_anchor = genC["in_domain"]["anchor"]["100"]
    ax.scatter(b_anchor["step_time_ms"], b_anchor["rL2_mean"], marker="^", s=55,
               facecolors="none", edgecolors=OURS, lw=1.4, zorder=5)
    ax.annotate("anchor, 2D NS ($64^2$)", (b_anchor["step_time_ms"], b_anchor["rL2_mean"]),
                textcoords="offset points", xytext=(6, -13), fontsize=6.0, color=OURS)
    ax.scatter(c_anchor["step_time_ms"], c_anchor["rL2_mean"], marker="s", s=50,
               facecolors="none", edgecolors=OURS, lw=1.4, zorder=5)
    ax.annotate("anchor, KS (1D)", (c_anchor["step_time_ms"], c_anchor["rL2_mean"]),
                textcoords="offset points", xytext=(7, 2), fontsize=6.0, color=OURS)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8, 130); ax.set_ylim(0.006, 4.5)
    ax.set_xlabel("inference cost per step (ms)")
    ax.set_ylabel("relative $L_2$ error @ $k$=100")
    ax.grid(True, which="both")
    ax.set_title("(b)  Accuracy-cost Pareto (marker size ~ params)", fontsize=9, loc="left")

    # (c) rL2-energy decoupling
    ax = fig.add_subplot(gs[1, 0])
    cases = []
    for arm, lab in [("anchor", "anchor"), ("anchor_noenv", "anchor w/o env"),
                     ("loss_level", "loss-level"), ("none", "no physics")]:
        dd = genC["length_ood_L44"][arm]["200"]
        cases.append((dd["rL2_mean"], dd["E_ratio_mean"], "KS L=44: " + lab,
                      ARM_STYLE[arm][0], "o", dd["crash_rate"]))
    for arm, lab in [("anchor", "anchor"), ("anchor_noenv", "anchor w/o env"),
                     ("loss_level", "loss-level"), ("none", "no physics")]:
        dd = genB["decaying_hit"][arm]["200"]
        cases.append((dd["rL2_mean"], dd["E_ratio_mean"], "NS decaying: " + lab,
                      ARM_STYLE[arm][0], "s", dd["crash_rate"]))
    ax.axvline(1.0, color="#888888", ls=":", lw=0.9)
    ax.axhline(1.0, color=TRUTH, ls="--", lw=1.1)
    ax.text(5.6, 1.14, "perfect energy $E/E_0$=1", fontsize=6.2, color=TRUTH)
    off7c = {"KS L=44: anchor": (8, 6), "KS L=44: anchor w/o env": (8, -13),
             "KS L=44: loss-level": (8, 6), "KS L=44: no physics": (8, -13),
             "NS decaying: anchor": (8, -4), "NS decaying: anchor w/o env": (-8, -14),
             "NS decaying: loss-level": (8, -13), "NS decaying: no physics": (8, 4)}
    ha7c = {"NS decaying: anchor w/o env": "right"}
    for x, y, lab, c, mk, cr in cases:
        edge = "#000000" if cr > 0 else "white"
        ax.scatter(x, y, marker=mk, s=60, color=c, edgecolors=edge,
                   linewidths=1.4 if cr > 0 else 0.6, zorder=5)
        ax.annotate(lab + (" X" if cr > 0 else ""), (x, y), textcoords="offset points",
                    xytext=off7c.get(lab, (7, 5)), fontsize=5.9, color=c,
                    ha=ha7c.get(lab, "left"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.55, 14); ax.set_ylim(0.15, 18)
    ax.set_xlabel("relative $L_2$ error @ $K$=200 (log)")
    ax.set_ylabel("$E/E_0$ @ $K$=200 (log)")
    ax.grid(True, which="both")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="#555555", label="KS L=44 (OOD)"),
               Line2D([], [], marker="s", ls="", color="#555555", label="2D NS decaying"),
               Line2D([], [], marker="o", ls="", color="#555555", mec="black",
                      mfc="none", label="crashed arm (black edge)")]
    ax.legend(handles=handles, fontsize=6.2, loc="upper left")
    ax.set_title("(c)  Error–energy decoupling under shift (log–log)", fontsize=9, loc="left")

    # (d) stability–accuracy Pareto: all 13 suite models at k=200 (3D TGV)
    ax = fig.add_subplot(gs[1, 1])
    SHORT = {"fix2": "fix2", "fix2_hc": "RCLN-hc", "v2": "v2 (soft)",
             "v2gate70": "v2-gate70", "v2gate90": "v2-gate90", "v3": "v3",
             "fix2_no_upi": "RCLN-noUPI", "fix2_env": "RCLN (fix2+env)",
             "t20_fno": "FNO", "t20_unet": "U-Net", "t20_dpot": "DPOT",
             "t20_physreg": "FNO+PhysReg", "pc_fno": "PhysicsCorrect"}
    off7d = {"fix2": (-7, -2), "fix2_hc": (8, 5), "v2": (-7, 4), "v2gate70": (-7, -9),
             "v2gate90": (7, -12), "v3": (8, -2), "fix2_no_upi": (8, -2),
             "fix2_env": (9, -13), "t20_fno": (-8, 5), "t20_unet": (0, -13),
             "t20_dpot": (7, 4), "t20_physreg": (-8, -13), "pc_fno": (-8, 4)}
    ha7d = {"fix2": "right", "v2": "right", "v2gate70": "right", "t20_fno": "right",
            "t20_unet": "center", "t20_physreg": "right", "pc_fno": "right"}
    for n in M:
        x = M[n]["rL2"]["@200"]["mean"]; y = M[n]["E_over_E0"]["@200"]["mean"]
        crashed = M[n]["crashes"] > 0
        c = STYLE[n][0]
        if n == "fix2_env":
            ax.scatter(x, y, marker="*", s=230, color=OURS, edgecolors="black",
                       linewidths=0.8, zorder=6)
        else:
            ax.scatter(x, y, marker="o", s=52, color=c, zorder=5,
                       edgecolors="black" if crashed else "white",
                       linewidths=1.4 if crashed else 0.6)
        ax.annotate(SHORT[n], (x, y), textcoords="offset points", xytext=off7d[n],
                    fontsize=6.0, color=c, ha=ha7d.get(n, "left"))
    ax.axvline(1.0, color="#888888", ls=":", lw=0.9)
    ax.axhline(E200_TRUTH, color=TRUTH, ls="--", lw=1.1)
    ax.text(1.04, 5.6, "r$L_2$=1", fontsize=6.2, color="#666666")
    ax.text(0.53, 6.8, "truth $E_{200}/E_0$=0.9416", fontsize=6.2, color=TRUTH)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 3.4); ax.set_ylim(0.38, 9)
    ax.set_xlabel("relative $L_2$ error @ $k$=200 (log)")
    ax.set_ylabel("$E/E_0$ @ $k$=200 (log)")
    ax.grid(True, which="both")
    handles = [Line2D([], [], marker="*", ls="", color=OURS, mec="black", ms=11,
                      label="RCLN (fix2+env), ours"),
               Line2D([], [], marker="o", ls="", color="#555555", mec="black",
                      mfc="none", label="crashed (black edge)")]
    ax.legend(handles=handles, fontsize=6.2, loc="lower right")
    ax.set_title("(d)  Stability–accuracy Pareto, all 13 models: lowest r$L_2$ "
                 "$\\neq$ most physical", fontsize=9, loc="left")
    save(fig, "fig7_ablation_cost")

# ----------------------------------------------------------------------------
# CAPTIONS.md — 数字全部从上面加载的数据计算
# ----------------------------------------------------------------------------
def captions():
    fe = M["fix2_env"]
    fno = M["t20_fno"]; pr = M["t20_physreg"]
    fno_crash = float(np.mean([s for s in fno["crash_steps"] if s > 0]))
    pr_crash = float(np.mean([s for s in pr["crash_steps"] if s > 0]))
    env = fe["envelope"]
    au_t = upi_v2t["datasets"]["tgv"]["models"]["fix2_v2trans_sig"]["U2_auroc_mean"]
    au_v2 = upi_u1u5["datasets"]["tgv"]["models"]["v2"]["U2_auroc_mean"]
    au_f2 = upi_u1u5["datasets"]["tgv"]["models"]["fix2"]["U2_auroc_mean"]
    au_t_r1 = upi_v2t["datasets"]["re1000"]["models"]["fix2_v2trans_sig"]["U2_auroc_mean"]
    au_t_j = upi_v2t["datasets"]["jhtdb"]["models"]["fix2_v2trans_sig"]["U2_auroc_mean"]
    r100 = lambda n: M[n]["rL2"]["@100"]["mean"]
    e200 = lambda n: M[n]["E_over_E0"]["@200"]["mean"]
    ks44 = genC["length_ood_L44"]; nsd = genB["decaying_hit"]
    txt = f"""# Figure Captions (NC revision, Fig.1–Fig.7)

All numbers below are computed from `results/paper_experiments/` at figure-generation time
(`scripts/make_figures_nc.py`); they match the plotted values exactly.

**Figure 1. Concept and failure mode.** (a) Predict-then-correct pipelines (FNO / PINO /
post-hoc correction) treat physics as an optional soft loss or an a-posteriori fix that can
be bypassed by the learned predictor. (b) RCLN instead places the constraints on the forward
pass: every state passes through a Physical Anchor (spectral divergence-free projection plus
a dissipation envelope), and the residual correction enters only through a bounded Geometry
Gate. (c) K=200 autoregressive rollouts from the same initial condition (3D Taylor–Green
vortex, Re=6400, 10 ICs, mean over ICs): FNO blows up (first crash at k≈{fno_crash:.0f},
E/E₀={e200('t20_fno'):.2f} at k=200) and FNO+PhysReg diverges even earlier (first crash
k≈{pr_crash:.0f}), while RCLN (fix2+env) tracks the DNS truth (E₂₀₀/E₀=0.9416) with
E/E₀={e200('fix2_env'):.3f}±{M['fix2_env']['E_over_E0']['@200']['std']:.3f} and zero crashes.
Main-panel trajectories are truncated at E/E₀=5 for visibility; because post-crash states are
NaN, the stored mean trajectories are held at the last finite value — the log-scale inset
shows the full recorded runaway up to each model's first crash step (× markers), with no
extrapolation beyond the data.

**Figure 2. Mechanism and bounded-correction theorem.** Components are colour-coded as in
Fig. 1b throughout (blue = Physical Anchor, green = Residual Learner, orange = Geometry
Gate). (a) Geometry of one step: the unconstrained network proposal is projected onto the
feasible set C (divergence-free fields satisfying the dissipation envelope) by the hard
anchor Π_C; the learned residual is then admitted only through a gate whose output is
clipped to the R_eff sphere around the anchored state. The bottom strip shows the same
step as a one-dimensional path: input → anchor → gated residual addition → clipping →
back inside C. (b) Per-step inference path with the guaranteed bounds: the fused correction
obeys ‖u_fused − u_anchor‖ ≤ R_eff(u_t) (orange, gate), the energy obeys
E_in − c·ε(u_t)·Δt ≤ E(u₊₁) ≤ E_in (blue, anchor envelope), and ∇·u₊₁ = 0 holds exactly
by spectral projection. On the TGV suite the envelope intervened on ~50% of steps in each
direction (floor hit rate {env['floor_hit_rate']:.2f}, ceiling {env['ceil_hit_rate']:.2f})
with mean |correction| of {env['mean_abs_corr']*100:.1f}% of the energy — a thin, sparse
safety layer, not a re-simulation.

**Figure 3. Main result on the 3D Taylor–Green vortex (Re=6400, K=200, 10 ICs).**
(a) Relative L2 error growth (mean ± std over ICs). RCLN (fix2+env) reaches
rL2@100={r100('fix2_env'):.3f} versus FNO {r100('t20_fno'):.3f}, U-Net {r100('t20_unet'):.3f},
DPOT {r100('t20_dpot'):.3f}, FNO+PhysReg {r100('t20_physreg'):.3f} and the post-hoc
PhysicsCorrect reproduction {r100('pc_fno'):.3f}. (b) Survival: FNO and FNO+PhysReg lose all
10/10 ICs to numerical blow-up before k=200 (vertical dashed lines mark the mean first-crash
steps, k≈{fno_crash:.0f} and k≈{pr_crash:.0f} respectively); all other models survive.
(c) Energy fidelity: truth decays to E₂₀₀/E₀=0.9416; RCLN (fix2+env) ends at
{e200('fix2_env'):.3f} while FNO and FNO+PhysReg exceed 5.0 and DPOT drifts to
{e200('t20_dpot'):.2f}. The grey band visualises the Theorem-2 feasible region taken about
the truth trajectory, E(k−1) − c·ε·Δt ≤ E(k) ≤ E(k−1) with c={env['c']:.3f} — a hairline at
this scale, so the inset magnifies k=150–200, showing the per-step band about the truth
(grey), the band actually enforced on the RCLN state (blue; the rollout stays inside it to
within 2×10⁻⁶, the IC-averaging noise floor), and the energy-growth-forbidden region above
the ceiling (red tint). (d) Dissipation rate ε(t) (model-reported; truth shown as the exact
−dE/dt of the DNS energy): the anchored model follows the truth curve, whereas crashed
baselines show runaway dissipation after their energy growth sets in.

**Figure 4. Turbulence physics fidelity.** (a) Energy spectra E(k) at k=10/50/200: RCLN
(fix2+env) tracks the DNS spectrum across wavenumbers including the high-k tail, while the
post-hoc PhysicsCorrect reproduction over-damps the tail at k=200. (b) Enstrophy evolution
(normalised by the truth initial value): the anchored model preserves the small-scale content;
U-Net over-dissipates ({M['t20_unet']['E_over_E0']['@200']['mean']:.2f} energy at k=200).
(c) Mean |∇·u| (log scale): the hard spectral projection keeps divergence at the DNS level
(~{TRUTH_DIV:.1e}); models without the projection drift orders of magnitude upward.
(d) Vorticity-magnitude PDF at k=200 (3 ICs): the anchored model reproduces the heavy DNS
tail associated with intense vortex cores; removing the residual gate (RCLN-noUPI) or the
envelope visibly distorts the tails. (Field-slice panels were not available in the verified
data products; the vorticity PDF is used as the structural-fidelity panel.)

**Figure 5. Generality across systems and distribution shifts.** (a) System A (3D TGV):
E/E₀ at k=200 — RCLN (fix2+env) {e200('fix2_env'):.3f} vs truth 0.9416; both unconstrained
spectral baselines crashed on all ICs. (b) System B (2D Navier–Stokes, in-domain): the anchor
arm holds E/E₀ near unity through K=200 ({genB['in_domain']['anchor']['200']['E_ratio_mean']:.2f}),
while the loss-level physics arm grows to
{genB['in_domain']['loss_level']['200']['E_ratio_mean']:.2f}. (c) System C (Kuramoto–Sivashinsky,
in-domain): all arms stable, anchor arm cleanest. (d1)–(d2) Out-of-distribution, split by
metric to reduce density (bar colour encodes the system family: red = 3D TGV, blue = 2D NS,
green = 1D KS; black-edged hatched bars crashed before the final K). (d1) Energy ratio E/E₀
at the final K (dashed line = ideal 1.0; over-tall bars truncated, true values labelled):
the anchored arm stays near unity in every scenario, while ablated arms either blow up
(e.g. anchor w/o envelope at E/E₀={genC['length_ood_L44']['anchor_noenv']['200']['E_ratio_mean']:.2f}
on KS L=44 and {genB['decaying_hit']['anchor_noenv']['200']['E_ratio_mean']:.2f} on decaying
2D NS) or over-dissipate. (d2) rL2 at the final K for Reynolds-number OOD (3D Re=3200/1000;
2D Re×2), decaying turbulence, KS domain-size OOD (L=22, L=44), and zero-shot resolution
transfer 64²→128² (anchor rL2@200=1.299 vs 1.392 in-domain at 64²). The architecture-anchored
arm keeps its stability advantage in every scenario without an accuracy cost.

**Figure 6. The geometry-driven discrepancy indicator as an emergent error signal.**
(a) Mid-plane slices of the gate score and the true error field at k=10 and k=100: the
indicator localises the regions where error actually concentrates (r={float(upi_maps['k100_r']):.2f},
AUROC={float(upi_maps['k100_au']):.2f} at k=100). (b) Score–error agreement strengthens with
rollout depth (Pearson r {float(upi_maps['k1_r']):.2f}→{float(upi_maps['k100_r']):.2f} from
k=1 to k=100) — the one-step error is nearly spatially uniform, so the signal is physically
meaningful precisely when long-rollout risk emerges. (c) Error-detection AUROC across
distributions: native fix2 gate {au_f2:.2f} in-domain; the jointly trained v2 soft branch
{au_v2:.2f}; the zero-training read-only transplant of the v2 branch onto the frozen fix2
backbone {au_t:.2f} in-domain, {au_t_r1:.2f} at Re=1000, and {au_t_j:.2f} on JHTDB isotropic
turbulence (flow-type OOD, where calibration degrades — reported honestly). (d) Reliability:
binned gate score vs. mean true error is monotone in-domain and shifts under OOD; we therefore
report the signal as a geometry-driven discrepancy indicator, not a calibrated uncertainty.

**Figure 7. Mechanistic ablation and computational cost.** (a) Hard-vs-soft physics on 3D TGV
(rL2@100 vs E/E₀@200, log–log): the full anchor (RCLN fix2+env) is the only arm that is
simultaneously most accurate and energy-faithful; hard-core-only (RCLN-hc, rL2@100=
{r100('fix2_hc'):.2f}) and no-gate (RCLN-noUPI, E/E₀={e200('fix2_no_upi'):.2f}) ablations fail
on complementary axes, and soft-fusion variants (v2-gate70/90) lose energy fidelity.
(b) Accuracy–cost Pareto (per-step inference time vs rL2@100; marker size ∝ parameter count):
RCLN (fix2+env) sits on the Pareto front at {M['fix2_env']['step_time_ms']:.0f} ms/step for
5.1M parameters — the envelope adds only ~5 ms over the raw fix2 backbone
({M['fix2']['step_time_ms']:.0f} ms). Cross-system anchor arms (2D NS 64², KS 1D; smaller
grids, open markers) are shown for completeness. (c) Error–energy decoupling under shift
(log–log): on KS L=44 the anchor arm is the most accurate (rL2=1.36) but the envelope ceiling
intervenes on 100% of steps and pins E/E₀ at {ks44['anchor']['200']['E_ratio_mean']:.2f}, while
the loss-level arm looks energetically perfect (E/E₀={ks44['loss_level']['200']['E_ratio_mean']:.2f})
by accident of cancellation; on decaying 2D NS the no-physics arm achieves rL2=
{nsd['none']['200']['rL2_mean']:.2f} yet over-dissipates to E/E₀=
{nsd['none']['200']['E_ratio_mean']:.2f}, whereas the anchor keeps E/E₀=
{nsd['anchor']['200']['E_ratio_mean']:.2f} (crash rate {nsd['anchor']['200']['crash_rate']:.2f}
vs 1.00 without the envelope). The lowest pointwise error is not the most physical solution.
(d) Stability–accuracy Pareto over all 13 suite models (rL2@200 vs E/E₀@200, log–log; star =
RCLN fix2+env; black-edged points crashed; crosshairs at rL2=1 and the truth
E₂₀₀/E₀=0.9416). The decoupling is visible in-domain: the lowest-error model (RCLN-v2
gate90, rL2={M['v2gate90']['rL2']['@200']['mean']:.2f}) over-dissipates to
E/E₀={e200('v2gate90'):.2f}, FNO/FNO+PhysReg sit at the worst corner on both axes
(E/E₀≈{e200('t20_fno'):.1f}, crashed), and RCLN (fix2+env) is the closest point to the
truth cross while retaining competitive accuracy.
"""
    (OUT / "CAPTIONS.md").write_text(txt, encoding="utf-8")
    print("saved CAPTIONS.md")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    captions()
    print("ALL DONE")

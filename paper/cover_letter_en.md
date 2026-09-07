# Cover Letter (draft, English)

Dear Editors,

We submit our manuscript entitled **"Inference-path physics stabilizes long-horizon neural PDE prediction"** for consideration at *Nature Communications*.

**The problem.** Long-horizon autoregressive rollout is the recognised weak point of neural PDE surrogates: errors compound over hundreds of steps and end in energy blow-up or unphysical over-dissipation. We argue that this fragility is not a matter of model capacity or training data, but of *where* physical knowledge is placed. As a soft penalty in the training loss, physics is a suggestion; as post-processing after the network, it is a patch; neither can stop a network from leaving the physically feasible set at inference time.

**The finding.** We show that moving exactly known physics—conservation projections, exact linear propagators, and dissipation envelopes—onto the inference path as zero-parameter, non-bypassable operators changes the stability *properties* of the learned dynamics, not just its metrics. Under a single controlled protocol in which every compared model is retrained from scratch (same data, budget, optimiser and curriculum), our architecture completes 200-step rollouts of 3D decaying turbulence with zero crashes and a 1.1% energy bias, while all retrained mainstream baselines fail; the same conclusion reproduces on 2D forced turbulence and the 1D chaotic Kuramoto–Sivashinsky equation, and the inference-path constraints transfer across systems and resolutions with zero recalibration. Three theorems provide deterministic guarantees that never invoke network internals, and the geometric discrepancy between anchor and residual emerges as a zero-cost error-diagnosis signal. We report the boundaries with equal prominence: a scenario where loss-level physics legitimately wins, and a cross-regime OOD failure.

**Why Nature Communications.** The position principle is not tied to any single architecture, system or discipline; it speaks simultaneously to machine learning, computational physics and numerical analysis, and it converts directly into three actionable reporting standards for the growing long-horizon-simulation community. All code, per-initial-condition data and figure sources are openly provided, including the verification scripts and deprecation records of our data archaeology.

**Suggested reviewer directions** (specific names at the editors' discretion): physics-informed machine learning; structure-preserving numerical analysis; turbulence and statistical physics.

We confirm that this manuscript is original, is not under consideration elsewhere, and that all authors have approved the submission.

Sincerely,
Yuze Chen

---

## Anticipated reviewer questions and where the manuscript pre-answers them

| Anticipated question | Where addressed |
|---|---|
| "Is this just an FNO with hard constraints?" | Results, "Why this is not an FNO with hard constraints" (role separation + necessity of each component) |
| "Does envelope intervention mask the true dynamics?" | Main benchmark: in-domain hit rate <1%, mild and two-sided corrections (mean 1.9%) |
| "Only 10 ICs—statistical significance?" | Protocol section + Limitations (iv); per-IC curves fully released; strict crash criteria |
| "Why these three systems?" | Introduction: they cover the three hard-codable physics families with distinct chaotic/turbulent character |
| "Comparison with classical numerical methods?" | Discussion (relation to structure-preserving numerics) + SI-B point-by-point table |
| "Is the discrepancy signal uncertainty quantification?" | Naming and transparency note (Results, discrepancy-indicator section): named indicator, not uncertainty estimator; failed cross-regime calibration reported |

"""
Generate figures for OOD generalization battlefield
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('paper/figures', exist_ok=True)

# Load results
with open('comparison_results/ood_generalization.json', 'r') as f:
    data = json.load(f)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Extract data
re_vals = [1000, 2000, 3000, 5000]
rcln_divs = [data['rcln'][str(re)]['div'] for re in re_vals]
pino_divs = [data['pino'][str(re)]['div'] for re in re_vals]
rcln_mse = [data['rcln'][str(re)]['mse'] for re in re_vals]
pino_mse = [data['pino'][str(re)]['mse'] for re in re_vals]
rcln_drift = [data['rcln'][str(re)]['energy_drift'] for re in re_vals]
pino_drift = [data['pino'][str(re)]['energy_drift'] for re in re_vals]

# Figure 1: Divergence vs Re
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(re_vals, rcln_divs, 'b-o', linewidth=2.5, markersize=10, label='RCLN-PsiNN')
ax.plot(re_vals, pino_divs, 'r-s', linewidth=2.5, markersize=10, label='PINO')
ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5, label='Training Re')
ax.set_xlabel('Reynolds Number', fontsize=13)
ax.set_ylabel('||div u|| (lower is better)', fontsize=13)
ax.set_title('Out-of-Distribution Generalization: Divergence Control', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Add advantage annotations
for i, (re, rcln, pino) in enumerate(zip(re_vals, rcln_divs, pino_divs)):
    ratio = pino / rcln
    ax.annotate(f'{ratio:.1f}×', xy=(re, rcln), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('paper/figures/ood_divergence.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/ood_divergence.png')
plt.close()

# Figure 2: Energy drift comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(re_vals))
width = 0.35

bars1 = ax.bar(x - width/2, rcln_drift, width, label='RCLN-PsiNN', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, pino_drift, width, label='PINO', color='red', alpha=0.7)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Reynolds Number', fontsize=13)
ax.set_ylabel('Energy Drift after 100 steps (%)', fontsize=13)
ax.set_title('Long-term Energy Conservation: OOD Test', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in re_vals])
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper/figures/ood_energy.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/ood_energy.png')
plt.close()

# Figure 3: Combined metrics heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RCLN heatmap
ax1 = axes[0]
metrics = ['MSE', 'Div', 'Energy\nDrift']
re_labels = [str(r) for r in re_vals]

# Normalize for visualization
rcln_data = np.array([
    [m/0.08 for m in rcln_mse],  # Normalized MSE
    [d/0.3 for d in rcln_divs],   # Normalized div
    [abs(d)/30 for d in rcln_drift]  # Normalized energy drift
])

im1 = ax1.imshow(rcln_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2)
ax1.set_xticks(range(len(re_vals)))
ax1.set_xticklabels(re_labels)
ax1.set_yticks(range(len(metrics)))
ax1.set_yticklabels(metrics)
ax1.set_title('RCLN-PsiNN: Normalized Metrics\n(green=good, red=bad)', fontsize=12)
ax1.set_xlabel('Reynolds Number')

# Add text annotations
for i in range(len(metrics)):
    for j in range(len(re_vals)):
        val = rcln_data[i, j]
        color = 'white' if val > 1 else 'black'
        ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

# PINO heatmap
ax2 = axes[1]
pino_data = np.array([
    [m/0.08 for m in pino_mse],
    [d/0.3 for d in pino_divs],
    [abs(d)/30 for d in pino_drift]
])

im2 = ax2.imshow(pino_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2)
ax2.set_xticks(range(len(re_vals)))
ax2.set_xticklabels(re_labels)
ax2.set_yticks(range(len(metrics)))
ax2.set_yticklabels(metrics)
ax2.set_title('PINO: Normalized Metrics\n(green=good, red=bad)', fontsize=12)
ax2.set_xlabel('Reynolds Number')

for i in range(len(metrics)):
    for j in range(len(re_vals)):
        val = pino_data[i, j]
        color = 'white' if val > 1 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

plt.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
plt.tight_layout()
plt.savefig('paper/figures/ood_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/ood_heatmap.png')
plt.close()

# Summary
print("\n" + "="*70)
print("OOD GENERALIZATION FIGURES GENERATED")
print("="*70)
print("\nKey Results:")
print(f"  RCLN divergence increase from Re=1000 to Re=5000: {rcln_divs[-1]/rcln_divs[0]:.1f}×")
print(f"  PINO divergence increase: {pino_divs[-1]/pino_divs[0]:.1f}×")
print(f"  RCLN maintains {rcln_divs[-1]/pino_divs[-1]:.1f}× better divergence control at Re=5000")
print(f"\n  RCLN energy drift: {np.mean(rcln_drift):.1f}% ± {np.std(rcln_drift):.1f}%")
print(f"  PINO energy drift: {np.mean(pino_drift):.1f}% ± {np.std(pino_drift):.1f}%")

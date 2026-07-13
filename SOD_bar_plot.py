import numpy as np
from matplotlib import pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "text.latex.preamble": r"\usepackage{helvet} \renewcommand{\familydefault}{\sfdefault}",
    "font.size": 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'legend.fontsize': 7,
})

k_indices = [0, 1, 3, 4, 5]
k_labels = [r'$k_1$', r'$k_{2a}$', r'$k_{2b}$', r'$k_3$', r'$k_4$']

k_control = [0.091, 1.2, 0.12, 0.053, 1.3]
k_control_err = [0.0015, 0.3, 0.01, 1.1e-4, 2.1e-2]
k_sod = [0.11, 1.8, 0.073, 0.037, 1.06]
k_sod_err = [0.0018, 0.4, 0.005, 1.1e-4, 1.4e-2]

fold_changes = [sod / ctrl if ctrl != 0 else np.nan for sod, ctrl in zip(k_sod, k_control)]

# Propagation of error for ratio R = A/B: dR = R * sqrt((dA/A)^2 + (dB/B)^2)
fold_change_err = []
for sod, sod_err, ctrl, ctrl_err, fc in zip(k_sod, k_sod_err, k_control, k_control_err, fold_changes):
    if ctrl != 0 and not np.isnan(fc):
        # Handle zero errors to avoid division by zero if any, though k values are non-zero here
        rel_sod = sod_err / sod if sod != 0 else 0
        rel_ctrl = ctrl_err / ctrl
        err = fc * np.sqrt(rel_sod**2 + rel_ctrl**2)
        fold_change_err.append(err)
    else:
        fold_change_err.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(k_labels, fold_changes, yerr=fold_change_err, capsize=5,
             color=['C0', 'C1', 'C2', 'C3', 'C4'], alpha=0.8, edgecolor='black')

ax.set_ylabel('Fold-change (SOD / Control)')
# ax.set_title('Fold-change in kinetic rates (SOD vs Control)', fontsize=14, fontweight='bold')

# Add text labels on top of bars
for bar, err in zip(bars, fold_change_err):
    height = bar.get_height()
    error = err if not np.isnan(err) else 0
    ax.text(bar.get_x() + bar.get_width() / 2., height + error + 0.02,
            f'{height:.2f}', ha='center', va='bottom')

ax.set_ylim(0, max(fold_changes) + max(fold_change_err) + 0.5)

# ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Save fold-change plot
# fc_plot_file = os.path.join(base_dir, 'k_fold_change_sod_control.png')
# plt.savefig(fc_plot_file, dpi=300, bbox_inches='tight')
# print(f"Fold-change plot saved to: {fc_plot_file}")
# plt.close(fig)

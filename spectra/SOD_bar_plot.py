import sys
import os

# Add the project root to sys.path to allow imports of utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import utils

utils.setup_plotting()

k_labels = [r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$']

k_control = [0.057, 0.18, 0.057, 2.00]
k_control_err = [0.001, 0.01, 0.0001, 0.07]
k_sod = [0.056, 0.10, 0.05, 1.88]
k_sod_err = [0.00054, 0.0068, 0.00008, 0.020]

fold_changes = np.array([sod / ctrl if ctrl != 0 else np.nan for sod, ctrl in zip(k_sod, k_control)])

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
fold_change_err = np.array(fold_change_err)

log_err = fold_change_err / (fold_changes * np.log(2))
log_fold_changes = np.log2(fold_changes)

fig, ax = plt.subplots(figsize=(90/25.4, 90/25.4), dpi=300)
bars = ax.bar(k_labels, log_fold_changes, yerr=log_err, capsize=5,
             color=['C0', 'C1', 'C2', 'C3', 'C4'], alpha=1, edgecolor='black')

ax.set_ylabel(r'Fold-change $\log_2$(SOD / Control)')
# ax.set_title('Fold-change in kinetic rates (SOD vs Control)', fontsize=14, fontweight='bold')

# Add text labels on top of bars
for bar, err in zip(bars, fold_change_err):
    height = bar.get_height()
    error = err if not np.isnan(err) else 0
    if height > 0:
        ax.text(bar.get_x() + bar.get_width() / 2., (height + error) * 1.1,
                f'{height:.2f}', ha='center', va='bottom')
    else:
        print(height)
        print(error)
        ax.text(bar.get_x() + bar.get_width() / 2., height - error - 0.1,
                f'{height:.2f}', ha='center', va='top')


ax.set_ylim([-1, 0.5])
ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
ax.spines['top'].set_position(('data', 0))
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
# sns.despine()

# ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Save fold-change plot
# fc_plot_file = os.path.join(base_dir, 'k_fold_change_sod_control.png')
# plt.savefig(fc_plot_file, dpi=300, bbox_inches='tight')
# print(f"Fold-change plot saved to: {fc_plot_file}")
# plt.close(fig)

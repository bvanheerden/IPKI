import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import os
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
from kinetic_models import kinetic_model
import h5py
from matplotlib import pyplot as plt
import utils

utils.setup_plotting()

base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'

# Datasets to process (map display names to folder names)
datasets = {
    'Ambient air': 'LHCII Magnet',
    'Oxygen scavengers': 'LHCII GCO Control',
}

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 0.5, 1.0, 0.5]

# Global defaults; these can be overridden per-dataset by a params.json or params.txt
default_startind = 1
default_onlen = 50
default_offlen = 600

onlyplot = False


def fittrace(data_dir, partlist=None, onlen=None, offlen=None, p0=None, startind=None):

    # load per-dataset params; allow partlist in params to override provided partnums
    params = kinetic_model.load_params(data_dir, 
                                       defaults={'startind': default_startind, 
                                                'onlen': default_onlen, 
                                                'offlen': default_offlen, 
                                                'partlist': default_partlist, 
                                                'p0': default_p0})
    startind_param = int(params.get('startind', default_startind))
    onlen_param = int(params.get('onlen', default_onlen))
    offlen_param = int(params.get('offlen', default_offlen))
    partlist_param = params.get('partlist', default_partlist)
    p0_param = params.get('p0', default_p0)
    
    if startind is None:
        startind = startind_param
    if onlen is None:
        onlen = onlen_param
    if offlen is None:
        offlen = offlen_param
    if p0 is None:
        p0 = p0_param
    if partlist is None:
        partlist = partlist_param
    # debug: show which partnums and p0 will be used for this dataset
    try:
        print(f"Dataset {data_dir} -> using partnums={partlist}, startind={startind}, onlen={onlen}, offlen={offlen}, p0={p0}")
    except Exception:
        pass

    if onlyplot:
        norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partlist, startind=slice(None, startind))
    else:
        norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partlist, startind=slice(None, startind))
        norm_pulsephotons = norm_pulsephotons[startind:]
        print(timestep)
    datapoints = len(norm_pulsephotons)
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.


    def fitfunc(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(t, k1, kr1, kr2, k3, k4, f, 1, q_fraction, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[3]+sol1.y[4], sol2.y[3][1:]+sol2.y[4][1:], sol3.y[3][1:]+sol3.y[4][1:],
                               sol4.y[3][1:]+sol4.y[4][1:], sol5.y[3][1:]+sol5.y[4][1:], sol6.y[3][1:]+sol6.y[4][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        # noinspection PyTupleAssignmentBalance
        # Parameters: k1, kr1, kr2, k3, k4, f, q_sum, q_fraction
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0, 0],
                                                                       [10, 10, 10, 10, 10, 1, 2, 1]), verbose=2)

    tau = [1 / popt[i] if popt[i] != 0 else np.nan for i in range(5)]
    q_sum = popt[6]
    q_fraction = popt[7]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        # propagate error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err = [perr[i] / popt[i] ** 2 if popt[i] != 0 else np.nan for i in range(5)]
        q_sum_err = perr[6]
        q_fraction_err = perr[7]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 5
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print(f'Tau1 = {tau[0]:.2g} ± {tau_err[0]:.2g} s')
    print(f'Tau_r1 = {tau[1]:.2g} ± {tau_err[1]:.2g} s')
    print(f'Tau_r2 = {tau[2]:.2g} ± {tau_err[2]:.2g} s')
    print(f'Tau3 = {tau[3]:.2g} ± {tau_err[3]:.2g} s')
    print(f'Tau4 = {tau[4]:.2g} ± {tau_err[4]:.2g} s')
    print(f'f = {popt[5]:.2g} ± {perr[5]:.2g}')
    print(f'Q_sum = {q_sum:.2g} ± {q_sum_err:.2g} cps')
    print(f'Q_fraction = {q_fraction:.2g} ± {q_fraction_err:.2g}')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, *popt)

    # Return normalized data, model, time base (exclude last because model uses concatenation offsets), taus and their errors
    return norm_pulsephotons, model, t_plot, tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err, popt, perr, pcov


def analyze_covariance(pcov, param_names=None):
    """Analyze the covariance matrix and print correlation information.

    Args:
        pcov: Covariance matrix from curve_fit
        param_names: List of parameter names (default: ['k1', 'k2', 'k3', 'k4', 'q0'])

    Returns:
        corr_matrix: Correlation coefficient matrix
    """
    if param_names is None:
        param_names = ['k1', 'kr1', 'kr2', 'k3', 'k4', 'f', 'q_sum', 'q_fraction']

    # Convert covariance to correlation matrix
    # corr[i,j] = cov[i,j] / (std[i] * std[j])
    std = np.sqrt(np.diag(pcov))
    corr_matrix = pcov / np.outer(std, std)

    # Print correlation information
    print("\nParameter Correlations (from covariance matrix):")
    print("-" * 60)

    # Find strongly correlated pairs (|corr| > 0.5)
    strong_corr_pairs = []
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > 0.5:
                strong_corr_pairs.append((param_names[i], param_names[j], corr_val))

    if strong_corr_pairs:
        print("Strong correlations (|r| > 0.5):")
        for p1, p2, corr in sorted(strong_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {p1:4s} <-> {p2:4s}  :  r = {corr:+.3f}")
    else:
        print("  No strong correlations found (threshold: |r| > 0.5)")

    print("\nFull correlation matrix:")
    print("-" * 60)
    corr_df = pd.DataFrame(corr_matrix, index=param_names, columns=param_names)
    print(corr_df.round(3))

    return corr_matrix


# Process all datasets
print("=" * 80)
print("Processing all datasets...")
print("=" * 80)

results = []
all_data = {}
covariance_data = {}

for display_name, folder_name in datasets.items():
    folder_path = os.path.join(base_dir, folder_name)

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found for {display_name}: {folder_path}")
        continue

    # Check if measurement files exist
    h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
    if not h5_files:
        print(f"Warning: No measurement files found in {folder_path}")
        continue

    print(f"\nProcessing: {display_name}")
    try:
        # call fittrace without passing the global default_partlist so that any per-dataset
        # 'partlist' in params.json / params.txt will be used automatically
        norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, popt, perr, pcov = fittrace(folder_path)

        # Store data for plotting
        all_data[display_name] = {
            'data': norm_pulsephotons,
            'model': model,
            'time': t_plot,
            'popt': popt,
            'perr': perr,
        }

        # Store covariance matrix for correlation analysis
        covariance_data[display_name] = pcov

        # Store results for table (with errors when available)
        def fmt_val_err(val, err):
            try:
                if np.isfinite(err):
                    return f"{val:.2g} ± {err:.2g}"
            except Exception:
                pass
            return f"{val:.2g}"

        results.append({
            'Dataset': display_name,
            'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
            'Tau_r1 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
            'Tau_r2 (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
            'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
            'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
            'f': fmt_val_err(popt[5], perr[5]),
            'Q_sum': fmt_val_err(q_sum, q_sum_err),
            'Q_fraction': fmt_val_err(q_fraction, q_fraction_err)
        })

        print(f"✓ Successfully processed {display_name}")
    except Exception as e:
        print(f"✗ Error processing {display_name}: {str(e)}")
        continue

print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Table for Notion
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY TABLE (Markdown/Notion)")
    print("=" * 60)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print("| " + " | ".join([str(val) for val in row]) + " |")
    print("=" * 60)

    # Save results to CSV
    output_file = os.path.join(base_dir, 'fitted_parameters.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

# Create separate plots
print("\n" + "=" * 80)
print("Creating plots...")
print("=" * 80)


# Plot 2: Control with GCO Control, MV, and SOD
print("\nCreating Plot 2: Control with Treatment Datasets...")
fig2, ax2 = plt.subplots(figsize=(90/25.4, 60/25.4))

control_treatment_datasets = ['Ambient air', 'Oxygen scavengers']
colors_ct = {'Ambient air': 'C0', 'Oxygen scavengers': 'C1'}

for display_name in control_treatment_datasets:
    if display_name in all_data:
        data = all_data[display_name]
        # if display_name == 'LHCII GCO Control':
        #     data['model'] = None
        color = colors_ct[display_name]
        # Plot experimental data
        ax2.plot(data['time'], data['data'], '.', color=color, alpha=0.2, markersize=1)
        # Plot model fit
        if data['model'] is not None:
            ax2.plot(data['time'], data['model'], '-', color=color, label=f'{display_name}',
                    alpha=1, markersize=1)

# Control
onlen = 2.5
offlen = 30

t_dark = onlen
t_light = t_dark + offlen
t_dark2 = t_light + onlen
t_light2 = t_dark2 + offlen
t_dark3 = t_light2 + onlen

phases = [
    (0, t_dark, 'white'),
    (t_dark, t_light, 'C0'),
    (t_light, t_dark2, 'white'),
    (t_dark2, t_light2, 'C0'),
    (t_light2, t_dark3, 'white'),
    (t_dark3, t_dark3+offlen, 'C0')
]

for start, end, color in phases:
    ax2.axvspan(start, end, ymin=0.96, ymax=1.0, facecolor=color,
               edgecolor='black', linewidth=0.5, transform=ax2.get_xaxis_transform())

# GCO
onlen = 5

t_dark = onlen
t_light = t_dark + offlen
t_dark2 = t_light + onlen
t_light2 = t_dark2 + offlen
t_dark3 = t_light2 + onlen

phases = [
    (0, t_dark, 'white'),
    (t_dark, t_light, 'C1'),
    (t_light, t_dark2, 'white'),
    (t_dark2, t_light2, 'C1'),
    (t_light2, t_dark3, 'white'),
    (t_dark3, t_dark3+offlen, 'C1')
]

for start, end, color in phases:
    ax2.axvspan(start, end, ymin=0.92, ymax=0.96, facecolor=color,
                edgecolor='black', linewidth=0.5, transform=ax2.get_xaxis_transform())

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Normalized fluorescence')
ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.93))
ax2.set_xlim(0, 90)
# sns.despine()
plt.tight_layout()

# Save plot 2
# plot_file2 = os.path.join(base_dir, 'control_treatment_comparison.png')
# plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
# print(f"Plot 2 saved to: {plot_file2}")
plt.show()

# Analyze covariance matrices for all datasets
# print("\n" + "=" * 80)
# print("Covariance Matrix Analysis - Parameter Correlations")
# print("=" * 80)
#
# param_names = ['k1', 'k2', 'k2_light', 'k3', 'k4', 'q_sum', 'q_fraction']
#
# all_corr_matrices = {}
#
# for display_name, pcov in covariance_data.items():
#     if pcov is not None and np.all(np.isfinite(pcov)):
#         print(f"\n{display_name}:")
#         corr_matrix = analyze_covariance(pcov, param_names)
#         all_corr_matrices[display_name] = corr_matrix
#     else:
#         print(f"\n{display_name}: No valid covariance data available")
#
# # Create heatmaps of correlation matrices
# print("\n" + "=" * 80)
# print("Creating correlation heatmaps...")
# print("=" * 80)
#
# # Create subplots for each dataset's correlation matrix
# n_datasets = len(all_corr_matrices)
# n_cols = min(3, n_datasets)  # max 3 columns
# n_rows = (n_datasets + n_cols - 1) // n_cols
#
# if n_datasets > 0:
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
#     if n_rows == 1 and n_cols == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1 or n_cols == 1:
#         axes = axes.reshape(n_rows, n_cols)
#
#     axes = axes.flatten()  # flatten for easier iteration
#
#     for idx, (display_name, corr_matrix) in enumerate(all_corr_matrices.items()):
#         ax = axes[idx]
#
#         # Create heatmap
#         sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
#                    vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
#                    xticklabels=param_names, yticklabels=param_names,
#                    linewidths=0.5, linecolor='gray')
#         ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
#
#     # Hide unused subplots
#     for idx in range(n_datasets, len(axes)):
#         axes[idx].set_visible(False)
#
#     plt.tight_layout()
#
#     # Save the figure
#     corr_plot_file = os.path.join(base_dir, 'correlation_matrices.png')
#     plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
#     print(f"Correlation heatmaps saved to: {corr_plot_file}")
#     plt.close(fig)
#
# print("\nCovariance analysis complete!")
#
#
#
#

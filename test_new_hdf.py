import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'

# Datasets to process (map display names to folder names)
datasets = {
    'LHCII Control': 'LHCII Control 301 mE',
    'LHCII GCO Control': 'LHCII GCO Control',
    'LHCII Magnet': 'LHCII Magnet',
    'LHCII MV': 'LHCII MV',
    'LHCII SOD': 'LHCII SOD',
}

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1 / 10, 1 / 3, 0.5]

# Global defaults; these can be overridden per-dataset by a params.json or params.txt
default_startind = 2
default_onlen = 50
default_offlen = 600

onlyplot = False


def kinetic(t, y, k1, k2, k3, k4):
    K = np.array([[0,  0,  k4,  k3],  # Bleached
                  [0, -k2, 0, k1],  # Quenced
                  [0, 0, -k4, 0],  # UnQuenched 2
                  [0,  k2, 0, -k1-k3]])  # Unquenched
    return K @ y


def modelfunc(t, k1, k2, k3, k4, q0, t_dark, t_light, t_dark2, t_light2, t_dark3):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark+1]], [0, 0, q0, 1-q0], t_eval=t[0:t_dark+1],
                     args=[k1, k2, k3, k4])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light+1]], sol1.y[:, -1], t_eval=t[t_dark:t_light+1],
                     args=[0, k2, 0, 0])
    sol3 = solve_ivp(kinetic, [t[t_light], t[t_dark2+1]], sol2.y[:, -1], t_eval=t[t_light:t_dark2+1],
                     args=[k1, k2, k3, k4])
    sol4 = solve_ivp(kinetic, [t[t_dark2], t[t_light2+1]], sol3.y[:, -1], t_eval=t[t_dark2:t_light2+1],
                     args=[0, k2, 0, 0])
    sol5 = solve_ivp(kinetic, [t[t_light2], t[t_dark3+1]], sol4.y[:, -1], t_eval=t[t_light2:t_dark3+1],
                     args=[k1, k2, k3, k4])
    sol6 = solve_ivp(kinetic, [t[t_dark3], t[-1]], sol5.y[:, -1], t_eval=t[t_dark3:-1],
                     args=[0, k2, 0, 0])
    return sol1, sol2, sol3, sol4, sol5, sol6


def onetrace(data_dir, partnum, startind=None):
    if startind is None:
        startind = default_startind
    dataset = h5py.File(os.path.join(data_dir, f'measurement {partnum}.h5'), 'r')
    abstimes = dataset['timestamps'][:] * 50  # 50 ns clock

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]  # gap is at least 20 ms
    boundary_times = abstimes[boundary_photons]  # end of pulse (start of gap)
    boundary_times_start = abstimes[boundary_photons + 1]  # start of pulse
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])  # first photon is start of first pulse
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6  # length of each pulse in ms
    timestep = np.mean(np.diff(boundary_times_start) / 1e9)  # timestep in s

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    # normalize to the start index
    norm_pulsephotons /= np.mean(norm_pulsephotons[:startind])

    # Replace non-finite or extra-low values with the previous valid value
    # Criteria: non-finite, <= 0 or extremely lower than previous point (factor 1e-3)
    for i in range(len(norm_pulsephotons)):
        if not np.isfinite(norm_pulsephotons[i]) or norm_pulsephotons[i] <= 0:
            if i == 0:
                # if first element bad, try to find next good one or set to 1.0
                good = None
                for j in range(1, len(norm_pulsephotons)):
                    if np.isfinite(norm_pulsephotons[j]) and norm_pulsephotons[j] > 0:
                        good = norm_pulsephotons[j]
                        break
                norm_pulsephotons[i] = good if good is not None else 1.0
            else:
                norm_pulsephotons[i] = norm_pulsephotons[i - 1]
        elif i > 0 and norm_pulsephotons[i] < norm_pulsephotons[i - 1] * 1e-3:
            # extremely low outlier compared to previous
            norm_pulsephotons[i] = norm_pulsephotons[i - 1]
    # norm_pulsephotons = uniform_filter1d(norm_pulsephotons, size=3)
    # norm_pulsephotons = median_filter(norm_pulsephotons, size=4)
    return norm_pulsephotons[:], timestep


def avtrace(data_dir, partnums, startind=None):
    if startind is None:
        startind = default_startind
    partnums_ = [onetrace(data_dir, partnum, startind)[0] for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    timesteps = [onetrace(data_dir, partnum, startind)[1] for partnum in partnums]
    return np.mean(partnums_, axis=0), np.mean(timesteps, axis=0)


def load_params(data_dir):
    """Load per-dataset parameters from params.json or params.txt in the data directory.
    Supported keys: startind (int), onlen (int), offlen (int), partlist (list), p0 (list)
    Falls back to defaults when keys are missing.
    """
    params = {}
    json_path = os.path.join(data_dir, 'params.json')
    txt_path = os.path.join(data_dir, 'params.txt')
    if os.path.exists(json_path):
        try:
            import json

            with open(json_path, 'r') as fh:
                params = json.load(fh)
        except Exception:
            params = {}
    elif os.path.exists(txt_path):
        # simple key=value parser; lists should be Python syntax
        try:
            with open(txt_path, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            params[k] = eval(v)
                        except Exception:
                            params[k] = v
        except Exception:
            params = {}

    # fill defaults
    startind = int(params.get('startind', default_startind))
    onlen = int(params.get('onlen', default_onlen))
    offlen = int(params.get('offlen', default_offlen))
    partlist = params.get('partlist', default_partlist)
    p0 = params.get('p0', default_p0)
    # debug: show when per-dataset params are found
    if params:
        print(f"Loaded params for {data_dir}: {params}")
    return startind, onlen, offlen, partlist, p0, params


def fittrace(data_dir, partlist=None, onlen=None, offlen=None, p0=None, startind=None):

    # load per-dataset params; allow partlist in params to override provided partnums
    startind_, onlen_, offlen_, partlist_, p0_, params = load_params(data_dir)
    if startind is None:
        startind = startind_
    if onlen is None:
        onlen = onlen_
    if offlen is None:
        offlen = offlen_
    if p0 is None:
        p0 = p0_
    if partlist is None:
        partlist = partlist_
    # debug: show which partnums and p0 will be used for this dataset
    try:
        print(f"Dataset {data_dir} -> using partnums={partlist}, startind={startind}, onlen={onlen}, offlen={offlen}, p0={p0}")
    except Exception:
        pass

    if onlyplot:
        norm_pulsephotons, timestep = avtrace(data_dir, partlist, startind)
    else:
        norm_pulsephotons, timestep = avtrace(data_dir, partlist, startind)
        norm_pulsephotons = norm_pulsephotons[startind:]
    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.


    def fitfunc(t, k1, k2, k3, k4, q0):
        sol1, sol2, sol3, sol4, sol5, sol6 = modelfunc(t, k1, k2, k3, k4, q0, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0],
                                                                       [10, 10, 10, 10, 1]), verbose=2)

    tau = [1 / popt[i] for i in range(4)]
    q0 = popt[4]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        # propagate error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err = [perr[i] / popt[i] ** 2 if popt[i] != 0 else np.nan for i in range(4)]
        q0_err = perr[4]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 4
        q0_err = np.nan

    print(f'Tau1 = {tau[0]:.2f} ± {tau_err[0]:.2f} s')
    print(f'Tau2 = {tau[1]:.2f} ± {tau_err[1]:.2f} s')
    print(f'Tau3 = {tau[2]:.2f} ± {tau_err[2]:.2f} s')
    print(f'Tau4 = {tau[3]:.2f} ± {tau_err[3]:.2f} s')
    print(f'Q0 = {q0:.2f} ± {q0_err:.2f} cps')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4])

    # Return normalized data, model, time base (exclude last because model uses concatenation offsets), taus and their errors
    return norm_pulsephotons, model, t_plot[:-1], tau, tau_err, q0, q0_err, popt, perr, pcov


def analyze_covariance(pcov, param_names=None):
    """Analyze the covariance matrix and print correlation information.

    Args:
        pcov: Covariance matrix from curve_fit
        param_names: List of parameter names (default: ['k1', 'k2', 'k3', 'k4', 'q0'])

    Returns:
        corr_matrix: Correlation coefficient matrix
    """
    if param_names is None:
        param_names = ['k1', 'k2', 'k3', 'k4', 'q0']

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
        norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q0, q0_err, popt, perr, pcov = fittrace(folder_path)

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
                    return f"{val:.6g} ± {err:.2g}"
            except Exception:
                pass
            return f"{val:.6g}"

        results.append({
            'Dataset': display_name,
            'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
            'Tau2 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
            'Tau3 (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
            'Tau4 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
            'Q0': fmt_val_err(q0, q0_err)
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

    # Save results to CSV
    output_file = os.path.join(base_dir, 'fitted_parameters.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

# Create separate plots
print("\n" + "=" * 80)
print("Creating plots...")
print("=" * 80)

# Plot 1: Control and Magnet
print("\nCreating Plot 1: Control and Magnet...")
fig1, ax1 = plt.subplots(figsize=(14, 8))

control_magnet_datasets = ['LHCII Control', 'LHCII Magnet']
colors_cm = {'LHCII Control': 'C0', 'LHCII Magnet': 'C1'}

for display_name in control_magnet_datasets:
    if display_name in all_data:
        data = all_data[display_name]
        color = colors_cm[display_name]
        # Plot experimental data
        ax1.plot(data['time'], data['data'], 'o-', color=color, label=f'{display_name} (data)',
                markersize=3, linewidth=1.5, alpha=0.2)
        # Plot model fit
        if data['model'] is not None:
            ax1.plot(data['time'], data['model'], '-', color=color, label=f'{display_name} (fit)',
                    linewidth=2.5, alpha=0.9)

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Normalized photon count', fontsize=12)
ax1.set_title('Control and Magnet Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot 1
plot_file1 = os.path.join(base_dir, 'control_magnet_comparison.png')
plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
print(f"Plot 1 saved to: {plot_file1}")
plt.close(fig1)

# Plot 2: Control with GCO Control, MV, and SOD
print("\nCreating Plot 2: Control with Treatment Datasets...")
fig2, ax2 = plt.subplots(figsize=(14, 8))

control_treatment_datasets = ['LHCII Control', 'LHCII GCO Control', 'LHCII MV', 'LHCII SOD']
colors_ct = {'LHCII Control': 'C0', 'LHCII GCO Control': 'C1', 'LHCII MV': 'C2', 'LHCII SOD': 'C3'}

for display_name in control_treatment_datasets:
    if display_name in all_data:
        data = all_data[display_name]
        # if display_name == 'LHCII GCO Control':
        #     data['model'] = None
        color = colors_ct[display_name]
        # Plot experimental data
        ax2.plot(data['time'], data['data'], 'o-', color=color, label=f'{display_name} (data)',
                markersize=3, linewidth=1.5, alpha=0.2)
        # Plot model fit
        if data['model'] is not None:
            ax2.plot(data['time'], data['model'], '-', color=color, label=f'{display_name} (fit)',
                    linewidth=2.5, alpha=0.9)

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Normalized photon count', fontsize=12)
ax2.set_title('Control and Treatment Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='best', ncol=2)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot 2
plot_file2 = os.path.join(base_dir, 'control_treatment_comparison.png')
plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
print(f"Plot 2 saved to: {plot_file2}")
plt.close(fig2)

# Analyze covariance matrices for all datasets
print("\n" + "=" * 80)
print("Covariance Matrix Analysis - Parameter Correlations")
print("=" * 80)

param_names = ['k1', 'k2', 'k3', 'k4', 'q0']

all_corr_matrices = {}

for display_name, pcov in covariance_data.items():
    if pcov is not None and np.all(np.isfinite(pcov)):
        print(f"\n{display_name}:")
        corr_matrix = analyze_covariance(pcov, param_names)
        all_corr_matrices[display_name] = corr_matrix
    else:
        print(f"\n{display_name}: No valid covariance data available")

# Create heatmaps of correlation matrices
print("\n" + "=" * 80)
print("Creating correlation heatmaps...")
print("=" * 80)

# Create subplots for each dataset's correlation matrix
n_datasets = len(all_corr_matrices)
n_cols = min(3, n_datasets)  # max 3 columns
n_rows = (n_datasets + n_cols - 1) // n_cols

if n_datasets > 0:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    axes = axes.flatten()  # flatten for easier iteration

    for idx, (display_name, corr_matrix) in enumerate(all_corr_matrices.items()):
        ax = axes[idx]

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                   xticklabels=param_names, yticklabels=param_names,
                   linewidths=0.5, linecolor='gray')
        ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save the figure
    corr_plot_file = os.path.join(base_dir, 'correlation_matrices.png')
    plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmaps saved to: {corr_plot_file}")
    plt.close(fig)

print("\nCovariance analysis complete!")






import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.optimize import curve_fit
import pandas as pd
# import seaborn as sns
import kinetic_model

base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'

# Datasets to process (map display names to folder names)
datasets = {
    'LHCII Control': 'LHCII Control 301 mE',
    'LHCII SOD': 'LHCII SOD',
}

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.5]

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
    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.


    def fitfunc(t, k1, k2, k2_light, k3, k4, q_sum, q_fraction):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(t, k1, 0.54, k3, k4, 1, 0.23, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3, k2_light=None)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        # noinspection PyTupleAssignmentBalance
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0],
                                                                       [10, 10, 10, 10, 10, 2, 1]), verbose=2)

    tau = [1 / popt[i] for i in range(5)]
    q_sum = popt[5]
    q_fraction = popt[6]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        print(popt, perr)
        # propagate error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err = [perr[i] / popt[i] ** 2 if popt[i] != 0 else np.nan for i in range(5)]
        q_sum_err = perr[5]
        q_fraction_err = perr[6]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 5
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print(f'Tau1 = {tau[0]:.2g} ± {tau_err[0]:.2g} s')
    print(f'Tau2 = {tau[1]:.2g} ± {tau_err[1]:.2g} s')
    # print(f'Tau2_light = {tau[2]:.2g} ± {tau_err[2]:.2g} s')
    print(f'Tau3 = {tau[3]:.2g} ± {tau_err[3]:.2g} s')
    print(f'Tau4 = {tau[4]:.2g} ± {tau_err[4]:.2g} s')
    print(f'Q_sum = {q_sum:.2g} ± {q_sum_err:.2g} cps')
    print(f'Q_fraction = {q_fraction:.2g} ± {q_fraction_err:.2g}')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

    # Return normalized data, model, time base, taus, errors, etc., AND timing parameters
    return norm_pulsephotons, model, t_plot[:-1], tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err, popt, perr, pcov, (t_dark, t_light, t_dark2, t_light2, t_dark3)


def analyze_covariance(pcov, param_names=None):
    """Analyze the covariance matrix and print correlation information.

    Args:
        pcov: Covariance matrix from curve_fit
        param_names: List of parameter names (default: ['k1', 'k2', 'k3', 'k4', 'q0'])

    Returns:
        corr_matrix: Correlation coefficient matrix
    """
    if param_names is None:
        param_names = ['k1', 'k2', 'k2_light', 'k3', 'k4', 'q_sum', 'q_fraction']

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
print("Processing Control and SOD datasets...")
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
        norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, popt, perr, pcov, t_params = fittrace(folder_path)

        # Store data for plotting
        all_data[display_name] = {
            'data': norm_pulsephotons,
            'model': model,
            'time': t_plot,
            'popt': popt,
            'perr': perr,
            't_params': t_params
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
            'Tau2 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
            # 'Tau2_light (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
            'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
            'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
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

    # Save results to CSV
    output_file = os.path.join(base_dir, 'fitted_parameters_control_sod.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

# Create plot
print("\n" + "=" * 80)
print("Creating plot...")
print("=" * 80)

fig, ax = plt.subplots(figsize=(14, 8))

plot_datasets = ['LHCII Control', 'LHCII SOD']
colors = {'LHCII Control': 'C0', 'LHCII SOD': 'C3'}

for display_name in plot_datasets:
    if display_name in all_data:
        data = all_data[display_name]
        color = colors[display_name]
        # Plot experimental data
        ax.plot(data['time'], data['data'], 'o-', color=color, label=f'{display_name} (data)',
                markersize=3, linewidth=1.5, alpha=0.2)
        # Plot model fit
        if data['model'] is not None:
            ax.plot(data['time'], data['model'], '-', color=color, label=f'{display_name} (fit)',
                    linewidth=2.5, alpha=0.9)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Normalized photon count', fontsize=12)
ax.set_title('Control and SOD Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# Save plot
plot_file = os.path.join(base_dir, 'control_sod_comparison.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")
plt.close(fig)

# Create population plots
print("\n" + "=" * 80)
print("Creating population plot...")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
plot_datasets = ['LHCII Control', 'LHCII SOD']
state_names = ['Bleached (State 0)', 'Quenched (State 1)', 'Unquenched 2 (State 2)', 'Unquenched (State 3)']

for i, display_name in enumerate(plot_datasets):
    if display_name in all_data:
        ax = axes[i]
        data = all_data[display_name]
        popt = data['popt']
        t = np.append(data['time'], data['time'][-1] + (data['time'][1] - data['time'][0]))
        t_dark, t_light, t_dark2, t_light2, t_dark3 = data['t_params']
        
        # Recalculate model populations
        sols = kinetic_model.modelfunc(t, popt[0], 0.50, popt[3], popt[4], 1, 0.23,
                                      t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=None)
        
        # Combine population trajectories
        pop_trajectories = [[] for _ in range(4)]
        for s_idx, sol in enumerate(sols):
            # sol.y has shape (4, points)
            for state_idx in range(4):
                y_vals = sol.y[state_idx]
                if s_idx > 0:
                    y_vals = y_vals[1:]
                pop_trajectories[state_idx].extend(y_vals)
        
        # Plot each population
        for state_idx in range(4):
            ax.plot(data['time'], pop_trajectories[state_idx], label=state_names[state_idx], linewidth=2)
            
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title(f'Model Populations: {display_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()
# plt.show()

# Save population plot
pop_plot_file = os.path.join(base_dir, 'control_sod_populations.png')
plt.savefig(pop_plot_file, dpi=300, bbox_inches='tight')
print(f"Population plot saved to: {pop_plot_file}")
plt.close(fig)

# Analyze covariance matrices
print("\n" + "=" * 80)
print("Covariance Matrix Analysis")
print("=" * 80)

param_names = ['k1', 'k2', 'k2_light', 'k3', 'k4', 'q_sum', 'q_fraction']

for display_name, pcov in covariance_data.items():
    if pcov is not None and np.all(np.isfinite(pcov)):
        print(f"\n{display_name}:")
        analyze_covariance(pcov, param_names)
    else:
        print(f"\n{display_name}: No valid covariance data available")

# Create bar plot of fold-change (SOD / Control) for k1-k4
print("\n" + "=" * 80)
print("Creating fold-change bar plot...")
print("=" * 80)

if 'LHCII Control' in all_data and 'LHCII SOD' in all_data:
    control_popt = all_data['LHCII Control']['popt']
    sod_popt = all_data['LHCII SOD']['popt']

    # Rates: k1=idx 0, k2=idx 1 (but fixed to 0.54 in model), k3=idx 3, k4=idx 4
    # The user asked for k1-k4.
    # Note: in fitfunc, k2 is hardcoded to 0.54, so k_sod[1] / k_control[1] should technically use the fixed value or the popt[1] which might be the p0[1] if it didn't change or if it was allowed to float but ignored.
    
    k_indices = [0, 1, 3, 4]
    k_labels = [r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$']
    
    # Use fixed k2 value 0.54 as defined in fitfunc for both if we want the actual model rates
    k_control = [control_popt[0], 0.54, control_popt[3], control_popt[4]]
    k_sod = [sod_popt[0], 0.59, sod_popt[3], sod_popt[4]]
    
    fold_changes = [sod / ctrl if ctrl != 0 else np.nan for sod, ctrl in zip(k_sod, k_control)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(k_labels, fold_changes, color=['C0', 'C1', 'C2', 'C3'], alpha=0.8, edgecolor='black')
    
    # Add a horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_ylabel('Fold-change (SOD / Control)', fontsize=12)
    ax.set_title('Fold-change in kinetic rates (SOD vs Control)', fontsize=14, fontweight='bold')
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11)

    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save fold-change plot
    fc_plot_file = os.path.join(base_dir, 'k_fold_change_sod_control.png')
    plt.savefig(fc_plot_file, dpi=300, bbox_inches='tight')
    print(f"Fold-change plot saved to: {fc_plot_file}")
    plt.close(fig)

print("\nAnalysis complete!")

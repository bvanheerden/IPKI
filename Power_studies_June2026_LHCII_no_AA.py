import re
import numpy as np
import h5py
import os
import json
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
import kinetic_model

# Enable LaTeX rendering globally
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "text.latex.preamble": r"\usepackage{helvet} \renewcommand{\familydefault}{\sfdefault}",
    "savefig.dpi": 300,
    "font.size": 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'legend.fontsize': 7,
})

base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'

# Default parameters (used if no config file is found)
default_params = {
    'partlist': [0, 1, 2],
    'p0': [1 / 20, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.5],
    'onlen': 50,
    'offlen': 200,
    'startind': 6,
    'low_value_threshold': 0.1
}

onlyplot = False


def load_config(folder_path):
    """Load configuration from config.json in the folder, return dict with parameters."""
    config_file = os.path.join(folder_path, 'config.json')
    params = default_params.copy()

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                params.update(loaded_config)
            print(f"  Loaded config from {config_file}")
        except Exception as e:
            print(f"  Warning: Could not load config file: {str(e)}")
            print(f"  Using default parameters")
    else:
        print(f"  No config file found, using default parameters")

    return params


def fittrace(data_dir, partnums, onlen, offlen, startind, p0, low_value_threshold, k2_fixed):

    if onlyplot:
        norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partnums, startind=startind, low_value_threshold=low_value_threshold)
    else:
        norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partnums, startind=startind, low_value_threshold=low_value_threshold)
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
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(t, k1, k2_fixed, k3, k4, q_sum, 0.23, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        popt, pcov, *extra = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0.95, 0],
                                       [1, 10, 10, 1, 10, 1.05, 1]), verbose=2, max_nfev=100)

    k_list = popt
    tau_list = [1 / k if k != 0 else np.nan for k in k_list[:5]]
    q_sum = popt[5]
    q_fraction = popt[6]

    # Calculate errors
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        # Error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err_list = [perr[i] / (k_list[i]**2) if k_list[i] != 0 else np.nan for i in range(5)]
        q_sum_err = perr[5]
        q_fraction_err = perr[6]
    else:
        perr = [np.nan] * 7
        tau_err_list = [np.nan] * 5
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print(f'K1 = {k_list[0]:.2g} ± {perr[0]:.2g} s⁻¹, Tau1 = {tau_list[0]:.2g} ± {tau_err_list[0]:.2g} s')
    print(f'K2 = {k_list[1]:.2g} ± {perr[1]:.2g} s⁻¹, Tau2 = {tau_list[1]:.2g} ± {tau_err_list[1]:.2g} s')
    # print(f'K2_light = {k_list[2]:.2g} ± {perr[2]:.2g} s⁻¹, Tau2_light = {tau_list[2]:.2g} ± {tau_err_list[2]:.2g} s')
    print(f'K3 = {k_list[3]:.2g} ± {perr[3]:.2g} s⁻¹, Tau3 = {tau_list[3]:.2g} ± {tau_err_list[3]:.2g} s')
    print(f'K4 = {k_list[4]:.2g} ± {perr[4]:.2g} s⁻¹, Tau4 = {tau_list[4]:.2g} ± {tau_err_list[4]:.2g} s')
    print(f'Q_sum = {q_sum:.2g} ± {q_sum_err:.2g} cps')
    print(f'Q_fraction = {q_fraction:.2g} ± {q_fraction_err:.2g}')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, *k_list)

    return (norm_pulsephotons, model, t_plot[:-1], tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, k_list, perr)


# Loop through all power folders and collect results
results = []

# Get all folders in the base directory
all_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]

# Sort folders for consistent ordering
all_folders = sorted(all_folders)

print("Processing power folders...")
print("=" * 80)

for folder_name in all_folders:
    folder_path = os.path.join(base_data_dir, folder_name)

    # Check if there are measurement files in this folder
    h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
    if not h5_files:
        print(f"Skipping {folder_name} - no measurement files found")
        continue

    # Load configuration for this folder
    print(f"\nProcessing: {folder_name}")
    params = load_config(folder_path)

    # Extract parameters from config
    partlist = params['partlist']
    p0 = params['p0']
    onlen = params['onlen']
    offlen = params['offlen']
    startind = params['startind']
    low_value_threshold = params['low_value_threshold']

    print(f"  Parameters: onlen={onlen}, offlen={offlen}, partlist={partlist}")

    # Extract power and AA info from folder name
    has_aa = folder_name.endswith('AA')
    
    # Extract only the numeric part for power
    power_match = re.search(r"(\d+(\.\d+)?)", folder_name)
    if power_match:
        power_val = float(power_match.group(1))
        power_str = f"{power_val:.2f}" # For display in title
    else:
        power_val = 0.0
        power_str = "0"
        print(f"  Warning: Could not extract numeric power from {folder_name}")

    print(f"  Power: {power_val}, AA: {has_aa}")

    if has_aa:
        k2_fixed = 2.75
    else:
        k2_fixed = 0.359  # LHCII

    try:
        # Fit the trace and store result tuple
        # result = (norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, k_list, perr)
        result = fittrace(folder_path, partlist, onlen, offlen, startind, p0, low_value_threshold, k2_fixed)
        
        norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, k_list, perr = result

        def fmt_val_err(val, err):
            try:
                if np.isfinite(err):
                    return f"{val:.2g} ± {err:.2g}"
            except Exception:
                pass
            return f"{val:.2g}"

        # Store results
        results.append({
            'Power (mmol photons m⁻² s⁻¹)': power_val,
            'AA': 'Yes' if has_aa else 'No',
            'K1 (s⁻¹)': fmt_val_err(k_list[0], perr[0]),
            'K2 (s⁻¹)': fmt_val_err(k_list[1], perr[1]),
            # 'K2_light (s⁻¹)': fmt_val_err(k_list[2], perr[2]),
            'K3 (s⁻¹)': fmt_val_err(k_list[3], perr[3]),
            'K4 (s⁻¹)': fmt_val_err(k_list[4], perr[4]),
            'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
            'Tau2 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
            # 'Tau2_light (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
            'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
            'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
            'Q_sum': fmt_val_err(q_sum, q_sum_err),
            'Q_fraction': fmt_val_err(q_fraction, q_fraction_err)
        })

        # Create and save individual trace plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t_plot, norm_pulsephotons, 'o-', color='gray', label='Experimental data',
                markersize=4, linewidth=1.5, alpha=0.7)
        if model is not None:
            ax.plot(t_plot, model, '-', color='red', label='Model fit', linewidth=2)
        ax.set_xlabel(r'Time (s)')
        ax.set_ylabel(r'Normalized photon count')
        aa_label = " (with AA)" if has_aa else ""
        ax.set_title(rf'Trace and Fit for {power_str}{aa_label}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot with descriptive filename
        aa_suffix = '_AA' if has_aa else ''
        plot_filename = f"trace_{power_str.replace(' ', '_')}{aa_suffix}.png"
        plot_filepath = os.path.join(folder_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"✓ Successfully processed {folder_name}")
        print(f"  Trace plot saved to: {plot_filename}")

    except Exception as e:
        print(f"✗ Error processing {folder_name}: {str(e)}")
        continue

print("\n" + "=" * 80)
print("\nSummary of Results:")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('Power (mmol photons m⁻² s⁻¹)').reset_index(drop=True)
    print(df.to_string(index=False))

    # Table for Notion
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY TABLE (Markdown/Notion)")
    print("=" * 60)

    # Helper to get numeric values and errors from the formatted strings
    def get_numeric(series):
        return series.str.split(' ±').str[0].astype(float)

    def get_error(series):
        # Extract the error part, handle cases where no error is present
        parts = series.str.split(' ±')
        return parts.apply(lambda x: float(x[1]) if len(x) > 1 else 0.0)

    # Calculate means for numeric columns
    mean_row = []
    for col in df.columns:
        if col in ['Power (mmol photons m⁻² s⁻¹)', 'AA']:
            if col == 'Power (mmol photons m⁻² s⁻¹)':
                mean_row.append('**AVERAGE**')
            else:
                mean_row.append('')
        else:
            try:
                mean_val = get_numeric(df[col]).mean()
                mean_row.append(f"**{mean_val:.2g}**")
            except Exception:
                mean_row.append('')

    q_sum_no_aa = df[df['AA'] == 'No']['Q_sum'] if not df[df['AA'] == 'No'].empty else pd.Series()
    # q_sum_with_aa = df[df['AA'] == 'Yes']['Q_sum'] if not df[df['AA'] == 'Yes'].empty else pd.Series()

    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print("| " + " | ".join([str(val) for val in row]) + " |")
    print("| " + " | ".join(mean_row) + " |")
    print("=" * 60)

    if not q_sum_no_aa.empty:
        print(f"Average Q_sum (no AA): {get_numeric(q_sum_no_aa).mean():.2g}")
    # if not q_sum_with_aa.empty:
    #     print(f"Average Q_sum (with AA): {get_numeric(q_sum_with_aa).mean():.2g}")

    # Save results to CSV
    output_file = os.path.join(base_data_dir, 'analysis_results_LHCII_noAA.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create a plot of k values vs power
    print("\n" + "=" * 80)
    print("Creating k values vs power plot...")
    print("=" * 80)

    print('Average Q_sum = ', get_numeric(df['Q_sum']).mean())

    # Separate data with and without AA
    # Use numeric dataframes for plotting but handle cases where AA is missing
    df_no_aa = df[df['AA'] == 'No'].copy()

    if not df_no_aa.empty:
        df_no_aa = df_no_aa.sort_values('Power (mmol photons m⁻² s⁻¹)')
    else:
        print("No data without AA found.")
        exit()

    # Create figure with 1 plot for k1, k2, k3, k4
    fig, ax1 = plt.subplots(figsize=(10, 6))

    powers = df_no_aa['Power (mmol photons m⁻² s⁻¹)']
    k1 = get_numeric(df_no_aa['K1 (s⁻¹)'])
    k1_err = get_error(df_no_aa['K1 (s⁻¹)'])
    k2 = get_numeric(df_no_aa['K2 (s⁻¹)'])
    k2_err = get_error(df_no_aa['K2 (s⁻¹)'])
    k3 = get_numeric(df_no_aa['K3 (s⁻¹)'])
    k3_err = get_error(df_no_aa['K3 (s⁻¹)'])
    k4 = get_numeric(df_no_aa['K4 (s⁻¹)'])
    k4_err = get_error(df_no_aa['K4 (s⁻¹)'])

    # Plot K1, K2, K3 on primary axis
    ln1 = ax1.errorbar(powers, k1, yerr=k1_err, fmt='o-', label='$k_1$',
                       linewidth=2, markersize=8, color='C0', capsize=5)
    ln3 = ax1.errorbar(powers, k3, yerr=k3_err, fmt='o-', label='$k_3$',
                       linewidth=2, markersize=8, color='C1', capsize=5)
    
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')
    ax1.set_ylabel(r'$k_1, k_3$ (s$^{-1}$)')
    
    # Plot K4 on secondary axis
    ax2 = ax1.twinx()
    ln4 = ax2.errorbar(powers, k4, yerr=k4_err, fmt='o-', label='$k_4$',
                       linewidth=2, markersize=8, color='C2', capsize=5)
    ax2.set_ylabel(r'$k_2, k_4$ (s$^{-1}$)')
    # ax2.tick_params(axis='y', labelcolor='C2')

    # Plot fixed k2 as a dashed line
    k2_fixed_val = 0.359  # LHCII fixed value from line 168
    ln2 = ax2.axhline(y=k2_fixed_val, color='C3', linestyle='--', label=rf'$k_2$')

    # Combined legend
    lns = [ln1, ln2, ln3, ln4]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=False)

    # ax1.set_title('Kinetic Rates vs Laser Power (LHCII no AA)', fontsize=14, fontweight='bold')
    # ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(base_data_dir, 'k_values_vs_power_LHCII_noAA.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()
else:
    print("No results to display")


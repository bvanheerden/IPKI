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
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "savefig.dpi": 300,
})

# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'

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
load_saved_data = True  # Set to True to skip fitting and load from CSV
share_q_fraction = True  # Set to True to share q_fraction globally, False for local per dataset
share_k3 = False  # Set to True to share k3 globally, False for local per dataset
fix_q_sum_at_power = None  # Set to a power value (e.g. 144) to fix q_sum for that power
fixed_q_sum_value = 1.0  # The value to fix q_sum to


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
    datapoints = len(norm_pulsephotons)
    endpoint = (datapoints - 1) * timestep
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
        # curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0.95, 0],
        #                               [10, 10, 10, 1, 10, 1.05, 1]), verbose=2, max_nfev=100)
        popt, pcov = None, None # Deprecated by global fit
        return None

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
output_file = os.path.join(base_data_dir, 'analysis_results.csv')
if load_saved_data and os.path.exists(output_file):
    print(f"Loading saved results from {output_file}...")
    df = pd.read_csv(output_file)
    results = df.to_dict('records')
else:
    results = []
    all_datasets = []
    output_file = os.path.join(base_data_dir, 'analysis_results.csv')
    
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
            power_str = power_match.group(1) # For display in title
        else:
            power_val = 0.0
            power_str = "0"
            print(f"  Warning: Could not extract numeric power from {folder_name}")
    
        print(f"  Power: {power_val}, AA: {has_aa}")
    
        try:
            # Load and prepare the trace
            if onlyplot:
                norm_pulsephotons, timestep = kinetic_model.avtrace(folder_path, partlist, startind=startind, low_value_threshold=low_value_threshold)
            else:
                norm_pulsephotons, timestep = kinetic_model.avtrace(folder_path, partlist, startind=startind, low_value_threshold=low_value_threshold)
                norm_pulsephotons = norm_pulsephotons[startind:]
            
            datapoints = len(norm_pulsephotons)
            endpoint = (datapoints - 1) * timestep
            t = np.linspace(0, endpoint, datapoints)
            
            t_dark = onlen
            t_light = t_dark + offlen
            t_dark2 = t_light + onlen
            t_light2 = t_dark2 + offlen
            t_dark3 = t_light2 + onlen

            dataset = {
                'power': power_val,
                'power_str': power_str,
                'has_aa': has_aa,
                'folder_name': folder_name,
                't': t,
                'norm_pulsephotons': norm_pulsephotons,
                'params': params,
                't_phases': (t_dark, t_light, t_dark2, t_light2, t_dark3)
            }
            all_datasets.append(dataset)

        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")

# Global Fitting
if not load_saved_data and all_datasets:
    print("\n" + "=" * 80)
    print("Performing global fit...")
    print("=" * 80)

# Group datasets for separate global fits
    datasets_aa = [ds for ds in all_datasets if ds['has_aa']]
    datasets_no_aa = [ds for ds in all_datasets if not ds['has_aa']]
    # datasets_no_aa = None

    groups = []
    if datasets_aa:
        groups.append(('AA', datasets_aa))
    if datasets_no_aa:
        groups.append(('Non-AA', datasets_no_aa))

    for group_name, datasets in groups:
        print(f"\nPerforming global fit for {group_name} group ({len(datasets)} datasets)...")
        
        # Number of datasets in this group
        n_ds = len(datasets)
        
        # We'll put shared k2 at index 0, followed by shared q_fraction (if enabled), 
        # followed by shared k3 (if enabled), followed by per-dataset parameters
        def global_fitfunc(t_dummy, *args):
            k2_shared = args[0]
            offset = 1
            q_fraction_shared = None
            k3_shared = None
            if share_q_fraction:
                q_fraction_shared = args[offset]
                offset += 1
            if share_k3:
                k3_shared = args[offset]
                offset += 1
            
            fit_results = []
            for i in range(n_ds):
                ds = datasets[i]
                
                # Extract per-dataset parameters
                k1, k2_light = args[offset:offset+2]
                current_offset = offset + 2
                
                if share_k3:
                    k3_local = k3_shared
                else:
                    k3_local = args[current_offset]
                    current_offset += 1
                
                k4, q_sum = args[current_offset:current_offset+2]
                current_offset += 2
                
                if share_q_fraction:
                    q_fraction_local = q_fraction_shared
                else:
                    q_fraction_local = args[current_offset]
                    current_offset += 1
                
                # Update offset for next dataset
                offset = current_offset
                
                t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
                
                sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
                    ds['t'], k1, k2_shared, k3_local, k4, q_sum, q_fraction_local,
                    t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light,
                )
                res = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                       sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))
                
                # Ensure the model length matches the experimental data length
                if len(res) < len(ds['t']):
                    res = np.pad(res, (0, len(ds['t']) - len(res)), mode='edge')
                elif len(res) > len(ds['t']):
                    res = res[:len(ds['t'])]
                    
                fit_results.append(res)
            return np.concatenate(fit_results)

        # Concatenate experimental data for this group
        y_data_combined = np.concatenate([ds['norm_pulsephotons'] for ds in datasets])
        t_dummy = np.zeros_like(y_data_combined)

        # Prepare initial guesses and bounds for shared parameters
        p0_global = [0.5] # Shared k2 guess
        lower_bounds = [0]
        upper_bounds = [10]
        
        if share_q_fraction:
            p0_global.append(0.2) # Shared q_fraction guess
            lower_bounds.append(0.0)
            upper_bounds.append(0.4)
        
        if share_k3:
            p0_global.append(0.1 / 6) # Shared k3 guess
            lower_bounds.append(0)
            upper_bounds.append(0.5)

        for ds in datasets:
            dp0 = ds['params']['p0']
            # Common per-dataset parameters: k1, k2_light
            p0_global.extend([dp0[0], dp0[2]])
            lower_bounds.extend([0, 0])
            
            # Use different bounds for k2_light based on group
            if group_name == 'AA':
                upper_bounds.extend([5, 2.6]) # AA group: k1 up to 5, k2_light up to 10
            else:
                upper_bounds.extend([5, 15])  # Non-AA group: k1 up to 5, k2_light up to 3
            
            if not share_k3:
                p0_global.append(dp0[3]) # k3
                lower_bounds.append(0)
                upper_bounds.append(0.5)
            
            # Common per-dataset parameters: k4, q_sum
            p0_global.append(dp0[4]) # k4
            lower_bounds.append(0)
            upper_bounds.append(15)

            if fix_q_sum_at_power is not None and np.isclose(ds['power'], fix_q_sum_at_power):
                p0_global.append(fixed_q_sum_value)
                lower_bounds.append(fixed_q_sum_value - 1e-6) # Small range to satisfy some optimizers
                upper_bounds.append(fixed_q_sum_value + 1e-6)
                print(f"  Fixing q_sum for power {ds['power']} to {fixed_q_sum_value}")
            else:
                p0_global.append(dp0[5]) # q_sum
                lower_bounds.append(0.98)
                upper_bounds.append(1.05)
            
            if not share_q_fraction:
                p0_global.append(0.2) # q_fraction
                lower_bounds.append(0.1)
                upper_bounds.append(0.3)

        popt, pcov = curve_fit(global_fitfunc, t_dummy, y_data_combined, p0=p0_global, 
                              bounds=(lower_bounds, upper_bounds), verbose=2, max_nfev=500)
        
        perr_global = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
        
        # Extract shared parameters
        k2_shared = popt[0]
        k2_shared_err = perr_global[0]
        ds_offset = 1
        
        q_fraction_shared = None
        q_fraction_shared_err = 0
        if share_q_fraction:
            q_fraction_shared = popt[ds_offset]
            q_fraction_shared_err = perr_global[ds_offset]
            ds_offset += 1
        
        k3_shared = None
        k3_shared_err = 0
        if share_k3:
            k3_shared = popt[ds_offset]
            k3_shared_err = perr_global[ds_offset]
            ds_offset += 1
        
        # Extract results and store them
        for i in range(n_ds):
            ds = datasets[i]
            
            # Extract per-dataset parameters from popt and perr_global
            k1, k2_light = popt[ds_offset:ds_offset+2]
            pk1, pk2_l = perr_global[ds_offset:ds_offset+2]
            current_ds_offset = ds_offset + 2
            
            if share_k3:
                k3_val = k3_shared
                pk3 = k3_shared_err
            else:
                k3_val = popt[current_ds_offset]
                pk3 = perr_global[current_ds_offset]
                current_ds_offset += 1
            
            k4, q_sum = popt[current_ds_offset:current_ds_offset+2]
            pk4, pq_s = perr_global[current_ds_offset:current_ds_offset+2]
            current_ds_offset += 2
            
            if share_q_fraction:
                q_fraction_val = q_fraction_shared
                q_fraction_err = q_fraction_shared_err
            else:
                q_fraction_val = popt[current_ds_offset]
                q_fraction_err = perr_global[current_ds_offset]
                current_ds_offset += 1
            
            # Update ds_offset for next dataset
            ds_offset = current_ds_offset
            
            k_list = [k1, k2_shared, k2_light, k3_val, k4, q_sum, q_fraction_val]
            perr = [pk1, k2_shared_err, pk2_l, pk3, pk4, pq_s, q_fraction_err]
            
            tau_list = [1 / k if k != 0 else np.nan for k in k_list[:5]]
            tau_err_list = [perr[j] / (k_list[j]**2) if k_list[j] != 0 else np.nan for j in range(5)]
            
            # Calculate individual model for plotting
            t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
            sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
                ds['t'], k1, k2_shared, k3_val, k4, q_sum, q_fraction_val,
                t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light
            )
            model = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                   sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))
            
            # Ensure the model length matches the experimental data length
            if len(model) < len(ds['t']):
                model = np.pad(model, (0, len(ds['t']) - len(model)), mode='edge')
            elif len(model) > len(ds['t']):
                model = model[:len(ds['t'])]

            def fmt_val_err(val, err):
                try:
                    if np.isfinite(err):
                        return f"{val:.2g} ± {err:.2g}"
                except Exception:
                    pass
                return f"{val:.2g}"

            results.append({
                'Power (mE)': ds['power'],
                'AA': 'Yes' if ds['has_aa'] else 'No',
                'K1 (s⁻¹)': fmt_val_err(k1, pk1),
                'K2 (s⁻¹)': fmt_val_err(k2_shared, k2_shared_err),
                'K2_light (s⁻¹)': fmt_val_err(k2_light, pk2_l),
                'K3 (s⁻¹)': fmt_val_err(k3_val, pk3),
                'K4 (s⁻¹)': fmt_val_err(k4, pk4),
                'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
                'Tau2 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
                'Tau2_light (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
                'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
                'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
                'Q_sum': fmt_val_err(q_sum, pq_s),
                'Q_fraction': fmt_val_err(q_fraction_val, q_fraction_err)
            })

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ds['t'], ds['norm_pulsephotons'], 'o-', color='gray', label='Experimental data',
                    markersize=4, linewidth=1.5, alpha=0.7)
            ax.plot(ds['t'], model, '-', color='red', label='Global model fit', linewidth=2)
            ax.set_xlabel(r'Time (s)', fontsize=12)
            ax.set_ylabel(r'Normalized photon count', fontsize=12)
            aa_label = " (with AA)" if ds['has_aa'] else ""
            ax.set_title(rf'Trace and Global Fit for {ds["power_str"]}{aa_label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_name = f'k2_global_fit_{ds["power_str"]}{"_AA" if ds["has_aa"] else ""}.png'
            plt.savefig(os.path.join(base_data_dir, plot_name))
            plt.close()
            
            # offset += 6
    
    print("\n" + "=" * 80)
    print("\nSummary of Results:")
    print("=" * 80)
    
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('Power (mE)').reset_index(drop=True)
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
        if col in ['Power (mE)', 'AA']:
            if col == 'Power (mE)':
                mean_row.append('**AVERAGE**')
            else:
                mean_row.append('')
        else:
            try:
                mean_val = get_numeric(df[col]).mean()
                mean_row.append(f"**{mean_val:.2g}**")
            except Exception:
                mean_row.append('')

    # Calculate means for per-dataset q_sum if needed
    q_sum_no_aa = df[df['AA'] == 'No']['Q_sum'] if not df[df['AA'] == 'No'].empty else pd.Series()
    q_sum_with_aa = df[df['AA'] == 'Yes']['Q_sum'] if not df[df['AA'] == 'Yes'].empty else pd.Series()

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
    if not q_sum_with_aa.empty:
        print(f"Average Q_sum (with AA): {get_numeric(q_sum_with_aa).mean():.2g}")

    # Save results to CSV
    output_file = os.path.join(base_data_dir, 'analysis_results.csv')
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
df_with_aa = df[df['AA'] == 'Yes'].copy()

if not df_no_aa.empty:
    df_no_aa = df_no_aa.sort_values('Power (mE)')
if not df_with_aa.empty:
    df_with_aa = df_with_aa.sort_values('Power (mE)')

# Create figure for "with AA" data: single axis for k1, k3, k4
if not df_with_aa.empty:
    fig_aa, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    x_aa = df_with_aa['Power (mE)'].values
    x_extrap = np.linspace(min(2, x_aa.min()), max(2, x_aa.max()), 100)

    # Plot K1 (with AA)
    k1_aa = get_numeric(df_with_aa['K1 (s⁻¹)'])
    ax1.errorbar(x_aa, k1_aa,
                yerr=get_error(df_with_aa['K1 (s⁻¹)']), fmt='o', label=r'$k_1$ (data)',
                linewidth=2, markersize=8, alpha=0.7, color='C0', capsize=5)
    # Linear fit for K1 with zero intercept
    m1 = np.sum(x_aa[:] * k1_aa[:]) / np.sum(x_aa[:]**2)
    k1_extrap = m1 * 2
    ax1.plot(x_extrap, m1 * x_extrap, 'C0--', label=r'$k_1$ (linear fit)')
    ax1.text(2, k1_extrap, rf' {k1_extrap:.2g}', color='C0', va='bottom')

    # Plot K3 (with AA)
    k3_aa = get_numeric(df_with_aa['K3 (s⁻¹)'])
    ax1.errorbar(x_aa, k3_aa,
                yerr=get_error(df_with_aa['K3 (s⁻¹)']), fmt='^', label=r'$k_3$ (data)',
                linewidth=2, markersize=8, alpha=0.7, color='C1', capsize=5)
    # Linear fit for K3 with zero intercept
    m3 = np.sum(x_aa[:-1] * k3_aa[:-1]) / np.sum(x_aa[:-1]**2)
    k3_extrap = m3 * 2
    ax1.plot(x_extrap, m3 * x_extrap, 'C1--', label=r'$k_3$ (linear fit)')
    ax1.text(2, k3_extrap, rf' {k3_extrap:.2g}', color='C1', va='top')

    # Plot K4 (with AA)
    k4_aa = get_numeric(df_with_aa['K4 (s⁻¹)'])
    ax1.errorbar(x_aa, 2 * k4_aa,
                yerr=get_error(df_with_aa['K4 (s⁻¹)']), fmt='v', label=r'$2k_4$ (data)',
                linewidth=2, markersize=8, alpha=0.7, color='C2', capsize=5)
    # Linear fit for K4 with zero intercept
    m4 = np.sum(x_aa[:] * k4_aa[:]) / np.sum(x_aa[:]**2)
    k4_extrap = m4 * 2
    ax1.plot(x_extrap, 2 * m4 * x_extrap, 'C2--', label=r'$2k_4$ (linear fit)')
    ax1.text(2, 2*k4_extrap, rf' {k4_extrap:.2g}', color='C2', va='bottom')

    k2_aa = get_numeric(df_with_aa['K2 (s⁻¹)'])[0]
    ln2 = ax1.axhline(y=k2_aa, color='gray', linestyle='--', label=rf'$k_2$')

    ax1.set_ylabel(r'$k$ (s$^{-1}$)', fontsize=12)
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1.5, None)
    ax1.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plot_file_aa = os.path.join(base_data_dir, 'k_values_vs_power_with_AA.png')
    plt.savefig(plot_file_aa, dpi=300, bbox_inches='tight')
    print(f"With AA plot saved to: {plot_file_aa}")

# Create unified figure for "without AA" data
if not df_no_aa.empty:
    fig_no_aa, ax_main = plt.subplots(figsize=(10, 6))

    powers = df_no_aa['Power (mE)']
    k1 = get_numeric(df_no_aa['K1 (s⁻¹)'])
    k1_err = get_error(df_no_aa['K1 (s⁻¹)'])
    k2l = get_numeric(df_no_aa['K2_light (s⁻¹)'])
    k2l_err = get_error(df_no_aa['K2_light (s⁻¹)'])
    k3 = get_numeric(df_no_aa['K3 (s⁻¹)'])
    k3_err = get_error(df_no_aa['K3 (s⁻¹)'])
    k4 = get_numeric(df_no_aa['K4 (s⁻¹)'])
    k4_err = get_error(df_no_aa['K4 (s⁻¹)'])

    # Plot k1, k2_light and k4 on primary axis
    ln1 = ax_main.errorbar(powers, k1, yerr=k1_err, fmt='o-', label=r'$k_1$',
                          linewidth=2, markersize=8, color='C0', capsize=5)
    ln2l = ax_main.errorbar(powers, k2l, yerr=k2l_err, fmt='s-', label=r'$k_{2,light}$',
                          linewidth=2, markersize=8, color='C3', capsize=5)
    ln4 = ax_main.errorbar(powers, k4, yerr=k4_err, fmt='v-', label=r'$k_4$',
                          linewidth=2, markersize=8, color='C2', capsize=5)

    # Plot fixed k2 as a dashed line
    k2_fixed_val = 0.54  # Fixed value for no AA (Thylakoid) from line 177
    ln2 = ax_main.axhline(y=k2_fixed_val, color='gray', linestyle='--', label=rf'$k_2$')

    ax_main.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)', fontsize=12)
    ax_main.set_ylabel(r'$k_1, k_2, k_4$ (s$^{-1}$)', fontsize=12)

    # Plot k3 on secondary axis
    ax_k3 = ax_main.twinx()
    ln3 = ax_k3.errorbar(powers, k3, yerr=k3_err, fmt='^-', label=r'$k_3$',
                         linewidth=2, markersize=8, color='C1', capsize=5)
    ax_k3.set_ylabel(r'$k_3$ (s$^{-1}$)', fontsize=12, color='C1')
    ax_k3.tick_params(axis='y', labelcolor='C1')

    # Combined legend
    lns = [ln1, ln2, ln2l, ln3, ln4]
    labs = [l.get_label() for l in lns]
    ax_main.legend(lns, labs, fontsize=10, loc='best', frameon=False)


    plt.tight_layout()
    plot_file_no_aa = os.path.join(base_data_dir, 'k_values_vs_power_no_AA.png')
    plt.savefig(plot_file_no_aa, dpi=300, bbox_inches='tight')
    print(f"No AA plot saved to: {plot_file_no_aa}")

    plt.show()
else:
    print("No results to display")


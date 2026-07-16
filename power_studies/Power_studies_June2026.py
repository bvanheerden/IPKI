import sys
import os

# Add the project root to sys.path to allow imports of utils and kinetic_models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import re
import numpy as np
import h5py
import json
import pickle
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import utils
from kinetic_models import kinetic_model

utils.setup_plotting()

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
use_individual_fits = True  # Set to True for individual trace fitting error estimation
load_saved_data = False  # Set to True to skip fitting and load from CSV
use_pickle = True  # Set to True to save/load processed traces
fix_q_sum_at_power = None  # Set to a power value (e.g. 144) to fix q_sum for that power
fixed_q_sum_value = 1.0  # The value to fix q_sum to




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
pickle_file = os.path.join(base_data_dir, 'processed_data.pkl')

if load_saved_data and os.path.exists(output_file):
    print(f"Loading saved results from {output_file}...")
    df = pd.read_csv(output_file)
    results = df.to_dict('records')
    all_datasets = [] # Not needed if only plotting results from CSV
elif use_pickle and os.path.exists(pickle_file):
    import pickle
    print(f"Loading processed data from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        all_datasets = pickle.load(f)
    results = []
else:
    results = []
    all_datasets = []
    
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
        params = utils.load_config(folder_path, default_params)
    
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
            norm_pulsephotons_full, timestep, all_traces_full = kinetic_model.avtrace(folder_path, partlist, startind=startind, low_value_threshold=low_value_threshold, return_all=True)
            
            if onlyplot:
                norm_pulsephotons = norm_pulsephotons_full
                all_traces = all_traces_full
            else:
                norm_pulsephotons = norm_pulsephotons_full[startind:]
                all_traces = [tr[startind:] for tr in all_traces_full]
            
            datapoints = len(norm_pulsephotons)
            endpoint = (datapoints - 1) * timestep
            t = np.linspace(0, endpoint, datapoints)

            # Full trace time for plotting
            datapoints_full = len(norm_pulsephotons_full)
            endpoint_full = (datapoints_full - 1) * timestep
            t_full = np.linspace(0, endpoint_full, datapoints_full)
            
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
                't_full': t_full,
                'norm_pulsephotons': norm_pulsephotons,
                'norm_pulsephotons_full': norm_pulsephotons_full,
                'all_traces': all_traces,
                'all_traces_full': all_traces_full,
                'params': params,
                't_phases': (t_dark, t_light, t_dark2, t_light2, t_dark3)
            }
            all_datasets.append(dataset)

        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")

    # Save to pickle if requested
    if use_pickle and all_datasets:
        print(f"Saving processed data to {pickle_file}...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_datasets, f)

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
        
        # Stage 1: Global fit to get k2_shared
        def global_fitfunc(t_dummy, *args):
            k2_shared = args[0]
            offset = 1
            
            fit_results = []
            for i in range(n_ds):
                ds = datasets[i]
                
                # Extract per-dataset parameters
                k1, k2_light, k3_local, k4, q_sum, q_fraction = args[offset:offset+6]
                offset += 6
                
                t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
                
                sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
                    ds['t'], k1, k2_shared, k3_local, k4, q_sum, q_fraction,
                    t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=0 if ds['has_aa'] else k2_light,
                )
                res = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                       sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))
                
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
        p0_global = [0.5] # Shared k2
        lower_bounds = [0]
        upper_bounds = [10]
        
        for ds in datasets:
            dp0 = [1 / 20, 1, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.2]
            # Per-dataset parameters: k1, k2_light, k3, k4, q_sum, q_fraction
            p0_global.extend([dp0[0], dp0[2], dp0[3], dp0[4]])
            lower_bounds.extend([0, 0, 0, 0])
            
            if group_name == 'AA':
                upper_bounds.extend([5, 2.6, 0.5, 10])
            else:
                upper_bounds.extend([5, 15, 0.5, 15])

            if fix_q_sum_at_power is not None and np.isclose(ds['power'], fix_q_sum_at_power):
                p0_global.append(fixed_q_sum_value)
                lower_bounds.append(fixed_q_sum_value - 1e-6)
                upper_bounds.append(fixed_q_sum_value + 1e-6)
                print(f"  Fixing q_sum for power {ds['power']} to {fixed_q_sum_value}")
            else:
                p0_global.append(dp0[5] if len(dp0) > 5 else 1.0) # q_sum
                lower_bounds.append(0.95)
                upper_bounds.append(1.05)
            
            p0_global.append(dp0[6] if len(dp0) > 6 else 0.2) # q_fraction
            lower_bounds.append(0.0)
            upper_bounds.append(0.4)

        print(p0_global)
        print(lower_bounds)
        print(upper_bounds)

        popt_global, _ = curve_fit(global_fitfunc, t_dummy, y_data_combined, p0=p0_global, 
                              bounds=(lower_bounds, upper_bounds), verbose=2, max_nfev=1000, ftol=1e-6, xtol=1e-6)
        
        k2_shared_val = popt_global[0]
        print(f"  Stage 1 Global Fit Results: k2={k2_shared_val:.4f}")

        # Stage 2: Local fits for each individual dataset with trace-level jackknife
        for ds in datasets:
            print(f"  Processing dataset Power={ds['power']} AA={ds['has_aa']}...")
            
            def local_fitfunc(t, k1, k2_light, k3, k4, q_sum, q_fraction):
                t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
                sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
                    t, k1, k2_shared_val, k3, k4, q_sum, q_fraction,
                    t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=0 if ds['has_aa'] else k2_light,
                )
                res = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                      sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:],
                                      sol6.y[2][1:]+sol6.y[3][1:]))
                if len(res) < len(t):
                    res = np.pad(res, (0, len(t) - len(res)), mode='edge')
                elif len(res) > len(t):
                    res = res[:len(t)]
                return res

            dp0 = [1 / 20, 1, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.2]
            # Per-dataset parameters: k1, k2_light, k3, k4, q_sum, q_fraction
            p0_local = [dp0[0], dp0[2], dp0[3], dp0[4], dp0[5] if len(dp0) > 5 else 1.0, dp0[6] if len(dp0) > 6 else 0.2]
            l_local = [0, 0, 0.002, 0.01, 0.98, 0.1]
            u_local = [5, 2.6 if group_name == 'AA' else 13, 0.5, 15, 1.05, 0.3]
            if fix_q_sum_at_power is not None and np.isclose(ds['power'], fix_q_sum_at_power):
                p0_local[4] = fixed_q_sum_value
                l_local[4] = fixed_q_sum_value - 1e-6
                u_local[4] = fixed_q_sum_value + 1e-6
            # p0_local[4] = 1
            # l_local[4] = 1 - 1e-6
            # u_local[4] = 1 + 1e-6

            # Main fit for this dataset
            popt_local, _ = curve_fit(local_fitfunc, ds['t'], ds['norm_pulsephotons'], p0=p0_local,
                                      bounds=(l_local, u_local), max_nfev=2000, verbose=1, xtol=1e-6, ftol=1e-6)
            
            # Individual trace fitting for local fit
            individual_fit_results = []
            n_traces = len(ds['all_traces'])
            if use_individual_fits and n_traces > 1:
                print(f"    Performing individual fits for {n_traces} traces...")
                for j in range(n_traces):
                    y_trace = ds['all_traces'][j]
                    y_trace_full = ds['all_traces_full'][j]
                    
                    try:
                        popt_j, _ = curve_fit(local_fitfunc, ds['t'], y_trace, p0=p0_local, bounds=(l_local, u_local),
                                              max_nfev=100)
                        individual_fit_results.append(popt_j)
                        print(popt_j)
                        
                        # Plotting the individual trace fit
                        model_j = local_fitfunc(ds['t'], *popt_j)
                        fig_j, ax_j = plt.subplots(figsize=(10, 5))
                        ax_j.plot(ds['t_full'], y_trace_full, 'o', color='gray', label=f'Trace {j+1} data', markersize=3, alpha=0.5)
                        ax_j.plot(ds['t'], model_j, '-', color='blue', label=f'Fit {j+1}')
                        ax_j.set_xlabel('Time (s)')
                        ax_j.set_ylabel('Normalized photon count')
                        aa_suffix = "_AA" if ds['has_aa'] else ""
                        ax_j.set_title(f"Trace Fit {j+1}: Power {ds['power_str']}{aa_suffix}")
                        ax_j.legend()
                        fit_plot_name = f"individual_fit_{ds['power_str']}{aa_suffix}_trace_{j+1}.png"
                        plt.savefig(os.path.join(base_data_dir, fit_plot_name))
                        plt.close(fig_j)
                        
                    except Exception as e:
                        print(f"      Trace {j+1} fit failed: {e}")

                individual_fit_results = np.array(individual_fit_results)
                n_succ = len(individual_fit_results)
                if n_succ > 1:
                    # Error estimated as standard error of the mean: std / sqrt(n)
                    perr_local = np.std(individual_fit_results, axis=0) #/ np.sqrt(n_succ)
                else:
                    perr_local = np.zeros_like(popt_local)
            else:
                perr_local = np.zeros_like(popt_local)

            # Store results
            k1, k2_light, k3_val, k4, q_sum, q_fraction_val = popt_local
            pk1, pk2_l, pk3, pk4, pq_s, pq_f = perr_local
            
            k_list = [k1, k2_shared_val, k2_light, k3_val, k4, q_sum, q_fraction_val]
            perr = [pk1, 0, pk2_l, pk3, pk4, pq_s, pq_f] # Errors for shared params are set to 0 here for simplicity
            
            tau_list = [1 / k if k != 0 else np.nan for k in k_list[:5]]
            tau_err_list = [perr[j] / (k_list[j]**2) if k_list[j] != 0 else np.nan for j in range(5)]
            
            # Calculate individual model for plotting
            t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
            sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
                ds['t'], k1, k2_shared_val, k3_val, k4, q_sum, q_fraction_val,
                t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light
            )
            model = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                   sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))
            if len(model) < len(ds['t']):
                model = np.pad(model, (0, len(ds['t']) - len(model)), mode='edge')
            elif len(model) > len(ds['t']):
                model = model[:len(ds['t'])]

            def fmt_val_err(val, err):
                if np.isfinite(err) and err > 0:
                    return f"{val:.2g} ± {err:.2g}"
                return f"{val:.2g}"

            results.append({
                'Power (mE)': ds['power'],
                'AA': 'Yes' if ds['has_aa'] else 'No',
                'K1 (s⁻¹)': fmt_val_err(k1, pk1),
                'K2 (s⁻¹)': fmt_val_err(k2_shared_val, 0),
                'K2_light (s⁻¹)': fmt_val_err(k2_light, pk2_l),
                'K3 (s⁻¹)': fmt_val_err(k3_val, pk3),
                'K4 (s⁻¹)': fmt_val_err(k4, pk4),
                'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
                'Tau2 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
                'Tau2_light (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
                'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
                'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
                'Q_sum': fmt_val_err(q_sum, pq_s),
                'Q_fraction': fmt_val_err(q_fraction_val, pq_f)
            })

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ds['t'], ds['norm_pulsephotons'], 'o-', color='gray', label='Experimental data', markersize=4, linewidth=1.5, alpha=0.7)
            ax.plot(ds['t'], model, '-', color='red', label='Local model fit', linewidth=2)
            ax.set_xlabel(r'Time (s)', fontsize=12)
            ax.set_ylabel(r'Normalized photon count', fontsize=12)
            aa_label = " (with AA)" if ds['has_aa'] else ""
            ax.set_title(rf'Trace and Local Fit for {ds["power_str"]}{aa_label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_name = f'k2_local_fit_{ds["power_str"]}{"_AA" if ds["has_aa"] else ""}.png'
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
        if series.dtype == 'O':
            return series.str.split(' ±').str[0].astype(float)
        return series.astype(float)

    def get_error(series):
        if series.dtype == 'O':
            # Extract the error part, handle cases where no error is present
            parts = series.str.split(' ±')
            return parts.apply(lambda x: float(x[1]) if len(x) > 1 else 0.0)
        return pd.Series(0.0, index=series.index)

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
    with open('data_no_aa_k2.pkl', 'wb') as f:
        pickle.dump(df_no_aa, f)

if not df_with_aa.empty:
    df_with_aa = df_with_aa.sort_values('Power (mE)')
    with open('data_with_aa_k2.pkl', 'wb') as f:
        pickle.dump(df_with_aa, f)

print("\nData saved to data_no_aa.pkl and data_with_aa.pkl")
print("Run 'python plot_power_studies.py' to generate plots.")


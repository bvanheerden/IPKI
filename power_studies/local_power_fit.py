import sys
import os

# Add the project root to sys.path to allow imports of utils and kinetic_models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import re
import numpy as np
import json
import pickle
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import seaborn as sns

import utils
from kinetic_models import kinetic_model

utils.setup_plotting()

base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'

# Default parameters (used if no config file is found)
default_params = {
    'partlist': [0, 1, 2],
    'p0': [1 / 20, 1.1, 1, 1 / 10, 1 / 3, 1.0, 0.2],
    'onlen': 50,
    'offlen': 200,
    'startind': 6,
    'low_value_threshold': 0.1
}

onlyplot = False
use_individual_fits = True  # Set to True for individual trace fitting error estimation
load_saved_data = False  # Set to True to skip fitting and load from CSV
use_pickle = True  # Set to True to save/load processed traces
fixed_q_sum_value = 1.0  # The value to fix q_sum to


# Loop through all power folders and collect results
output_file = os.path.join(base_data_dir, 'local_analysis_results.csv')
pickle_file = os.path.join(base_data_dir, 'processed_data.pkl')

if load_saved_data and os.path.exists(output_file):
    print(f"Loading saved results from {output_file}...")
    df = pd.read_csv(output_file)
    results = df.to_dict('records')
    all_datasets = []
else:
    results = []
    all_datasets = []
    if use_pickle and os.path.exists(pickle_file):
        print(f"Loading processed data from {pickle_file}...")
        try:
            with open(pickle_file, 'rb') as f:
                all_datasets = pickle.load(f)
        except Exception as e:
            print(f"  Warning: Could not load processed data: {e}")

    if not all_datasets:
        # Get all folders in the base directory
        all_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
        all_folders = sorted(all_folders)
        
        print("Processing power folders...")
        print("=" * 80)
        
        for folder_name in all_folders:
            folder_path = os.path.join(base_data_dir, folder_name)
            h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
            if not h5_files:
                continue
        
            params = utils.load_config(folder_path, default_params)
            partlist = params['partlist']
            onlen = params['onlen']
            offlen = params['offlen']
            startind = params['startind']
            low_value_threshold = params['low_value_threshold']
        
            has_aa = folder_name.endswith('AA')
            power_match = re.search(r"(\d+(\.\d+)?)", folder_name)
            if power_match:
                power_val = float(power_match.group(1))
                power_str = power_match.group(1)
            else:
                power_val = 0.0
                power_str = "0"
        
            try:
                norm_pulsephotons_full, timestep, all_traces_full = kinetic_model.avtrace(folder_path, partlist, startind=startind, low_value_threshold=low_value_threshold, return_all=True)
                
                norm_pulsephotons = norm_pulsephotons_full[startind:]
                all_traces = [tr[startind:] for tr in all_traces_full]
                
                datapoints = len(norm_pulsephotons)
                endpoint = (datapoints - 1) * timestep
                t = np.linspace(0, endpoint, datapoints)
    
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

    if use_pickle and all_datasets:
        print(f"Saving processed data to {pickle_file}...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_datasets, f)

# Purely Local Fitting
if not load_saved_data and all_datasets:
    print("\n" + "=" * 80)
    print("Performing purely local fits for each power level...")
    print("=" * 80)

    for ds in all_datasets:
        print(f"\nProcessing dataset: {ds['folder_name']} (Power={ds['power']}, AA={ds['has_aa']})")
        
        # Define local fit function including all parameters
        def local_fitfunc(t, k1, k2, k2_light, k3, k4, q_sum, q_fraction):
            t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
            sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
                t, k1, k2, k3, k4, 1, q_fraction,
                t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=0 if ds['has_aa'] else k2_light
            )
            res = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                  sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:],
                                  sol6.y[2][1:]+sol6.y[3][1:]))
            if len(res) < len(t):
                res = np.pad(res, (0, len(t) - len(res)), mode='edge')
            elif len(res) > len(t):
                res = res[:len(t)]
            return res

        p0_local = [1, 2.2 if ds['has_aa'] else 0.3, 1, 1 / 10, 1 / 3, 1.0, 0.2]
        # Initial guesses and bounds for all 7 parameters
        # p0: [k1, k2, k2_light, k3, k4, q_sum, q_fraction]
        l_local = [0, 0, 0, 0.002, 0.01, 0.95, 0.0]
        u_local = [10, 10, 15, 0.5, 15, 1.05, 0.5]
        
        try:
            # Main fit for this dataset
            res_fit = curve_fit(local_fitfunc, ds['t'], ds['norm_pulsephotons'], p0=p0_local,
                                      bounds=(l_local, u_local), maxfev=2000)
            popt_local = res_fit[0]
            
            # Individual trace fitting for error estimation
            individual_fit_results = []
            n_traces = len(ds['all_traces'])
            if use_individual_fits and n_traces > 1:
                print(f"    Performing individual fits for {n_traces} traces...")
                for j in range(n_traces):
                    try:
                        res_j = curve_fit(local_fitfunc, ds['t'], ds['all_traces'][j], p0=p0_local, bounds=(l_local, u_local), maxfev=500)
                        popt_j = res_j[0]
                        individual_fit_results.append(popt_j)
                    except Exception as e:
                        print(f"      Trace {j+1} fit failed: {e}")

                individual_fit_results = np.array(individual_fit_results)
                if len(individual_fit_results) > 1:
                    perr_local = np.std(individual_fit_results, axis=0)
                else:
                    perr_local = np.zeros_like(popt_local)
            else:
                perr_local = np.zeros_like(popt_local)

            # Store results
            k_list = popt_local
            perr = perr_local
            tau_list = [1 / k if k != 0 else np.nan for k in k_list[:5]]
            tau_err_list = [perr[j] / (k_list[j]**2) if k_list[j] != 0 else np.nan for j in range(5)]
            
            def fmt_val_err(val, err):
                if np.isfinite(err) and err > 0:
                    return f"{val:.2g} ± {err:.2g}"
                return f"{val:.2g}"

            results.append({
                'Power (mE)': ds['power'],
                'AA': 'Yes' if ds['has_aa'] else 'No',
                'K1 (s⁻¹)': fmt_val_err(k_list[0], perr[0]),
                'K2 (s⁻¹)': fmt_val_err(k_list[1], perr[1]),
                'K2_light (s⁻¹)': fmt_val_err(k_list[2], perr[2]),
                'K3 (s⁻¹)': fmt_val_err(k_list[3], perr[3]),
                'K4 (s⁻¹)': fmt_val_err(k_list[4], perr[4]),
                'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
                'Tau2 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
                'Tau2_light (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
                'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
                'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
                'Q_sum': fmt_val_err(k_list[5], perr[5]),
                'Q_fraction': fmt_val_err(k_list[6], perr[6])
            })

            # Plotting
            model = local_fitfunc(ds['t'], *popt_local)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ds['t'], ds['norm_pulsephotons'], 'o-', color='gray', label='Experimental data', markersize=3, alpha=0.5)
            ax.plot(ds['t'], model, '-', color='red', label='Local fit')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized photon count')
            aa_suffix = " (with AA)" if ds['has_aa'] else ""
            ax.set_title(f"Local Fit: Power {ds['power_str']}{aa_suffix}")
            ax.legend()
            plt.tight_layout()
            plot_name = f'pure_local_fit_{ds["power_str"]}{"_AA" if ds["has_aa"] else ""}.png'
            plt.savefig(os.path.join(base_data_dir, plot_name))
            plt.close()

        except Exception as e:
            print(f"  Fitting failed for {ds['folder_name']}: {e}")

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('Power (mE)').reset_index(drop=True)
    print("\nSummary of Results:")
    print(df.to_string(index=False))

    # Save results to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

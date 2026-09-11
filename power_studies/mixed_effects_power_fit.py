import sys
import os

# Add the project root to sys.path to allow imports of utils and kinetic_models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import re
import numpy as np
import pickle
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from matplotlib import pyplot as plt
import utils
from kinetic_models import kinetic_model

utils.setup_plotting()

# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'
# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/1 August 2026/Thylakoid power study'
base_data_dir = r'C:\\Users\\bertu\\Desktop\\1 August 2026\\Thylakoid power study'
# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/31 July 2026/Thylakoid power study'

# Mixed-effects regularization strengths (1/sigma)
# Higher weight = more "global" (less variation between powers)
K2_PENALTY_WEIGHT = 1.0
QF_PENALTY_WEIGHT = 1.0
F_PENALTY_WEIGHT = 1.0

# Default parameters (used if no config file is found)
# k1, k2, k2_l, k3, k4, q_sum, q_frac
default_params = {
    'onlen': 50,
    'offlen': 600,
    'startind': 0,
    'partlist': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'low_value_threshold': None,
    'p0': [1, 1, 0.2, 0.2, 2, 1.0, 0.23],
    'lower_bounds': None, # Can be set per power level in config.json
    'upper_bounds': None,
}

use_pickle = True  # Set to True to save/load processed traces
USE_2Q_MODEL = False  # Set to True to use the 5-state kinetic_2q model
FIT_K2_LIGHT = False  # Set to True to fit k2_light, False to fix it at 0
USE_JACKKNIFE = True  # Set to True to use leave-one-out jackknife for error estimation

# Loop through all power folders and collect results
pickle_file = os.path.join(base_data_dir, 'processed_data.pkl')

if use_pickle and os.path.exists(pickle_file):
    print(f"Loading processed data from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        all_datasets = pickle.load(f)
    if all_datasets and 'individual_traces' not in all_datasets[0]:
        print("Pickled data is missing individual traces. Reprocessing...")
        all_datasets = []
else:
    all_datasets = []
    all_folders = sorted([f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))])
    
    print("Processing power folders...")
    for folder_name in all_folders:
        folder_path = os.path.join(base_data_dir, folder_name)
        h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
        if not h5_files: continue

        params = utils.load_config(folder_path, default_params)
        partlist = params['partlist']
        startind = params['startind']
        low_value_threshold = params['low_value_threshold']
        
        has_aa = folder_name.endswith('AA')
        power_match = re.search(r"(\d+(\.\d+)?)", folder_name)
        power_val = float(power_match.group(1)) if power_match else 0.0
        power_str = power_match.group(1) if power_match else "0"

        try:
            # Use return_all=True to get individual traces for error estimation
            norm_pulsephotons_full, timestep, individual_traces_full = kinetic_model.avtrace(
                folder_path, partlist, startind=startind, low_value_threshold=low_value_threshold, return_all=True
            )
            
            # Trim all traces to startind + 3*(onlen+offlen)
            max_len = 3 * (params['onlen'] + params['offlen'])
            norm_pulsephotons = norm_pulsephotons_full[startind : startind + max_len]
            individual_traces = [tr[startind : startind + max_len] for tr in individual_traces_full]
            
            t = np.linspace(0, (len(norm_pulsephotons) - 1) * timestep, len(norm_pulsephotons))
            
            onlen = params['onlen']
            onlen_first = params.get('onlen_first', onlen)
            print(onlen, onlen_first)
            offlen = params['offlen']
            t_phases_len = (onlen_first, offlen, onlen, offlen, onlen)
            all_datasets.append({
                'power': power_val,
                'power_str': power_str,
                'has_aa': has_aa,
                'folder_name': folder_name,
                't': t,
                'norm_pulsephotons': norm_pulsephotons,
                'individual_traces': individual_traces,
                'params': params,
                't_phases': t_phases_len
            })
        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")

    if use_pickle and all_datasets:
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_datasets, f)

# Global Fitting
if all_datasets:
    print("\nPerforming mixed-effects global fit...")
    datasets_aa = [ds for ds in all_datasets if ds['has_aa']]
    datasets_no_aa = [ds for ds in all_datasets if not ds['has_aa']]

    groups = []
    if datasets_aa: groups.append(('AA', datasets_aa))
    if datasets_no_aa: groups.append(('Non-AA', datasets_no_aa))

    final_results = []

    for group_name, datasets_all in groups:
        # Lowest power levels to exclude from global fit
        low_power_levels = [2.0, 5.0, 11.0]
        # low_power_levels = []
        datasets = [ds for ds in datasets_all if ds['power'] not in low_power_levels]
        datasets_excluded = [ds for ds in datasets_all if ds['power'] in low_power_levels]
        
        print(f"\nProcessing {group_name} group ({len(datasets)} global, {len(datasets_excluded)} excluded)...")
        n_ds = len(datasets)
        
        if n_ds == 0:
            print(f"  Warning: No high-power datasets for global fit in {group_name}. Skipping group.")
            continue

        # Prepare experimental data and dummy t
        n_penalties_per_ds = 4 if USE_2Q_MODEL else 2
        y_data_combined = np.concatenate([ds['norm_pulsephotons'] for ds in datasets] + [np.zeros(n_penalties_per_ds * n_ds)])
        t_dummy = np.zeros_like(y_data_combined)

        # Define indices for k-values that will be fitted as log(k)
        if USE_2Q_MODEL:
            k_indices_global = [0, 1]  # kr1_mean, kr2_mean
            k_indices_ds = [0, 1, 2, 3, 4]  # k1, kr1, kr2, k3, k4
            n_global = 4
        else:
            k_indices_global = [0]  # k2_mean
            k_indices_ds = [0, 1, 2, 3, 4] if FIT_K2_LIGHT else [0, 1, 2, 3] # k1, k2, (k2_light), k3, k4
            n_global = 2

        # Define scale factors for more efficient optimization
        n_ds_params = 8 if USE_2Q_MODEL else (7 if FIT_K2_LIGHT else 6)
        if USE_2Q_MODEL:
            scales = [1.0, 1.0, 0.5, 0.2] # kr1_mean, kr2_mean, f_mean, qf_mean
            for _ in range(n_ds):
                scales.extend([0.05, 1.0, 1.0, 0.1, 0.5, 0.5, 1.0, 0.2]) # k1, kr1, kr2, k3, k4, f, q_sum, q_f
        else:
            scales = [1.0, 0.2] # k2_mean, qf_mean
            for _ in range(n_ds):
                if FIT_K2_LIGHT:
                    scales.extend([1.0, 1.0, 1.0, 0.1, 0.5, 1.0, 0.2]) # k1, k2, k2_light, k3, k4, q_sum, q_f
                else:
                    scales.extend([0.05, 1.0, 0.1, 0.5, 1.0, 0.2]) # k1, k2, k3, k4, q_sum, q_f
        
        scales = np.array(scales)

        class CachedGlobalFit:
            def __init__(self, datasets, scales):
                self.datasets = datasets
                self.scales = scales
                self.n_ds = len(datasets)
                self.cache = [None] * self.n_ds
                self.last_params = [None] * self.n_ds
                self.last_means = None

            def __call__(self, t_dummy, *args_scaled):
                args = np.array(args_scaled) * self.scales
                
                if USE_2Q_MODEL:
                    kr1_mean, kr2_mean, f_mean, qf_mean = args[0], args[1], args[2], args[3]
                    offset = 4
                else:
                    k2_mean, qf_mean = args[0], args[1]
                    offset = 2
                
                fit_results = []
                penalties = []
                epsilon = 1e-6
                
                for i in range(self.n_ds):
                    ds = self.datasets[i]
                    ds_params = args[offset:offset + n_ds_params]
                    offset += n_ds_params
                    
                    # Optimization: only re-calculate if local parameters have changed
                    if (self.cache[i] is None or 
                        self.last_params[i] is None or 
                        not np.array_equal(ds_params, self.last_params[i])):
                        
                        t_on1, t_off1, t_on2, t_off2, t_on3 = ds['t_phases']
                        t_dark = t_on1
                        t_light = t_dark + t_off1
                        t_dark2 = t_light + t_on2
                        t_light2 = t_dark2 + t_off2
                        t_dark3 = t_light2 + t_on3
                        
                        if USE_2Q_MODEL:
                            k1, kr1, kr2, k3, k4, f, q_sum, q_f = ds_params
                            s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc_2q(
                                ds['t'], k1, kr1, kr2, k3, k4, f, q_sum, q_f,
                                t_dark, t_light, t_dark2, t_light2, t_dark3
                            )
                            # Unquenched states are 3 and 4
                            res = np.concatenate((s1.y[3]+s1.y[4], s2.y[3][1:]+s2.y[4][1:], s3.y[3][1:]+s3.y[4][1:],
                                                   s4.y[3][1:]+s4.y[4][1:], s5.y[3][1:]+s5.y[4][1:], s6.y[3][1:]+s6.y[4][1:]))
                        else:
                            if FIT_K2_LIGHT:
                                k1, k2, k2_light, k3, k4, q_sum, q_f = ds_params
                            else:
                                k1, k2, k3, k4, q_sum, q_f = ds_params
                                k2_light = 0
                            
                            s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
                                ds['t'], k1, k2, k3, k4, q_sum, q_f,
                                t_dark, t_light, t_dark2, t_light2, t_dark3, 
                                k2_light=k2_light,
                            )
                            # Unquenched states are 2 and 3
                            res = np.concatenate((s1.y[2]+s1.y[3], s2.y[2][1:]+s2.y[3][1:], s3.y[2][1:]+s3.y[3][1:],
                                                   s4.y[2][1:]+s4.y[3][1:], s5.y[2][1:]+s5.y[3][1:], s6.y[2][1:]+s6.y[3][1:]))
                        
                        if len(res) < len(ds['t']):
                            res = np.pad(res, (0, len(ds['t']) - len(res)), mode='edge')
                        else:
                            res = res[:len(ds['t'])]
                        
                        self.cache[i] = res
                        self.last_params[i] = ds_params
                    
                    fit_results.append(self.cache[i])
                    # Penalty terms: W * (log(parameter) - log(mean))
                    if USE_2Q_MODEL:
                        penalties.append(K2_PENALTY_WEIGHT * (np.log(ds_params[1] + epsilon) - np.log(kr1_mean + epsilon)))
                        penalties.append(K2_PENALTY_WEIGHT * (np.log(ds_params[2] + epsilon) - np.log(kr2_mean + epsilon)))
                        penalties.append(F_PENALTY_WEIGHT * (np.log(ds_params[5] + epsilon) - np.log(f_mean + epsilon)))
                        penalties.append(QF_PENALTY_WEIGHT * (np.log(ds_params[7] + epsilon) - np.log(qf_mean + epsilon)))
                    else:
                        penalties.append(K2_PENALTY_WEIGHT * (np.log(ds_params[1] + epsilon) - np.log(k2_mean + epsilon)))
                        penalties.append(QF_PENALTY_WEIGHT * (np.log(ds_params[n_ds_params - 1] + epsilon) - np.log(qf_mean + epsilon)))
                
                # Compare in log-space: return log(model)
                # Add epsilon to avoid log(0)
                log_model = np.log(np.concatenate(fit_results) + epsilon)
                return np.concatenate([log_model, np.array(penalties)])

        global_fit_obj = CachedGlobalFit(datasets, scales)

        # Bounds and initial guesses
        if USE_2Q_MODEL:
            p0_global = [0.3, 2, 0.5, 0.18] # Global means: kr1_mean, kr2_mean, f_mean, qf_mean
            lower_bounds = [0, 1, 0, 0.0]
            upper_bounds = [1, 10, 1.0, 0.5]
        else:
            p0_global = [2 if group_name == 'AA' else 0.3, 0.23] # Global means: k2_mean, qf_mean
            lower_bounds = [0 if group_name == 'AA' else 0, 0]
            upper_bounds = [10 if group_name == 'AA' else 0.8, 0.3]
        
        for ds in datasets:
            dp0 = ds['params']['p0']
            custom_lower = ds['params'].get('lower_bounds')
            custom_upper = ds['params'].get('upper_bounds')

            if USE_2Q_MODEL:
                # Expecting/Adapting to k1, kr1, kr2, k3, k4, f, q_sum, q_f
                if len(dp0) >= 8:
                    p0_global.extend(dp0[:8])
                else:
                    # Adapt from 4-state p0: k1, k2, k2_light, k3, k4, q_sum, q_fraction
                    p0_global.extend([dp0[0], dp0[1], 0.03, dp0[3], dp0[4], 0.5, dp0[5], dp0[6]])
                
                if custom_lower and len(custom_lower) >= 8:
                    lower_bounds.extend(custom_lower[:8])
                else:
                    lower_bounds.extend([0, 0, 0, 0, 0, 0, 0.95, 0.0])
                
                if custom_upper and len(custom_upper) >= 8:
                    upper_bounds.extend(custom_upper[:8])
                else:
                    upper_bounds.extend([5, 10, 20, 0.5, 6, 1.0, 1.05, 0.5])
            else:
                # dp0 order assumed: k1, k2, k2_light, k3, k4, q_sum, q_fraction
                # Per-dataset: k1, k2, (k2_light), k3, k4, q_sum, q_f
                print(dp0)
                print(custom_lower)
                print(custom_upper)
                if FIT_K2_LIGHT:
                    p0_global.extend([dp0[0], dp0[1], dp0[2], dp0[3], dp0[4]])
                    if custom_lower and len(custom_lower) >= 5:
                        lower_bounds.extend(custom_lower[:5])
                    else:
                        lower_bounds.extend([0, 0, 0, 0, 0.08])
                    
                    if custom_upper and len(custom_upper) >= 5:
                        upper_bounds.extend(custom_upper[:5])
                    else:
                        upper_bounds.extend([5, 10, 15, 0.5, 3])
                else:
                    p0_global.extend([dp0[0], dp0[1], dp0[3], dp0[4]])
                    if custom_lower and len(custom_lower) >= 5:
                        # If 5 or more elements, we assume index 2 is k2_light and skip it
                        lower_bounds.extend([custom_lower[0], custom_lower[1], custom_lower[3], custom_lower[4]])
                    elif custom_lower and len(custom_lower) >= 4:
                        lower_bounds.extend(custom_lower[:4])
                    else:
                        lower_bounds.extend([0, 0, 0, 0])
                    
                    if custom_upper and len(custom_upper) >= 5:
                        upper_bounds.extend([custom_upper[0], custom_upper[1], custom_upper[3], custom_upper[4]])
                    elif custom_upper and len(custom_upper) >= 4:
                        upper_bounds.extend(custom_upper[:4])
                    else:
                        upper_bounds.extend([2, 3, 0.2, 5])
                
                # q_sum
                p0_global.append(dp0[5] if len(dp0) > 5 else 1.0)
                if custom_lower and len(custom_lower) > 5:
                    lower_bounds.append(custom_lower[5])
                else:
                    lower_bounds.append(0.95)
                
                if custom_upper and len(custom_upper) > 5:
                    upper_bounds.append(custom_upper[5])
                else:
                    upper_bounds.append(1.05)
                
                # q_fraction
                p0_global.append(dp0[6] if len(dp0) > 6 else 0.23)
                if custom_lower and len(custom_lower) > 6:
                    lower_bounds.append(custom_lower[6])
                else:
                    lower_bounds.append(0.1)
                
                if custom_upper and len(custom_upper) > 6:
                    upper_bounds.append(custom_upper[6])
                else:
                    upper_bounds.append(0.3)

        p0_global = np.array(p0_global)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        print(f"  Fitting {len(p0_global)} parameters (scaled)...")
        print(p0_global, lower_bounds, upper_bounds)
        p0_scaled = p0_global / scales
        lower_scaled = lower_bounds / scales
        upper_scaled = upper_bounds / scales
        print(p0_scaled, lower_scaled, upper_scaled)

        # Log-transform the experimental data for log-scale fitting
        epsilon = 1e-6
        y_data_log = np.log(np.concatenate([ds['norm_pulsephotons'] for ds in datasets]) + epsilon)
        y_data_combined_log = np.concatenate([y_data_log, np.zeros(n_penalties_per_ds * n_ds)])

        popt_scaled, pcov_scaled = curve_fit(global_fit_obj, t_dummy, y_data_combined_log, p0=p0_scaled, 
                                             bounds=(lower_scaled, upper_scaled), verbose=2, max_nfev=2500,
                                             ftol=1e-6, xtol=1e-6)
        
        popt = popt_scaled * scales
        # Rescale covariance matrix
        pcov = pcov_scaled * np.outer(scales, scales)
        perr = np.sqrt(np.diag(pcov))
        
        # Extract and store results
        if USE_2Q_MODEL:
            kr1_mean_val, kr2_mean_val, f_mean_val, qf_mean_val = popt[0:4]
            offset = 4
        else:
            k2_mean_val, qf_mean_val = popt[0:2]
            offset = 2

        global_ds_idx = 0
        for ds in datasets_all:
            is_excluded = ds['power'] in low_power_levels
            
            if not is_excluded:
                k_vals = popt[offset:offset + n_ds_params]
                k_errs_from_cov = perr[offset:offset + n_ds_params]
                # Increment offset only for global datasets
                offset += n_ds_params
                global_ds_idx += 1
            else:
                # Local fit for excluded dataset using global means
                print(f"    Performing local fit for excluded power {ds['power']} mE ({ds['folder_name']})...")
                
                # Reconstruct bounds and p0 for this dataset
                dp0 = ds['params']['p0']
                custom_lower = ds['params'].get('lower_bounds')
                custom_upper = ds['params'].get('upper_bounds')
                
                # Reconstruct full k_vals initial guess and bounds as in the global fit
                if USE_2Q_MODEL:
                    if len(dp0) >= 8: ds_p0_full = np.array(dp0[:8])
                    else: ds_p0_full = np.array([dp0[0], dp0[1], 0.03, dp0[3], dp0[4], 0.5, dp0[5], dp0[6]])
                    
                    if custom_lower and len(custom_lower) >= 8: ds_lower_full = np.array(custom_lower[:8])
                    else: ds_lower_full = np.array([0, 0, 0, 0, 0, 0, 0.95, 0.0])
                    
                    if custom_upper and len(custom_upper) >= 8: ds_upper_full = np.array(custom_upper[:8])
                    else: ds_upper_full = np.array([5, 10, 20, 0.5, 6, 1.0, 1.05, 0.5])
                    
                    opt_indices = [0, 3, 4, 6]
                    fixed_indices = [1, 2, 5, 7]
                    fixed_values = [kr1_mean_val, kr2_mean_val, f_mean_val, qf_mean_val]
                else:
                    if FIT_K2_LIGHT:
                        ds_p0_full = np.array([dp0[0], dp0[1], dp0[2] if group_name != 'AA' else 0.0, dp0[3], dp0[4], dp0[5] if len(dp0)>5 else 1.0, dp0[6] if len(dp0)>6 else 0.23])
                        ds_lower_full = np.array([0, 0, 0, 0, 0.08, 0.95, 0.1])
                        ds_upper_full = np.array([5, 10, 15 if group_name != 'AA' else 1e-9, 0.5, 3, 1.05, 0.3])
                        if custom_lower and len(custom_lower) >= 7: ds_lower_full = np.array(custom_lower[:7])
                        if custom_upper and len(custom_upper) >= 7: ds_upper_full = np.array(custom_upper[:7])
                        
                        opt_indices = [0, 2, 3, 4, 5]
                        fixed_indices = [1, 6]
                        fixed_values = [k2_mean_val, qf_mean_val]
                    else:
                        ds_p0_full = np.array([dp0[0], dp0[1], dp0[3], dp0[4], dp0[5] if len(dp0)>5 else 1.0, dp0[6] if len(dp0)>6 else 0.23])
                        ds_lower_full = np.array([0, 0, 0, 0, 0.95, 0.1])
                        ds_upper_full = np.array([2, 3, 0.2, 5, 1.05, 0.3])
                        if custom_lower:
                            if len(custom_lower) >= 7: ds_lower_full = np.array([custom_lower[0], custom_lower[1], custom_lower[3], custom_lower[4], custom_lower[5], custom_lower[6]])
                            elif len(custom_lower) >= 5: ds_lower_full = np.array([custom_lower[0], custom_lower[1], custom_lower[3], custom_lower[4], 0.95, 0.1])
                            elif len(custom_lower) >= 4: ds_lower_full[:4] = custom_lower[:4]
                        if custom_upper:
                            if len(custom_upper) >= 7: ds_upper_full = np.array([custom_upper[0], custom_upper[1], custom_upper[3], custom_upper[4], custom_upper[5], custom_upper[6]])
                            elif len(custom_upper) >= 5: ds_upper_full = np.array([custom_upper[0], custom_upper[1], custom_upper[3], custom_upper[4], 1.05, 0.3])
                            elif len(custom_upper) >= 4: ds_upper_full[:4] = custom_upper[:4]

                        opt_indices = [0, 2, 3, 4]
                        fixed_indices = [1, 5]
                        fixed_values = [k2_mean_val, qf_mean_val]
                
                ds_p0_opt = ds_p0_full[opt_indices]
                ds_lower_opt = ds_lower_full[opt_indices]
                ds_upper_opt = ds_upper_full[opt_indices]

                # Transition times
                t_on1, t_off1, t_on2, t_off2, t_on3 = ds['t_phases']
                t_dark, t_light, t_dark2, t_light2, t_dark3 = t_on1, t_on1+t_off1, t_on1+t_off1+t_on2, t_on1+t_off1+t_on2+t_off2, t_on1+t_off1+t_on2+t_off2+t_on3

                def local_fit_model(t_m, *params_opt):
                    p_full = np.zeros(len(ds_p0_full))
                    p_full[opt_indices] = params_opt
                    p_full[fixed_indices] = fixed_values
                    if USE_2Q_MODEL:
                        k1, kr1, kr2, k3, k4, f, q_sum, q_f = p_full
                        s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc_2q(t_m, k1, kr1, kr2, k3, k4, f, q_sum, q_f, t_dark, t_light, t_dark2, t_light2, t_dark3)
                        res = np.concatenate((s1.y[3]+s1.y[4], s2.y[3][1:]+s2.y[4][1:], s3.y[3][1:]+s3.y[4][1:], s4.y[3][1:]+s4.y[4][1:], s5.y[3][1:]+s5.y[4][1:], s6.y[3][1:]+s6.y[4][1:]))
                    else:
                        if FIT_K2_LIGHT: k1, k2, k2_light, k3, k4, q_sum, q_f = p_full
                        else: k1, k2, k3, k4, q_sum, q_f = p_full; k2_light = 0
                        s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(t_m, k1, k2, k3, k4, q_sum, q_f, t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light)
                        res = np.concatenate((s1.y[2]+s1.y[3], s2.y[2][1:]+s2.y[3][1:], s3.y[2][1:]+s3.y[3][1:], s4.y[2][1:]+s4.y[3][1:], s5.y[2][1:]+s5.y[3][1:], s6.y[2][1:]+s6.y[3][1:]))
                    if len(res) < len(t_m): res = np.pad(res, (0, len(t_m) - len(res)), mode='edge')
                    else: res = res[:len(t_m)]
                    return np.log(res + 1e-6)

                try:
                    popt_local_opt, pcov_local_opt = curve_fit(local_fit_model, ds['t'], np.log(ds['norm_pulsephotons'] + 1e-6), p0=ds_p0_opt, bounds=(ds_lower_opt, ds_upper_opt))
                    k_vals = np.zeros(len(ds_p0_full))
                    k_vals[opt_indices] = popt_local_opt
                    k_vals[fixed_indices] = fixed_values
                    k_errs_from_cov = np.zeros(len(ds_p0_full))
                    k_errs_from_cov[opt_indices] = np.sqrt(np.diag(pcov_local_opt))
                except Exception as e:
                    print(f"      Warning: Local fit failed for {ds['folder_name']}: {e}")
                    k_vals = ds_p0_full
                    k_errs_from_cov = np.zeros(len(ds_p0_full))

            # Bounds for jackknife/individual fits
            if not is_excluded:
                ds_lower = lower_bounds[offset-n_ds_params:offset]
                ds_upper = upper_bounds[offset-n_ds_params:offset]
            else:
                ds_lower = ds_lower_full
                ds_upper = ds_upper_full

            tau_errs_from_cov = [k_errs_from_cov[j] / (k_vals[j]**2) if k_vals[j] != 0 else np.nan for j in range(5)]
            
            k_errs = k_errs_from_cov
            tau_errs = tau_errs_from_cov
            
            # Fit individual traces or do jackknife to get error bars
            if 'individual_traces' in ds and len(ds['individual_traces']) > 1:
                if USE_JACKKNIFE:
                    print(f"    Performing jackknife ({len(ds['individual_traces'])} iterations) for {ds['folder_name']}...")
                else:
                    print(f"    Fitting {len(ds['individual_traces'])} individual traces for {ds['folder_name']}...")
                
                individual_popt_list = []
                
                t_on1, t_off1, t_on2, t_off2, t_on3 = ds['t_phases']
                t_dark = t_on1
                t_light = t_dark + t_off1
                t_dark2 = t_light + t_on2
                t_light2 = t_dark2 + t_off2
                t_dark3 = t_light2 + t_on3
                
                def single_trace_model(t_m, *params_fit):
                    if is_excluded and len(params_fit) == len(opt_indices):
                        p_full = np.zeros(len(ds_p0_full))
                        p_full[opt_indices] = params_fit
                        p_full[fixed_indices] = fixed_values
                    else:
                        p_full = params_fit

                    if USE_2Q_MODEL:
                        k1, kr1, kr2, k3, k4, f, q_sum, q_f = p_full
                        s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc_2q(
                            t_m, k1, kr1, kr2, k3, k4, f, q_sum, q_f,
                            t_dark, t_light, t_dark2, t_light2, t_dark3
                        )
                        res = np.concatenate((s1.y[3]+s1.y[4], s2.y[3][1:]+s2.y[4][1:], s3.y[3][1:]+s3.y[4][1:],
                                               s4.y[3][1:]+s4.y[4][1:], s5.y[3][1:]+s5.y[4][1:], s6.y[3][1:]+s6.y[4][1:]))
                    else:
                        if FIT_K2_LIGHT:
                            k1, k2, k2_light, k3, k4, q_sum, q_f = p_full
                        else:
                            k1, k2, k3, k4, q_sum, q_f = p_full
                            k2_light = 0
                        s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
                            t_m, k1, k2, k3, k4, q_sum, q_f,
                            t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light
                        )
                        res = np.concatenate((s1.y[2]+s1.y[3], s2.y[2][1:]+s2.y[3][1:], s3.y[2][1:]+s3.y[3][1:],
                                               s4.y[2][1:]+s4.y[3][1:], s5.y[2][1:]+s5.y[3][1:], s6.y[2][1:]+s6.y[3][1:]))
                    if len(res) < len(t_m): res = np.pad(res, (0, len(t_m) - len(res)), mode='edge')
                    else: res = res[:len(t_m)]
                    
                    return np.log(res + 1e-6)

                n_traces = len(ds['individual_traces'])
                jackknife_models = []
                data_to_plot = []
                for idx in range(n_traces):
                    try:
                        if USE_JACKKNIFE:
                            # Average all traces except the current one
                            subset = [ds['individual_traces'][j] for j in range(n_traces) if j != idx]
                            tr_to_fit = np.mean(subset, axis=0)
                        else:
                            tr_to_fit = ds['individual_traces'][idx]

                        # Use linear-space initial guess and bounds
                        tr_to_fit_log = np.log(tr_to_fit + 1e-6)
                        
                        if is_excluded:
                            p0_ind = k_vals[opt_indices]
                            bounds_ind = (ds_lower[opt_indices], ds_upper[opt_indices])
                        else:
                            p0_ind = k_vals
                            bounds_ind = (ds_lower, ds_upper)

                        popt_ind_opt, _ = curve_fit(single_trace_model, ds['t'], tr_to_fit_log, p0=p0_ind, 
                                                bounds=bounds_ind, max_nfev=1000)
                        
                        if is_excluded:
                            popt_ind = np.zeros(len(ds_p0_full))
                            popt_ind[opt_indices] = popt_ind_opt
                            popt_ind[fixed_indices] = fixed_values
                        else:
                            popt_ind = popt_ind_opt
                        
                        individual_popt_list.append(popt_ind)
                        
                        # Calculate model (in linear space for plotting) and store data for plotting
                        model_log = single_trace_model(ds['t'], *popt_ind)
                        jackknife_models.append(np.exp(model_log))
                        data_to_plot.append(tr_to_fit)
                    except Exception as e:
                        print(f"      Warning: Trace fit failed at index {idx}: {e}")
                
                # Plot jackknife fits
                if jackknife_models:
                    n_plots = len(jackknife_models)
                    ncols = 4
                    nrows = int(np.ceil(n_plots / ncols))
                    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
                    axes = axes.flatten()

                    for i in range(n_plots):
                        ax = axes[i]
                        ax.plot(ds['t'], data_to_plot[i], 'o', color='gray', markersize=2, alpha=0.3, label='Data')
                        ax.plot(ds['t'], jackknife_models[i], 'r-', linewidth=1.5, label='Fit')
                        title = f'Jackknife {i}' if USE_JACKKNIFE else f'Trace {i}'
                        ax.set_title(title)
                        if i == 0:
                            ax.legend()
                    
                    # Remove empty subplots
                    for j in range(i + 1, len(axes)):
                        fig.delaxes(axes[j])
                    
                    fig.suptitle(f"{'Jackknife' if USE_JACKKNIFE else 'Individual'} Fits: {ds['folder_name']}")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    save_name = f"{'jackknife' if USE_JACKKNIFE else 'individual'}_fits_{ds['folder_name']}.png"
                    plt.savefig(os.path.join(base_data_dir, save_name))
                    plt.close()
                
                if len(individual_popt_list) > 1:
                    if USE_JACKKNIFE:
                        # Jackknife error: sqrt((n-1)/n * sum((theta_i - theta_mean)^2))
                        # theta_mean is the mean of the jackknife estimates
                        n_jk = len(individual_popt_list)
                        k_errs = np.std(individual_popt_list, axis=0) * np.sqrt(n_jk - 1)
                        k_errs *= 1.96 # 95% CI
                        
                        individual_taus = [[1/k if k != 0 else np.nan for k in p_ind[:5]] for p_ind in individual_popt_list]
                        tau_errs = np.nanstd(individual_taus, axis=0) * np.sqrt(n_jk - 1)
                    else:
                        # Standard deviation of individual fits
                        # k_errs = np.std(individual_popt_list, axis=0)
                        k_errs = median_abs_deviation(individual_popt_list, axis=0, scale='normal')
                        individual_taus = [[1/k if k != 0 else np.nan for k in p_ind[:5]] for p_ind in individual_popt_list]
                        tau_errs = np.nanstd(individual_taus, axis=0)
                        tau_errs /= np.sqrt(len(individual_taus)) # Standard error

            if USE_2Q_MODEL:
                k1, kr1, kr2, k3, k4, f_val, q_sum, q_f = k_vals
                pk1, pkr1, pkr2, pk3, pk4, pf, pqs, pqf = k_errs
                tau_list = [1/k if k != 0 else np.nan for k in [k1, kr1, kr2, k3, k4]]
                tau_errs = [k_errs[j] / (k_vals[j]**2) if k_vals[j] != 0 else np.nan for j in range(5)]
            else:
                if FIT_K2_LIGHT:
                    k1, k2, k2_light, k3, k4, q_sum, q_f = k_vals
                    pk1, pk2, pk2l, pk3, pk4, pqs, pqf = k_errs
                    tau_list = [1/k if k != 0 else np.nan for k in [k1, k2, k2_light, k3, k4]]
                    tau_errs = [k_errs[j] / (k_vals[j]**2) if k_vals[j] != 0 else np.nan for j in range(5)]
                else:
                    k1, k2, k3, k4, q_sum, q_f = k_vals
                    pk1, pk2, pk3, pk4, pqs, pqf = k_errs
                    k2_light = 0
                    tau_list = [1/k if k != 0 else np.nan for k in [k1, k2, 0, k3, k4]]
                    # k_errs indices for k1, k2, k3, k4 are 0, 1, 2, 3
                    tau_errs = [k_errs[0]/k1**2, k_errs[1]/k2**2, np.nan, k_errs[2]/k3**2, k_errs[3]/k4**2]
            
            def fmt(v, e): return f"{v:.3g} ± {e:.3g}" if np.isfinite(e) else f"{v:.3g}"
            
            res_dict = {
                'Power (mE)': ds['power'],
                'AA': 'Yes' if ds['has_aa'] else 'No',
                'K1 (s⁻¹)': fmt(k1, pk1),
                'K3 (s⁻¹)': fmt(k3, pk3),
                'K4 (s⁻¹)': fmt(k4, pk4),
                'Tau1 (s)': fmt(tau_list[0], tau_errs[0]),
                'Tau3 (s)': fmt(tau_list[3], tau_errs[3]),
                'Tau4 (s)': fmt(tau_list[4], tau_errs[4]),
                'Q_sum': fmt(q_sum, pqs),
                'Q_fraction': fmt(q_f, pqf)
            }
            if USE_2Q_MODEL:
                res_dict.update({
                    'Kr1 (s⁻¹)': fmt(kr1, pkr1),
                    'Kr2 (s⁻¹)': fmt(kr2, pkr2),
                    'f': fmt(f_val, pf),
                    'Tau_r1 (s)': fmt(tau_list[1], tau_errs[1]),
                    'Tau_r2 (s)': fmt(tau_list[2], tau_errs[2]),
                })
            else:
                res_dict.update({
                    'K2 (s⁻¹)': fmt(k2, pk2),
                    'Tau2 (s)': fmt(tau_list[1], tau_errs[1]),
                })
                if FIT_K2_LIGHT:
                    res_dict.update({
                        'K2_light (s⁻¹)': fmt(k2_light, pk2l),
                        'Tau2_light (s)': fmt(tau_list[2], tau_errs[2]),
                    })
            final_results.append(res_dict)
            
            # Recalculate model for plot
            t_on1, t_off1, t_on2, t_off2, t_on3 = ds['t_phases']
            t_dark = t_on1
            t_light = t_dark + t_off1
            t_dark2 = t_light + t_on2
            t_light2 = t_dark2 + t_off2
            t_dark3 = t_light2 + t_on3
            if USE_2Q_MODEL:
                k1, kr1, kr2, k3, k4, f_val, q_sum, q_f = k_vals
                s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc_2q(
                    ds['t'], k1, kr1, kr2, k3, k4, f_val, q_sum, q_f,
                    t_dark, t_light, t_dark2, t_light2, t_dark3
                )
                model = np.concatenate((s1.y[3]+s1.y[4], s2.y[3][1:]+s2.y[4][1:], s3.y[3][1:]+s3.y[4][1:],
                                       s4.y[3][1:]+s4.y[4][1:], s5.y[3][1:]+s5.y[4][1:], s6.y[3][1:]+s6.y[4][1:]))
            else:
                if FIT_K2_LIGHT:
                    k1, k2, k2_light, k3, k4, q_sum, q_f = k_vals
                else:
                    k1, k2, k3, k4, q_sum, q_f = k_vals
                    k2_light = 0
                s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
                    ds['t'], k1, k2, k3, k4, q_sum, q_f,
                    t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light,
                )
                model = np.concatenate((s1.y[2]+s1.y[3], s2.y[2][1:]+s2.y[3][1:], s3.y[2][1:]+s3.y[3][1:],
                                       s4.y[2][1:]+s4.y[3][1:], s5.y[2][1:]+s5.y[3][1:], s6.y[2][1:]+s6.y[3][1:]))
            if len(model) < len(ds['t']): model = np.pad(model, (0, len(ds['t']) - len(model)), mode='edge')
            else: model = model[:len(ds['t'])]

            plt.figure(figsize=(6, 3))
            plt.plot(ds['t'], ds['norm_pulsephotons'], 'o', color='gray', markersize=3, alpha=0.5, label='Data')
            plt.plot(ds['t'], model, 'r-', linewidth=2, label='Mixed-Effects Global Fit')
            plt.title(f"Mixed-Effects Fit: {ds['folder_name']}")
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized Photon Counts')
            plt.legend()
            plt.savefig(os.path.join(base_data_dir, f"mixed_fit_{ds['folder_name']}.pdf"))
            plt.close()

    # Final summary and export
    df = pd.DataFrame(final_results).sort_values(['AA', 'Power (mE)'])
    print("\nSummary of Mixed-Effects Results:")
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(base_data_dir, 'mixed_effects_results_k2l.csv'), index=False)
    
    df_no_aa = df[df['AA'] == 'No'].sort_values('Power (mE)')
    df_with_aa = df[df['AA'] == 'Yes'].sort_values('Power (mE)')

    results_dir = os.path.join(project_root, 'results')
    path = os.path.join(results_dir, r'data_no_aa_1q_k2l.pkl')
    with open(path, 'wb') as f:
        pickle.dump(df_no_aa, f)
    path = os.path.join(results_dir, r'data_with_aa_1q_k2l.pkl')
    with open(path, 'wb') as f:
        pickle.dump(df_with_aa, f)
    
    print(f"\nResults saved to mixed_effects_results.csv, data_no_aa_k2.pkl and data_with_aa_k2.pkl")

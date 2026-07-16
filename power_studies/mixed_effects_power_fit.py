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

# Mixed-effects regularization strengths (1/sigma)
# Higher weight = more "global" (less variation between powers)
K2_PENALTY_WEIGHT = 1.0
QF_PENALTY_WEIGHT = 1.0
F_PENALTY_WEIGHT = 1.0

# Default parameters (used if no config file is found)
default_params = {
    'partlist': [0, 1, 2],
    'p0': [1 / 20, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.23],
    'onlen': 50,
    'offlen': 200,
    'startind': 6,
    'low_value_threshold': 0.1
}

fix_q_sum_at_power = None  # Set to a power value (e.g. 144) to fix q_sum for that power
fixed_q_sum_value = 1.0  # The value to fix q_sum to
use_pickle = True  # Set to True to save/load processed traces
USE_2Q_MODEL = True  # Set to True to use the 5-state kinetic_2q model


# Loop through all power folders and collect results
pickle_file = os.path.join(base_data_dir, 'processed_data.pkl')

if use_pickle and os.path.exists(pickle_file):
    print(f"Loading processed data from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        all_datasets = pickle.load(f)
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
            norm_pulsephotons_full, timestep = kinetic_model.avtrace(folder_path, partlist, startind=startind, low_value_threshold=low_value_threshold)
            norm_pulsephotons = norm_pulsephotons_full[startind:]
            t = np.linspace(0, (len(norm_pulsephotons) - 1) * timestep, len(norm_pulsephotons))
            
            all_datasets.append({
                'power': power_val,
                'power_str': power_str,
                'has_aa': has_aa,
                'folder_name': folder_name,
                't': t,
                'norm_pulsephotons': norm_pulsephotons,
                'params': params,
                't_phases': (params['onlen'], params['offlen'], params['onlen'], params['offlen'], params['onlen'])
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

    for group_name, datasets in groups:
        print(f"\nProcessing {group_name} group ({len(datasets)} datasets)...")
        n_ds = len(datasets)
        
        # Prepare experimental data and dummy t
        n_penalties_per_ds = 4 if USE_2Q_MODEL else 2
        y_data_combined = np.concatenate([ds['norm_pulsephotons'] for ds in datasets] + [np.zeros(n_penalties_per_ds * n_ds)])
        t_dummy = np.zeros_like(y_data_combined)

        # Define scale factors for more efficient optimization
        n_ds_params = 8 if USE_2Q_MODEL else 7
        if USE_2Q_MODEL:
            scales = [1.0, 1.0, 0.5, 0.2] # kr1_mean, kr2_mean, f_mean, qf_mean
            for _ in range(n_ds):
                scales.extend([0.05, 1.0, 1.0, 0.1, 0.5, 0.5, 1.0, 0.2]) # k1, kr1, kr2, k3, k4, f, q_sum, q_f
        else:
            scales = [1.0, 0.2] # k2_mean, qf_mean
            for _ in range(n_ds):
                scales.extend([0.05, 1.0, 1.0, 0.1, 0.5, 1.0, 0.2]) # k1, k2, k2_light, k3, k4, q_sum, q_f
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
                
                for i in range(self.n_ds):
                    ds = self.datasets[i]
                    ds_params = args[offset:offset + n_ds_params]
                    offset += n_ds_params
                    
                    # Optimization: only re-calculate if local parameters have changed
                    if (self.cache[i] is None or 
                        self.last_params[i] is None or 
                        not np.array_equal(ds_params, self.last_params[i])):
                        
                        t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
                        
                        if USE_2Q_MODEL:
                            k1, kr1, kr2, k3, k4, f, q_sum, q_f = ds_params
                            s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc_2q(
                                ds['t'], k1, kr1, kr2, k3, k4, f, 1.0, q_f,
                                t_dark, t_light, t_dark2, t_light2, t_dark3
                            )
                            # Unquenched states are 3 and 4
                            res = np.concatenate((s1.y[3]+s1.y[4], s2.y[3][1:]+s2.y[4][1:], s3.y[3][1:]+s3.y[4][1:],
                                                   s4.y[3][1:]+s4.y[4][1:], s5.y[3][1:]+s5.y[4][1:], s6.y[3][1:]+s6.y[4][1:]))
                        else:
                            k1, k2, k2_light, k3, k4, q_sum, q_f = ds_params
                            s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
                                ds['t'], k1, k2, k3, k4, 1.0, q_f,
                                t_dark, t_light, t_dark2, t_light2, t_dark3, 
                                k2_light=0,
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
                    # Penalty terms: W * (parameter - mean)
                    if USE_2Q_MODEL:
                        penalties.append(K2_PENALTY_WEIGHT * (ds_params[1] - kr1_mean))
                        penalties.append(K2_PENALTY_WEIGHT * (ds_params[2] - kr2_mean))
                        penalties.append(F_PENALTY_WEIGHT * (ds_params[5] - f_mean))
                        penalties.append(QF_PENALTY_WEIGHT * (ds_params[7] - qf_mean))
                    else:
                        penalties.append(K2_PENALTY_WEIGHT * (ds_params[1] - k2_mean))
                        penalties.append(QF_PENALTY_WEIGHT * (ds_params[6] - qf_mean))
                
                return np.concatenate(fit_results + [np.array(penalties)])

        global_fit_obj = CachedGlobalFit(datasets, scales)

        # Bounds and initial guesses
        if USE_2Q_MODEL:
            p0_global = [0.3, 2, 0.5, 0.18] # Global means: kr1_mean, kr2_mean, f_mean, qf_mean
            lower_bounds = [0, 0, 0, 0.0]
            upper_bounds = [10, 20, 1.0, 0.5]
        else:
            p0_global = [2.6 if group_name == 'AA' else 0.3, 0.18] # Global means: k2_mean, qf_mean
            lower_bounds = [0, 0.0]
            upper_bounds = [10, 0.5]
        
        for ds in datasets:
            dp0 = ds['params']['p0']
            if USE_2Q_MODEL:
                # Expecting/Adapting to k1, kr1, kr2, k3, k4, f, q_sum, q_f
                if len(dp0) >= 8:
                    p0_global.extend(dp0[:8])
                else:
                    # Adapt from 4-state p0: k1, k2, k2_light, k3, k4, q_sum, q_fraction
                    p0_global.extend([dp0[0], dp0[1], 0.03, dp0[3], dp0[4], 0.5, dp0[5], dp0[6]])
                lower_bounds.extend([0, 0, 0, 0, 0, 0, 0.95, 0.0])
                upper_bounds.extend([5, 10, 10, 0.5, 15, 1.0, 1.05, 0.5])
            else:
                # dp0 order assumed: k1, k2, k2_light, k3, k4, q_sum, q_fraction
                # Per-dataset: k1, k2, k2_light, k3, k4, q_sum, q_f
                p0_global.extend([dp0[0], dp0[1], dp0[2] if group_name != 'AA' else 0.0, dp0[3], dp0[4]])
                lower_bounds.extend([0, 0, 0, 0, 0])
                upper_bounds.extend([5, 10, 15 if group_name != 'AA' else 1e-9, 0.5, 15])
                
                # q_sum
                if fix_q_sum_at_power is not None and np.isclose(ds['power'], fix_q_sum_at_power):
                    p0_global.append(fixed_q_sum_value)
                    lower_bounds.append(fixed_q_sum_value - 1e-6)
                    upper_bounds.append(fixed_q_sum_value + 1e-6)
                else:
                    p0_global.append(dp0[5] if len(dp0) > 5 else 1.0)
                    lower_bounds.append(0.95)
                    upper_bounds.append(1.05)
                
                # q_fraction
                p0_global.append(dp0[6] if len(dp0) > 6 else 0.23)
                lower_bounds.append(0.0)
                upper_bounds.append(0.5)

        print(f"  Fitting {len(p0_global)} parameters (scaled)...")
        p0_scaled = np.array(p0_global) / scales
        lower_scaled = np.array(lower_bounds) / scales
        upper_scaled = np.array(upper_bounds) / scales

        popt_scaled, pcov_scaled = curve_fit(global_fit_obj, t_dummy, y_data_combined, p0=p0_scaled, 
                                             bounds=(lower_scaled, upper_scaled), verbose=2, max_nfev=2500,
                                             ftol=1e-6, xtol=1e-6, diff_step=0.1)
        
        popt = popt_scaled * scales
        # Rescale covariance matrix: pcov = J^-1 * (J^-1)^T. If p = s * p_scaled, then J_scaled = J * s
        # pcov = pcov_scaled * s * s^T
        pcov = pcov_scaled * np.outer(scales, scales)
        perr = np.sqrt(np.diag(pcov))
        
        # Extract and store results
        if USE_2Q_MODEL:
            kr1_mean_val, kr2_mean_val, f_mean_val, qf_mean_val = popt[0:4]
            offset = 4
        else:
            k2_mean_val, qf_mean_val = popt[0:2]
            offset = 2

        for i in range(n_ds):
            ds = datasets[i]
            k_vals = popt[offset:offset + n_ds_params]
            k_errs = perr[offset:offset + n_ds_params]
            offset += n_ds_params
            
            if USE_2Q_MODEL:
                k1, kr1, kr2, k3, k4, f_val, q_sum, q_f = k_vals
                pk1, pkr1, pkr2, pk3, pk4, pf, pqs, pqf = k_errs
                tau_list = [1/k if k != 0 else np.nan for k in [k1, kr1, kr2, k3, k4]]
                tau_errs = [k_errs[j] / (k_vals[j]**2) if k_vals[j] != 0 else np.nan for j in range(5)]
            else:
                k1, k2, k2_light, k3, k4, q_sum, q_f = k_vals
                pk1, pk2, pk2l, pk3, pk4, pqs, pqf = k_errs
                tau_list = [1/k if k != 0 else np.nan for k in [k1, k2, k2_light, k3, k4]]
                tau_errs = [k_errs[j] / (k_vals[j]**2) if k_vals[j] != 0 else np.nan for j in range(5)]
            
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
                    'K2_light (s⁻¹)': fmt(k2_light, pk2l),
                    'Tau2 (s)': fmt(tau_list[1], tau_errs[1]),
                    'Tau2_light (s)': fmt(tau_list[2], tau_errs[2]),
                })
            final_results.append(res_dict)
            
            # Recalculate model for plot
            t_dark, t_light, t_dark2, t_light2, t_dark3 = ds['t_phases']
            if USE_2Q_MODEL:
                s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc_2q(
                    ds['t'], k1, kr1, kr2, k3, k4, f_val, 1.0, q_f,
                    t_dark, t_light, t_dark2, t_light2, t_dark3
                )
                model = np.concatenate((s1.y[3]+s1.y[4], s2.y[3][1:]+s2.y[4][1:], s3.y[3][1:]+s3.y[4][1:],
                                       s4.y[3][1:]+s4.y[4][1:], s5.y[3][1:]+s5.y[4][1:], s6.y[3][1:]+s6.y[4][1:]))
            else:
                s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
                    ds['t'], k1, k2, k3, k4, 1, q_f,
                    t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=0,
                )
                model = np.concatenate((s1.y[2]+s1.y[3], s2.y[2][1:]+s2.y[3][1:], s3.y[2][1:]+s3.y[3][1:],
                                       s4.y[2][1:]+s4.y[3][1:], s5.y[2][1:]+s5.y[3][1:], s6.y[2][1:]+s6.y[3][1:]))
            if len(model) < len(ds['t']): model = np.pad(model, (0, len(ds['t']) - len(model)), mode='edge')
            else: model = model[:len(ds['t'])]
            
            plt.figure(figsize=(10, 5))
            plt.plot(ds['t'], ds['norm_pulsephotons'], 'o', color='gray', markersize=3, alpha=0.5, label='Data')
            plt.plot(ds['t'], model, 'r-', linewidth=2, label='Mixed-Effects Global Fit')
            plt.title(f"Mixed-Effects Fit: {ds['folder_name']}")
            plt.legend()
            plt.savefig(os.path.join(base_data_dir, f"mixed_fit_{ds['folder_name']}.png"))
            plt.close()

    # Final summary and export
    df = pd.DataFrame(final_results).sort_values(['AA', 'Power (mE)'])
    print("\nSummary of Mixed-Effects Results:")
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(base_data_dir, 'mixed_effects_results.csv'), index=False)
    
    df_no_aa = df[df['AA'] == 'No'].sort_values('Power (mE)')
    df_with_aa = df[df['AA'] == 'Yes'].sort_values('Power (mE)')
    
    # with open('data_no_aa.pkl', 'wb') as f:
    #     pickle.dump(df_no_aa, f)
    # with open('data_with_aa.pkl', 'wb') as f:
    #     pickle.dump(df_with_aa, f)
    
    print(f"\nResults saved to mixed_effects_results.csv, data_no_aa_k2.pkl and data_with_aa_k2.pkl")

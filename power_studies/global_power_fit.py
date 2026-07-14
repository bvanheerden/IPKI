import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import numpy as np
import os
import json
import pickle
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    'mathtext.fontset': 'stixsans',
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "font.size": 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'legend.fontsize': 7,
})
import seaborn as sns
import kinetic_model

sns.set_palette("deep")

# Data directory - adjust as needed
base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'

pickle_file = os.path.join(base_data_dir, 'processed_data.pkl')

def load_config(folder_path):
    config_file = os.path.join(folder_path, 'config.json')
    params = {
        'partlist': [0, 1, 2],
        'p0': [1 / 20, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.5],
        'onlen': 50,
        'offlen': 200,
        'startind': 6,
        'low_value_threshold': 0.1
    }
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        except Exception as e:
            print(f"  Warning: Could not load config: {e}")
    return params

def prepare_data():
    if os.path.exists(pickle_file):
        print(f"Loading processed data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            # If the user saved a DataFrame instead of the expected list of dicts
            if isinstance(data, pd.DataFrame):
                print("  Warning: Loaded data is a DataFrame, not a list of datasets.")
                # The script expects a list of dictionaries with 'y', 't', 'power', etc.
                # If the DataFrame doesn't have these, we can't easily convert it.
                return []
            return data
    
    all_datasets = []
    all_folders = sorted([f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))])
    
    for folder_name in all_folders:
        folder_path = os.path.join(base_data_dir, folder_name)
        h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
        if not h5_files: continue
        
        params = load_config(folder_path)
        has_aa = folder_name.endswith('AA')
        power_match = re.search(r"(\d+(\.\d+)?)", folder_name)
        power_val = float(power_match.group(1)) if power_match else 0.0
        
        print(f"Processing: {folder_name} (Power={power_val}, AA={has_aa})")
        
        try:
            norm_pulsephotons_full, timestep = kinetic_model.avtrace(
                folder_path, params['partlist'], 
                startind=params['startind'], 
                low_value_threshold=params['low_value_threshold']
            )
            
            startind = params['startind']
            norm_pulsephotons = norm_pulsephotons_full[startind:]
            t = np.linspace(0, (len(norm_pulsephotons) - 1) * timestep, len(norm_pulsephotons))
            
            all_datasets.append({
                'power': power_val,
                'has_aa': has_aa,
                't': t,
                'y': norm_pulsephotons,
                'phases': (params['onlen'], params['offlen']),
                'folder_name': folder_name
            })
        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")
            
    print(f"Saving processed data to {pickle_file}...")
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_datasets, f)
    except Exception as e:
        print(f"  Warning: Could not save processed data to pickle: {e}")
        
    return all_datasets

def model_func(datasets, y1, k2, y2_light, y3, y4, q_fraction, *correction_factors):
    fit_results = []
    for i, ds in enumerate(datasets):
        # Use correction factor if provided, otherwise default to 1.0
        c_factor = correction_factors[i] if i < len(correction_factors) else 1.0
        power = ds['power'] * c_factor
        onlen, offlen = ds['phases']
        
        # Scale rates by power
        k1 = y1 * power
        k2_light = y2_light * power
        k3 = y3 * power
        k4 = y4 * power
        
        t = ds['t']
        t_dark = onlen
        t_light = t_dark + offlen
        t_dark2 = t_light + onlen
        t_light2 = t_dark2 + offlen
        t_dark3 = t_light2 + onlen
        
        # Using kinetic_model.modelfunc
        # Signature: modelfunc(t, k1, k2, k3, k4, q_sum, q_fraction, t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=None)
        s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
            t, k1, k2, k3, k4, 1.0, q_fraction,
            t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light
        )
        
        res = np.concatenate((
            s1.y[2]+s1.y[3], 
            s2.y[2][1:]+s2.y[3][1:], 
            s3.y[2][1:]+s3.y[3][1:],
            s4.y[2][1:]+s4.y[3][1:], 
            s5.y[2][1:]+s5.y[3][1:], 
            s6.y[2][1:]+s6.y[3][1:]
        ))
        
        # Match length
        if len(res) < len(t):
            res = np.pad(res, (0, len(t) - len(res)), mode='edge')
        else:
            res = res[:len(t)]
        
        fit_results.append(res)
    return np.concatenate(fit_results)

def cost_func(params_scaled, datasets, y_all, scale):
    params = np.array(params_scaled) * scale
    y_fit = model_func(datasets, *params)
    # Simple sum of squared residuals
    residuals = y_all - y_fit
    return np.sum(residuals**2)

def global_fit(datasets, group_name="Data"):
    if not datasets:
        print(f"No datasets for {group_name}")
        return
    
    print(f"\nStarting Global Quantum Yield Fit for {group_name}...")
    
    # Concatenate all y data
    try:
        y_all = np.concatenate([ds['y'] for ds in datasets])
    except KeyError as e:
        print(f"Error: Dataset missing key {e}. Check if data was processed correctly.")
        if datasets and isinstance(datasets[0], dict):
             print(f"Available keys: {list(datasets[0].keys())}")
        return
    except Exception as e:
        print(f"Error concatenating data: {e}")
        return
    
    # Define the global model
    # Parameters to fit: y1, y2, y2_light, y3, y4 (quantum yields)
    # plus q_sum and q_fraction which might be shared or per-trace.
    # User said: 'the quantum yields become global fitting parameters that are multiplied with each power for the relevant trace'
    # 'there is no more local fitting steps'
    
    # We will assume q_sum is 1.0 (normalized) and q_fraction is shared globally for the group.
    
    # Initial guesses for quantum yields (k_i / power)
    # Typical k ~ 0.1 to 1.0, typical power ~ 100 to 1000. So y ~ 1e-3
    p0_yields = [4e-3, 4, 2e-3, 2e-3, 5e-3, 0.13]
    # Correction factors for each power (initially 1.0)
    p0_corrections = [1.0] * len(datasets)
    p0_orig = p0_yields + p0_corrections
    
    scale = np.array(p0_orig)
    p0_scaled = np.ones(len(p0_orig))
    
    # Yield bounds and q_fraction bounds
    yield_bounds_lower = [0, 0.1, 0, 0, 0, 0]
    yield_bounds_upper = [0.1, 10, 2, 0.02, 0.2, 0.5]
    
    # Correction factor bounds (0.8 to 1.2)
    corr_bounds_lower = [0.7] * len(datasets)
    corr_bounds_upper = [1.5] * len(datasets)
    
    bounds_orig = (yield_bounds_lower + corr_bounds_lower, yield_bounds_upper + corr_bounds_upper)
    lower_bounds = np.array(bounds_orig[0]) / scale
    upper_bounds = np.array(bounds_orig[1]) / scale
    bounds_scaled = (lower_bounds, upper_bounds)

    def model_func_scaled(t_dummy, *params_scaled):
        params = np.array(params_scaled) * scale
        return model_func(datasets, *params)

    try:
        # differential_evolution bounds are list of (min, max)
        de_bounds = list(zip(lower_bounds, upper_bounds))
        
        print(f"Running Differential Evolution for {group_name}...")
        result = differential_evolution(
            cost_func, 
            de_bounds,
            args=(datasets, y_all, scale),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            workers=-1, # Use all available cores
            disp=True
        )
        
        popt = result.x * scale
        # Differential evolution does not provide pcov. 
        # To get errors, one would usually run a local fit (curve_fit) starting from popt.
        # But the request was to switch to DE.
        perr = np.zeros(len(popt)) 
        
        # Optional: Run a local fit to get the covariance matrix
        try:
            res_lsq = curve_fit(
                model_func_scaled, np.zeros(len(y_all)), y_all, p0=result.x, bounds=bounds_scaled,
                ftol=1e-8, xtol=1e-8
            )
            popt_scaled_local = res_lsq[0]
            pcov_scaled_local = res_lsq[1]
            popt = popt_scaled_local * scale
            pcov = pcov_scaled_local * np.outer(scale, scale)
            perr = np.sqrt(np.diag(pcov))
        except Exception as e_local:
            print(f"  Warning: Local refinement failed: {e_local}")

        names = ['y1', 'y2', 'y2_light', 'y3', 'y4', 'q_fraction']
        # Add names for correction factors
        names += [f'corr_{ds["power"]}' for ds in datasets]
        
        print(f"\nResults for {group_name}:")
        for name, val, err in zip(names, popt, perr):
            print(f"  {name}: {val:.4f} +/- {err:.4f}")
            
        # Plotting the fits
        num_plots = len(datasets)
        cols = 3
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        for i, ds in enumerate(datasets):
            # Recalculate individual fit for plotting
            # Correction factor for this power is in popt[6+i]
            c_factor = popt[6+i]
            power = ds['power'] * c_factor
            k_vals = [popt[j] * power for j in range(5)]
            k_vals[1] = popt[1]  # k2 is not scaled by power
            onlen, offlen = ds['phases']
            t_dark = onlen
            t_light = t_dark + offlen
            t_dark2 = t_light + onlen
            t_light2 = t_dark2 + offlen
            t_dark3 = t_light2 + onlen
            
            s1, s2, s3, s4, s5, s6 = kinetic_model.modelfunc(
                ds['t'], k_vals[0], k_vals[1], k_vals[3], k_vals[4], 1.0, popt[5],
                t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=k_vals[2]
            )
            fit = np.concatenate((s1.y[2]+s1.y[3], s2.y[2][1:]+s2.y[3][1:], s3.y[2][1:]+s3.y[3][1:],
                                  s4.y[2][1:]+s4.y[3][1:], s5.y[2][1:]+s5.y[3][1:], s6.y[2][1:]+s6.y[3][1:]))
            if len(fit) < len(ds['t']): fit = np.pad(fit, (0, len(ds['t']) - len(fit)), mode='edge')
            else: fit = fit[:len(ds['t'])]

            ax = axes[i]
            ax.plot(ds['t'], ds['y'], 'k.', alpha=0.3, label='Data', markersize=2)
            ax.plot(ds['t'], fit, 'r-', label='Global Fit', linewidth=1.5)
            ax.set_title(f"P={ds['power']} (corr={c_factor:.3f}) | {ds['folder_name']}", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Norm. Photons")
            if i == 0:
                ax.legend()
            
            # Also keep saving individual plots as before but maybe in a subfolder
            plt.figure(figsize=(10, 5))
            plt.plot(ds['t'], ds['y'], 'k.', alpha=0.3, label='Data')
            plt.plot(ds['t'], fit, 'r-', label='Global Fit')
            plt.title(f"Global Fit: {ds['folder_name']} (Power={ds['power']}, corr={c_factor:.3f})")
            plt.legend()
            plt.savefig(f"global_fit_{group_name}_{ds['power']}.png")
            plt.close()
            
        # Hide unused axes
        for k in range(len(datasets), len(axes)):
            axes[k].axis('off')
            
        fig.tight_layout()
        fig.suptitle(f"Global Fit Results: {group_name}", fontsize=16)
        fig.subplots_adjust(top=0.92)
        plt.savefig(f"global_fit_summary_{group_name}.png")
        plt.close(fig)

        # Concatenated plot with residuals
        y_fit_all = model_func(datasets, *popt)
        residuals = y_all - y_fit_all

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(y_all, 'k.', alpha=0.3, label='Data', markersize=1)
        ax1.plot(y_fit_all, 'r-', label='Global Fit', linewidth=1)
        ax1.set_ylabel("Norm. Photons")
        ax1.set_title(f"Concatenated Global Fit: {group_name}")
        ax1.legend()

        ax2.plot(residuals, 'b.', alpha=0.3, markersize=1)
        ax2.axhline(0, color='r', linestyle='--')
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("Concatenated Index")

        plt.tight_layout()
        plt.savefig(f"global_fit_concatenated_{str(group_name)}.png")
        plt.close()
            
    except Exception as e:
        print(f"Global fit failed for {group_name}: {e}")

if __name__ == "__main__":
    datasets = prepare_data()
    
    datasets_aa = [ds for ds in datasets if ds['has_aa']]
    datasets_no_aa = [ds for ds in datasets if not ds['has_aa']]
    
    global_fit(datasets_aa, "AA")
    global_fit(datasets_no_aa, "No_AA")
import sys
import os

# Add the project root to sys.path to allow imports from utils and kinetic_models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import utils
from kinetic_models import kinetic_model

utils.setup_plotting()

# Base directory and dataset configuration
base_dir = r'C:\Users\bertu\Desktop'
dataset_folder = '301 mE'
dataset_name = '301 mE'

# Default parameters for fitting
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0_2q = [1 / 4, 1 / 6, 1 / 10, 1 / 10, 1 / 3, 0.5, 1.0, 0.5]
default_p0_2q_fixed = [1 / 4, 1 / 6, 1 / 10, 1 / 10, 1 / 3, 0.5, 1.0] # No q_fraction

startind = 1
default_onlen = 50
default_offlen = 600

def load_and_preprocess_trace(data_dir, partlist=None, onlen=None, offlen=None):
    """Load trace data, average across measurements, and determine phase transition timing."""
    params = kinetic_model.load_params(
        data_dir,
        defaults={
            'onlen': default_onlen,
            'offlen': default_offlen,
            'partlist': default_partlist
        }
    )
    onlen_param = int(params.get('onlen', default_onlen))
    offlen_param = int(params.get('offlen', default_offlen))
    partlist_param = params.get('partlist', default_partlist)

    if onlen is None:
        onlen = onlen_param
    if offlen is None:
        offlen = offlen_param
    if partlist is None:
        partlist = partlist_param

    print(f"Dataset {data_dir} -> using partnums={partlist}, onlen={onlen}, offlen={offlen}")

    norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partlist, startind=slice(None, startind))
    norm_pulsephotons = norm_pulsephotons[startind:]
    datapoints = len(norm_pulsephotons)
    endpoint = (datapoints - 1) * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen

    phase_indices = (t_dark, t_light, t_dark2, t_light2, t_dark3)
    return norm_pulsephotons, t, timestep, phase_indices

def fit_trace_2q(t, norm_pulsephotons, phase_indices, p0=None, fix_qfraction=False):
    """Fit kinetic model with two quenched states (2Q)."""
    t_dark, t_light, t_dark2, t_light2, t_dark3 = phase_indices

    if p0 is None:
        p0 = default_p0_2q_fixed if fix_qfraction else default_p0_2q

    def fitfunc_2q(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction=0.0):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(
            t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction,
            t_dark, t_light, t_dark2, t_light2, t_dark3
        )
        out = np.concatenate((
            sol1.y[3] + sol1.y[4],
            sol2.y[3][1:] + sol2.y[4][1:],
            sol3.y[3][1:] + sol3.y[4][1:],
            sol4.y[3][1:] + sol4.y[4][1:],
            sol5.y[3][1:] + sol5.y[4][1:],
            sol6.y[3][1:] + sol6.y[4][1:]
        ))
        if len(out) < len(t):
            out = np.pad(out, (0, len(t) - len(out)), mode='edge')
        elif len(out) > len(t):
            out = out[:len(t)]
        return out

    if fix_qfraction:
        def fitfunc_fixed(t, k1, kr1, kr2, k3, k4, f, q_sum):
            return fitfunc_2q(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction=0.0)
        
        bounds = ([0, 0, 0, 0, 0, 0, 0.1], [10, 10, 10, 10, 10, 1, 10])
        popt, pcov = curve_fit(fitfunc_fixed, t, norm_pulsephotons, p0=p0, bounds=bounds, verbose=0)
        popt = np.append(popt, 0.0) # pad with q_fraction=0
    else:
        bounds = ([0, 0, 0, 0, 0, 0, 0.1, 0], [10, 10, 10, 10, 10, 1, 10, 1])
        popt, pcov = curve_fit(fitfunc_2q, t, norm_pulsephotons, p0=p0, bounds=bounds, verbose=0)

    model = fitfunc_2q(t, *popt)
    residuals = norm_pulsephotons - model
    return model, residuals, popt

def plot_comparison(t, norm_pulsephotons, model_std, res_std, model_fixed, res_fixed, phase_times, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 75 / 25.4), sharex=True, gridspec_kw={'height_ratios': [3.2, 1], 'hspace': 0.08, 'wspace': 0.25})
    
    models = [
        (model_std, res_std, 'With $U_2$', 'C3', axes[0, 0], axes[1, 0]),
        (model_fixed, res_fixed, 'Without $U_2$', 'C2', axes[0, 1], axes[1, 1])
    ]

    for model, res, title, color, ax_top, ax_bot in models:
        ax_top.plot(t, norm_pulsephotons, '.', color='gray', markersize=1, alpha=0.4)
        ax_top.plot(t, model, '-', color=color, lw=1)
        ax_top.set_title(title, fontsize=7.5, pad=3)
        ax_bot.plot(t, res, '.', color=color, markersize=1, alpha=0.4)
        ax_bot.axhline(0, color='black', linestyle='--', linewidth=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def main():
    folder_path = os.path.join(base_dir, dataset_folder)
    norm_pulsephotons, t, timestep, phase_indices = load_and_preprocess_trace(folder_path)
    phase_times = [idx * timestep for idx in phase_indices]
    
    model_std, res_std, popt_std = fit_trace_2q(t, norm_pulsephotons, phase_indices, fix_qfraction=False)
    model_fixed, res_fixed, popt_fixed = fit_trace_2q(t, norm_pulsephotons, phase_indices, fix_qfraction=True)
    
    out_cmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"IPKI_fit_qfraction_compare_{dataset_folder.replace(' ', '_')}.png")
    plot_comparison(t, norm_pulsephotons, model_std, res_std, model_fixed, res_fixed, phase_times, save_path=out_cmp)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
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
import os
from scipy.optimize import curve_fit
import pandas as pd
import kinetic_model

base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'
# dataset_folder = 'LHCII GCO Control'
dataset_folder = 'LHCII Control 301 mE'
# dataset_name = 'LHCII GCO Control'
dataset_name = 'LHCII Control'

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.5]

startind = 1

# Global defaults; these can be overridden per-dataset by a params.json or params.txt
default_onlen = 50
default_offlen = 600

onlyplot = False


def fittrace(data_dir, partlist=None, onlen=None, offlen=None, p0=None):

    # load per-dataset params; allow partlist in params to override provided partnums
    params = kinetic_model.load_params(data_dir, 
                                       defaults={'onlen': default_onlen, 
                                                'offlen': default_offlen, 
                                                'partlist': default_partlist, 
                                                'p0': default_p0})
    onlen_param = int(params.get('onlen', default_onlen))
    offlen_param = int(params.get('offlen', default_offlen))
    partlist_param = params.get('partlist', default_partlist)
    p0_param = params.get('p0', default_p0)
    
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
        print(f"Dataset {data_dir} -> using partnums={partlist}, onlen={onlen}, offlen={offlen}, p0={p0}")
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
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(t, k1, k2, k3, k4, q_sum, q_fraction, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3, k2_light=k2_light)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0],
                                                                       [10, 10, 10, 10, 10, 2, 1]), verbose=2)

    tau = [1 / popt[i] for i in range(5)]
    q_sum = popt[5]
    q_fraction = popt[6]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        # propagate error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err = [perr[i] / popt[i] ** 2 if popt[i] != 0 else np.nan for i in range(5)]
        q_sum_err = perr[5]
        q_fraction_err = perr[6]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 5
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print(f'Tau1 = {tau[0]:.2f} ± {tau_err[0]:.2f} s')
    print(f'Tau2 = {tau[1]:.2f} ± {tau_err[1]:.2f} s')
    print(f'Tau2_light = {tau[2]:.2f} ± {tau_err[2]:.2f} s')
    print(f'Tau3 = {tau[3]:.2f} ± {tau_err[3]:.2f} s')
    print(f'Tau4 = {tau[4]:.2f} ± {tau_err[4]:.2f} s')
    print(f'Q_sum = {q_sum:.2f} ± {q_sum_err:.2f} cps')
    print(f'Q_fraction = {q_fraction:.2f} ± {q_fraction_err:.2f}')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

    # Return normalized data, model, time base (exclude last because model uses concatenation offsets), taus and their errors
    return norm_pulsephotons, model, t_plot[:-1], tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err, popt, perr


# Process GCO dataset
print("=" * 80)
print(f"Processing: {dataset_name}")
print("=" * 80)

folder_path = os.path.join(base_dir, dataset_folder)

# Check if folder exists
if not os.path.exists(folder_path):
    print(f"Error: Folder not found: {folder_path}")
    exit(1)

# Check if measurement files exist
h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
if not h5_files:
    print(f"Error: No measurement files found in {folder_path}")
    exit(1)

try:
    norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, popt, perr = fittrace(folder_path)

    print(f"\n✓ Successfully processed {dataset_name}")

    # Create plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot experimental data
    ax.plot(t_plot, norm_pulsephotons, 'o-', color='C0', label=f'{dataset_name} (data)',
            markersize=3, linewidth=1.5, alpha=0.2)

    # Plot model fit
    if model is not None:
        ax.plot(t_plot, model, '-', color='C0', label=f'{dataset_name} (fit)',
                linewidth=2.5, alpha=0.9)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized photon count', fontsize=12)
    ax.set_title(f'{dataset_name} - Data and Fit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(base_dir, 'gco_control_fit.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()

except Exception as e:
    print(f"✗ Error processing {dataset_name}: {str(e)}")
    import traceback
    traceback.print_exc()

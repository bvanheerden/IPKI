import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import os
from scipy.optimize import curve_fit
from kinetic_models import kinetic_model
from matplotlib import pyplot as plt
import seaborn as sns
import utils

utils.setup_plotting()

base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
dataset_folder = '301 mE'
dataset_name = '301 mE'

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1 / 10, 1 / 10, 1 / 3, 0.5, 1.0, 0.5]

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
    datapoints = len(norm_pulsephotons)
    endpoint = (datapoints - 1) * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.


    def fitfunc(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[3]+sol1.y[4], sol2.y[3][1:]+sol2.y[4][1:], sol3.y[3][1:]+sol3.y[4][1:],
                               sol4.y[3][1:]+sol4.y[4][1:], sol5.y[3][1:]+sol5.y[4][1:], sol6.y[3][1:]+sol6.y[4][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0.1, 0],
                                                                       [10, 10, 10, 10, 10, 1, 10, 1]), verbose=2)

    tau = [1 / popt[i] for i in range(5)]
    f_param = popt[5]
    q_sum = popt[6]
    q_fraction = popt[7]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        # propagate error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err = [perr[i] / popt[i] ** 2 if popt[i] != 0 else np.nan for i in range(5)]
        f_err = perr[5]
        q_sum_err = perr[6]
        q_fraction_err = perr[7]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 5
        f_err = np.nan
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print(f'Tau1 = {tau[0]:.2f} ± {tau_err[0]:.2f} s')
    print(f'Taur1 = {tau[1]:.2f} ± {tau_err[1]:.2f} s')
    print(f'Taur2 = {tau[2]:.2f} ± {tau_err[2]:.2f} s')
    print(f'Tau3 = {tau[3]:.2f} ± {tau_err[3]:.2f} s')
    print(f'Tau4 = {tau[4]:.2f} ± {tau_err[4]:.2f} s')
    print(f'f = {f_param:.2f} ± {f_err:.2f}')
    print(f'Q_sum = {q_sum:.2f} ± {q_sum_err:.2f} cps')
    print(f'Q_fraction = {q_fraction:.2f} ± {q_fraction_err:.2f}')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])

    # Return normalized data, model, time base, taus and their errors
    return (norm_pulsephotons, model, t_plot, tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err, 
            popt, perr, (t_dark, t_light, t_dark2, t_light2, t_dark3), timestep)


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
    (norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err,
     popt, perr, transition_indices, timestep) = fittrace(folder_path)

    t_dark, t_light, t_dark2, t_light2, t_dark3 = [idx * timestep for idx in transition_indices]

    print(f"\n✓ Successfully processed {dataset_name}")

    # Create plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(90/25.4, 50/25.4))

    # Add light/dark bar at the top
    # Phases: [0, t_dark] Light, [t_dark, t_light] Dark, [t_light, t_dark2] Light, ...
    phases = [
        (0, t_dark, 'white'),
        (t_dark, t_light, 'black'),
        (t_light, t_dark2, 'white'),
        (t_dark2, t_light2, 'black'),
        (t_light2, t_dark3, 'white'),
        (t_dark3, t_plot[-1] if len(t_plot) > 0 else 0, 'black')
    ]
    
    # Plot experimental data
    ax.plot(t_plot, norm_pulsephotons, '.', color='gray', label=f'{dataset_name} (data)',
            markersize=1, alpha=0.4)

    # Plot model fit
    if model is not None:
        ax.plot(t_plot, model, '-', color='C3', label=f'{dataset_name} (fit)', lw=1)

    for start, end, color in phases:
        ax.axvspan(start, end, ymin=0.96, ymax=1.0, facecolor=color,
                   edgecolor='black', linewidth=0.5, transform=ax.get_xaxis_transform())

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized fluorescence')
    ax.set_xlim(0, 90)
    plt.tight_layout()

    # Save plot
    # plot_file = os.path.join(base_dir, 'gco_control_fit.png')
    # plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to: {plot_file}")
    plt.show()

except Exception as e:
    print(f"✗ Error processing {dataset_name}: {str(e)}")
    import traceback
    traceback.print_exc()

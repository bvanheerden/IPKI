import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd

base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'
# dataset_folder = 'LHCII GCO Control'
dataset_folder = 'LHCII Control 301 mE'
# dataset_name = 'LHCII GCO Control'
dataset_name = 'LHCII Control'

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1/ 6, 1 / 10, 1 / 3, 0.5]

startind = 1

# Global defaults; these can be overridden per-dataset by a params.json or params.txt
default_onlen = 50
default_offlen = 600

onlyplot = False


def kinetic(t, y, k1, k2, k3, k4):
    K = np.array([[0,  0,  k4,  k3],  # Bleached
                  [0, -k2, 0, k1],  # Quenced
                  [0, 0, -k4, 0],  # UnQuenched 2
                  [0,  k2, 0, -k1-k3]])  # Unquenched
    return K @ y


def modelfunc(t, k1, k2, k2_light, k3, k4, q0, t_dark, t_light, t_dark2, t_light2, t_dark3):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark+1]], [0, 0, q0, 1-q0], t_eval=t[0:t_dark+1],
                     args=[k1, k2_light, k3, k4])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light+1]], sol1.y[:, -1], t_eval=t[t_dark:t_light+1],
                     args=[0, k2, 0, 0])
    sol3 = solve_ivp(kinetic, [t[t_light], t[t_dark2+1]], sol2.y[:, -1], t_eval=t[t_light:t_dark2+1],
                     args=[k1, k2_light, k3, k4])
    sol4 = solve_ivp(kinetic, [t[t_dark2], t[t_light2+1]], sol3.y[:, -1], t_eval=t[t_dark2:t_light2+1],
                     args=[0, k2, 0, 0])
    sol5 = solve_ivp(kinetic, [t[t_light2], t[t_dark3+1]], sol4.y[:, -1], t_eval=t[t_light2:t_dark3+1],
                     args=[k1, k2_light, k3, k4])
    sol6 = solve_ivp(kinetic, [t[t_dark3], t[-1]], sol5.y[:, -1], t_eval=t[t_dark3:-1],
                     args=[0, k2, 0, 0])
    return sol1, sol2, sol3, sol4, sol5, sol6


def onetrace(data_dir, partnum):
    dataset = h5py.File(os.path.join(data_dir, f'measurement {partnum}.h5'), 'r')
    abstimes = dataset['timestamps'][:] * 50  # 50 ns clock

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]  # gap is at least 20 ms
    boundary_times = abstimes[boundary_photons]  # end of pulse (start of gap)
    boundary_times_start = abstimes[boundary_photons + 1]  # start of pulse
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])  # first photon is start of first pulse
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6  # length of each pulse in ms
    timestep = np.mean(np.diff(boundary_times_start) / 1e9)  # timestep in s

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    # normalize to the start index
    norm_pulsephotons /= np.mean(norm_pulsephotons[:startind])

    # Replace non-finite or extra-low values with the previous valid value
    # Criteria: non-finite, <= 0 or extremely lower than previous point (factor 1e-3)
    for i in range(len(norm_pulsephotons)):
        if not np.isfinite(norm_pulsephotons[i]) or norm_pulsephotons[i] <= 0:
            if i == 0:
                # if first element bad, try to find next good one or set to 1.0
                good = None
                for j in range(1, len(norm_pulsephotons)):
                    if np.isfinite(norm_pulsephotons[j]) and norm_pulsephotons[j] > 0:
                        good = norm_pulsephotons[j]
                        break
                norm_pulsephotons[i] = good if good is not None else 1.0
            else:
                norm_pulsephotons[i] = norm_pulsephotons[i - 1]
        elif i > 0 and norm_pulsephotons[i] < norm_pulsephotons[i - 1] * 1e-3:
            # extremely low outlier compared to previous
            norm_pulsephotons[i] = norm_pulsephotons[i - 1]
    # norm_pulsephotons = uniform_filter1d(norm_pulsephotons, size=3)
    # norm_pulsephotons = median_filter(norm_pulsephotons, size=4)
    return norm_pulsephotons[:], timestep


def avtrace(data_dir, partnums):
    partnums_ = [onetrace(data_dir, partnum)[0] for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    timesteps = [onetrace(data_dir, partnum)[1] for partnum in partnums]
    return np.mean(partnums_, axis=0), np.mean(timesteps, axis=0)


def load_params(data_dir):
    """Load per-dataset parameters from params.json or params.txt in the data directory.
    Supported keys: onlen (int), offlen (int), partlist (list), p0 (list)
    Falls back to defaults when keys are missing.
    """
    params = {}
    json_path = os.path.join(data_dir, 'params.json')
    txt_path = os.path.join(data_dir, 'params.txt')
    if os.path.exists(json_path):
        try:
            import json

            with open(json_path, 'r') as fh:
                params = json.load(fh)
        except Exception:
            params = {}
    elif os.path.exists(txt_path):
        # simple key=value parser; lists should be Python syntax
        try:
            with open(txt_path, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            params[k] = eval(v)
                        except Exception:
                            params[k] = v
        except Exception:
            params = {}

    # fill defaults
    onlen = int(params.get('onlen', default_onlen))
    offlen = int(params.get('offlen', default_offlen))
    partlist = params.get('partlist', default_partlist)
    p0 = params.get('p0', default_p0)
    # debug: show when per-dataset params are found
    if params:
        print(f"Loaded params for {data_dir}: {params}")
    return onlen, offlen, partlist, p0, params


def fittrace(data_dir, partlist=None, onlen=None, offlen=None, p0=None):

    # load per-dataset params; allow partlist in params to override provided partnums
    onlen_, offlen_, partlist_, p0_, params = load_params(data_dir)
    if onlen is None:
        onlen = onlen_
    if offlen is None:
        offlen = offlen_
    if p0 is None:
        p0 = p0_
    if partlist is None:
        partlist = partlist_
    # debug: show which partnums and p0 will be used for this dataset
    try:
        print(f"Dataset {data_dir} -> using partnums={partlist}, onlen={onlen}, offlen={offlen}, p0={p0}")
    except Exception:
        pass

    if onlyplot:
        norm_pulsephotons, timestep = avtrace(data_dir, partlist)
    else:
        norm_pulsephotons, timestep = avtrace(data_dir, partlist)
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


    def fitfunc(t, k1, k2, k2_light, k3, k4, q0):
        sol1, sol2, sol3, sol4, sol5, sol6 = modelfunc(t, k1, k2, k2_light, k3, k4, q0, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0],
                                                                       [10, 10, 10, 10, 10, 1]), verbose=2)

    tau = [1 / popt[i] for i in range(5)]
    q0 = popt[5]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        # propagate error for tau = 1/k: sigma_tau = sigma_k / k^2
        tau_err = [perr[i] / popt[i] ** 2 if popt[i] != 0 else np.nan for i in range(5)]
        q0_err = perr[5]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 5
        q0_err = np.nan

    print(f'Tau1 = {tau[0]:.2f} ± {tau_err[0]:.2f} s')
    print(f'Tau2 = {tau[1]:.2f} ± {tau_err[1]:.2f} s')
    print(f'Tau2_light = {tau[2]:.2f} ± {tau_err[2]:.2f} s')
    print(f'Tau3 = {tau[3]:.2f} ± {tau_err[3]:.2f} s')
    print(f'Tau4 = {tau[4]:.2f} ± {tau_err[4]:.2f} s')
    print(f'Q0 = {q0:.2f} ± {q0_err:.2f} cps')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

    # Return normalized data, model, time base (exclude last because model uses concatenation offsets), taus and their errors
    return norm_pulsephotons, model, t_plot[:-1], tau, tau_err, q0, q0_err, popt, perr


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
    norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q0, q0_err, popt, perr = fittrace(folder_path)

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


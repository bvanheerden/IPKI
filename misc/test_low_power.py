import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import utils
import numpy as np
import os
from scipy.optimize import curve_fit
from kinetic_models import kinetic_model
from matplotlib import pyplot as plt
import seaborn as sns

utils.setup_plotting()

# base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/1 August 2026/Thylakoid power study'
base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/1 August 2026'
datasets = [
    # {'folder': '2 mE AA', 'label': '2', 'color': 'C0', 'fit_color': 'C1'},
    {'folder': '5 mE AA', 'label': '5', 'color': 'C2', 'fit_color': 'C2'},
    # {'folder': '11 mE AA', 'label': '11', 'color': 'C1', 'fit_color': 'C1'}
]

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1 / 6, 1 / 10, 1 / 3, 1.0, 0.5]

# startind = 150
startind = 20

# Global defaults; these can be overridden per-dataset by a params.json or params.txt
default_onlen_first = 1270
default_onlen = 50
default_offlen = 600

onlyplot = False


def fittrace(data_dir, partlist=None, onlen=None, onlen_first=None, offlen=None, p0=None):

    # load per-dataset params; allow partlist in params to override provided partnums
    params = kinetic_model.load_params(data_dir, 
                                       defaults={'onlen': default_onlen, 
                                                'onlen_first': default_onlen_first,
                                                'offlen': default_offlen, 
                                                'partlist': default_partlist, 
                                                'p0': default_p0})
    onlen_param = int(params.get('onlen', default_onlen))
    onlen_first_param = int(params.get('onlen_first', onlen_param))
    offlen_param = int(params.get('offlen', default_offlen))
    partlist_param = params.get('partlist', default_partlist)
    p0_param = params.get('p0', default_p0)
    
    if onlen is None:
        onlen = onlen_param
    if onlen_first is None:
        onlen_first = onlen_first_param
    if offlen is None:
        offlen = offlen_param
    if p0 is None:
        p0 = p0_param
    if partlist is None:
        partlist = partlist_param
    # debug: show which partnums and p0 will be used for this dataset
    try:
        print(f"Dataset {data_dir} -> using partnums={partlist}, onlen={onlen}, onlen_first={onlen_first}, offlen={offlen}, p0={p0}")
    except Exception:
        pass

    norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partlist, startind=slice(None, startind))
    
    norm_pulsephotons_fit = norm_pulsephotons[startind:]
    datapoints_fit = len(norm_pulsephotons_fit)
    endpoint_fit = datapoints_fit * timestep
    t_fit = np.linspace(0, endpoint_fit, datapoints_fit)

    t_dark = onlen_first  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.


    def fitfunc(t, k1, k2, k2_light, k3, k4, q_sum, q_fraction):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(t, k1, 0.8, k3, k4, 1, q_fraction, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3, k2_light=0)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        popt, pcov = curve_fit(fitfunc, t_fit, norm_pulsephotons_fit, p0=p0, bounds=([0, 0, 0, 0, 0, 1, 0],
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

    t_plot = np.arange(len(norm_pulsephotons)) * timestep
    if onlyplot:
        model = None
    else:
        model_fit = fitfunc(t_fit, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        model = np.full_like(norm_pulsephotons, np.nan)
        model[startind:] = model_fit

    # Return normalized data, model, time base (exclude last because model uses concatenation offsets), taus and their errors
    return (norm_pulsephotons, model, t_plot, tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err,
            popt, perr, (t_dark, t_light, t_dark2, t_light2, t_dark3), timestep)


# Create plot
print("\nCreating plot...")
fig, ax = plt.subplots(figsize=utils.get_figure_size(90, 50))

for ds in datasets:
    dataset_name = ds['label']
    dataset_folder = ds['folder']
    
    print("=" * 80)
    print(f"Processing: {dataset_name}")
    print("=" * 80)

    folder_path = os.path.join(base_dir, dataset_folder)

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        continue

    # Check if measurement files exist
    h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
    if not h5_files:
        print(f"Error: No measurement files found in {folder_path}")
        continue

    try:
        (norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err,
         popt, perr, transition_indices, timestep) = fittrace(folder_path)

        t_dark, t_light, t_dark2, t_light2, t_dark3 = [idx * timestep for idx in transition_indices]

        print(f"\n✓ Successfully processed {dataset_name}")

        # Plot experimental data
        ax.plot(t_plot, norm_pulsephotons, '.', color=ds['color'], markersize=1, alpha=0.4)

        # Plot model fit
        if model is not None:
            ax.plot(t_plot, model, '-', color=ds['fit_color'], label=f'{dataset_name}', alpha=1)
            
    except Exception as e:
        print(f"✗ Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()


plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95), frameon=False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized fluorescence (a.u.)')
# ax.set_xlim(0, 50)
# ax.text(45, 0.55, '-AA', color='C0')
# ax.text(45, 0.8, '+AA', color='C1')
plt.tight_layout()

# Save plot
# plot_file = os.path.join(base_dir, 'aa_comparison_602.png')
# plt.savefig(plot_file, dpi=300, bbox_inches='tight')
# print(f"Plot saved to: {plot_file}")
plt.show()

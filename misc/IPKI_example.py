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

# base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
base_dir = r'C:\\Users\\bertu\\Desktop'
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


    def fitfunc(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction, return_populations=False):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        if return_populations:
            # y components: [Bleached, Q1, Q2, Unquenched2, Unquenched]
            pops = []
            for i in range(5):
                pop_i = np.concatenate((sol1.y[i], sol2.y[i][1:], sol3.y[i][1:],
                                        sol4.y[i][1:], sol5.y[i][1:], sol6.y[i][1:]))
                pops.append(pop_i)
            return np.array(pops)

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
        populations = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
        populations = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], 
                              return_populations=True)

    # Return normalized data, model, time base, taus and their errors
    return (norm_pulsephotons, model, populations, t_plot, tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err, 
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
    (norm_pulsephotons, model, populations, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err,
     popt, perr, transition_indices, timestep) = fittrace(folder_path)

    t_dark, t_light, t_dark2, t_light2, t_dark3 = [idx * timestep for idx in transition_indices]

    print(f"\n✓ Successfully processed {dataset_name}")

    # Create plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(90/25.4, 65/25.4))

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

    # Create figure for all 5 populations
    if populations is not None:
        print("Creating populations plot...")

        fontsize = 6
        plt.rcParams.update({"font.size": fontsize,
                             'axes.titlesize': fontsize,
                             'axes.labelsize': fontsize,
                             'xtick.labelsize': fontsize,
                             'legend.fontsize': fontsize,
                             })
        fig2, (ax2, ax_thy_aa, ax_thy_naa) = plt.subplots(3, 1, figsize=(60/25.4, 120/25.4), sharex=False)
        pop_labels = ['B', 'Q1', 'Q2', 'U2', 'U1']
        order = [4, 3, 1, 2, 0]
        for i in order:
            ax2.plot(t_plot, populations[i], label=pop_labels[i])
        ax2.set_title('LHCII Populations')
        
        # --- Add Thylakoid Populations ---
        # Paths for thylakoid data
        thy_base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'
        thy_folders = {'AA': '301 mE AA', 'non-AA': '301 mE'}
        
        def fit_thylakoid(folder_path, has_aa):
            # Parameters from Power_studies_June2026.py and typical thylakoid fits
            onlen_thy = 50
            offlen_thy = 200
            startind_thy = 6
            k2_shared_val = 0.5  # Typical shared k2 for thylakoids
            q_fraction_thy = 0.23 # Fixed in Power_studies_June2026.py fitfunc
            
            # Load and average traces
            norm_pulsephotons, timestep = kinetic_model.avtrace(folder_path, [0, 1, 2], startind=startind_thy, 
                                                                low_value_threshold=0.1)
            norm_pulsephotons = norm_pulsephotons[startind_thy:]
            t = np.linspace(0, (len(norm_pulsephotons)-1)*timestep, len(norm_pulsephotons))
            
            t_dark = onlen_thy
            t_light = t_dark + offlen_thy
            t_dark2 = t_light + onlen_thy
            t_light2 = t_dark2 + offlen_thy
            t_dark3 = t_light2 + onlen_thy
            
            # Local fitfunc for 1-quencher thylakoid model
            def thy_fitfunc(t, k1, k2_light, k3, k4, q_sum, q_fraction, return_populations=False):
                # In AA, k2_light is effectively 0 or handled differently, but here we follow local_fitfunc logic
                res = kinetic_model.modelfunc(t, k1, k2_shared_val, k3, k4, q_sum, q_fraction,
                                              t_dark, t_light, t_dark2, t_light2, t_dark3, 
                                              k2_light=0 if has_aa else k2_light,
                                              return_populations=return_populations)
                if return_populations:
                    return res
                # Return sum of unquenched populations
                sol1, sol2, sol3, sol4, sol5, sol6 = res
                out = np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                                      sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:],
                                      sol6.y[2][1:]+sol6.y[3][1:]))
                if len(out) < len(t):
                    out = np.pad(out, (0, len(t) - len(out)), mode='edge')
                elif len(out) > len(t):
                    out = out[:len(t)]
                return out

            # Initial guess [k1, k2_light, k3, k4, q_sum, q_fraction]
            p0_thy = [0.05, 1.0, 0.16, 0.1, 1.0, 0.2]
            bounds_thy = ([0, 0, 0.002, 0.01, 0.98, 0.1], [5, 13, 0.5, 15, 1.05, 0.3])
            
            popt, _ = curve_fit(thy_fitfunc, t, norm_pulsephotons, p0=p0_thy, bounds=bounds_thy)
            
            # Extract populations
            pops = thy_fitfunc(t, *popt, return_populations=True)
            return t, pops

        try:
            thy_pop_labels = ['B', 'Q', 'U2', 'U1']
            print("Fitting Thylakoid AA...")
            t_aa, pops_aa = fit_thylakoid(os.path.join(thy_base_dir, thy_folders['AA']), has_aa=True)
            for i in [3, 1, 2, 0]: # Order: U1, U2, Q, B
                ax_thy_aa.plot(t_aa, pops_aa[i], label=thy_pop_labels[i])
            ax_thy_aa.set_title('Thylakoid AA')
            
            print("Fitting Thylakoid non-AA...")
            t_naa, pops_naa = fit_thylakoid(os.path.join(thy_base_dir, thy_folders['non-AA']), has_aa=False)
            for i in [2, 3, 1, 0]: # Order: U1, U2, Q, B
                ax_thy_naa.plot(t_naa, pops_naa[i], label=thy_pop_labels[i])
            ax_thy_naa.set_title('Thylakoid non-AA')
        except Exception as thy_e:
            print(f"Warning: Could not plot Thylakoid populations: {thy_e}")

        for ax_curr in [ax_thy_aa, ax_thy_naa]:
            for start, end, color in phases:
                ax_curr.axvspan(start, end, ymin=0.96, ymax=1.0, facecolor=color,
                           edgecolor='black', linewidth=0.5, transform=ax_curr.get_xaxis_transform())
            ax_curr.set_ylabel('Population')
            ax_curr.set_xlim(0, 51)
            ax_curr.set_ylim(0, None)
            ax_curr.legend(loc='lower right', frameon=False)

        ax2.set_xlim(0, 97)
        ax_thy_naa.set_xlabel('Time (s)')
        plt.tight_layout()



except Exception as e:
    print(f"✗ Error processing {dataset_name}: {str(e)}")
    import traceback
    traceback.print_exc()

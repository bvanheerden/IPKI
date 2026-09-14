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
# base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
dataset_folder = '301 mE'
dataset_name = '301 mE'

# Default parameters for fitting
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0_2q = [1 / 4, 1 / 6, 1 / 10, 1 / 10, 1 / 3, 0.5, 1.0, 0.5]
default_p0_1q = [1 / 4, 1 / 8, 1 / 10, 1 / 3, 1.0, 0.5]

startind = 1
default_onlen = 52
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


def fit_trace_2q(t, norm_pulsephotons, phase_indices, p0=None):
    """
    Fit kinetic model with two quenched states (2Q).
    States: [Bleached, Q1, Q2, Unquenched2, Unquenched]
    """
    t_dark, t_light, t_dark2, t_light2, t_dark3 = phase_indices

    if p0 is None:
        p0 = default_p0_2q

    def fitfunc_2q(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction, return_populations=False):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(
            t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction,
            t_dark, t_light, t_dark2, t_light2, t_dark3
        )
        if return_populations:
            pops = []
            for i in range(5):
                pop_i = np.concatenate((
                    sol1.y[i],
                    sol2.y[i][1:],
                    sol3.y[i][1:],
                    sol4.y[i][1:],
                    sol5.y[i][1:],
                    sol6.y[i][1:]
                ))
                if len(pop_i) < len(t):
                    pop_i = np.pad(pop_i, (0, len(t) - len(pop_i)), mode='edge')
                elif len(pop_i) > len(t):
                    pop_i = pop_i[:len(t)]
                pops.append(pop_i)
            return np.array(pops)

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

    bounds = ([0, 0, 0, 0, 0, 0, 0.1, 0], [10, 10, 10, 10, 10, 1, 10, 1])
    popt, pcov = curve_fit(fitfunc_2q, t, norm_pulsephotons, p0=p0, bounds=bounds, verbose=0)

    tau = [1 / popt[i] if popt[i] != 0 else np.nan for i in range(5)]
    f_param = popt[5]
    q_sum = popt[6]
    q_fraction = popt[7]

    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        tau_err = [perr[i] / (popt[i] ** 2) if popt[i] != 0 else np.nan for i in range(5)]
        f_err = perr[5]
        q_sum_err = perr[6]
        q_fraction_err = perr[7]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 5
        f_err = np.nan
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print("\n--- 2-Quenched States (2Q) Fit Results ---")
    print(f'Tau1 (1/k1)   = {tau[0]:.2f} ± {tau_err[0]:.2f} s  (k1 = {popt[0]:.4f} ± {perr[0]:.4f} s⁻¹)')
    print(f'Taur1 (1/kr1) = {tau[1]:.2f} ± {tau_err[1]:.2f} s  (kr1 = {popt[1]:.4f} ± {perr[1]:.4f} s⁻¹)')
    print(f'Taur2 (1/kr2) = {tau[2]:.2f} ± {tau_err[2]:.2f} s  (kr2 = {popt[2]:.4f} ± {perr[2]:.4f} s⁻¹)')
    print(f'Tau3 (1/k3)   = {tau[3]:.2f} ± {tau_err[3]:.2f} s  (k3 = {popt[3]:.4f} ± {perr[3]:.4f} s⁻¹)')
    print(f'Tau4 (1/k4)   = {tau[4]:.2f} ± {tau_err[4]:.2f} s  (k4 = {popt[4]:.4f} ± {perr[4]:.4f} s⁻¹)')
    print(f'f             = {f_param:.2f} ± {f_err:.2f}')
    print(f'Q_sum         = {q_sum:.2f} ± {q_sum_err:.2f} cps')
    print(f'Q_fraction    = {q_fraction:.2f} ± {q_fraction_err:.2f}')

    model = fitfunc_2q(t, *popt)
    populations = fitfunc_2q(t, *popt, return_populations=True)
    residuals = norm_pulsephotons - model
    ssr = np.sum(residuals ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f'Sum of Squared Residuals (SSR) = {ssr:.4e}, RMSE = {rmse:.4e}')

    return model, residuals, populations, popt, perr, tau, tau_err


def fit_trace_1q(t, norm_pulsephotons, phase_indices, p0=None):
    """
    Fit kinetic model with a single quenched state (1Q, single k2 value).
    States: [Bleached, Quenched, Unquenched2, Unquenched]
    """
    t_dark, t_light, t_dark2, t_light2, t_dark3 = phase_indices

    if p0 is None:
        p0 = default_p0_1q

    def fitfunc_1q(t, k1, k2, k3, k4, q_sum, q_fraction, return_populations=False):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc(
            t, k1, k2, k3, k4, q_sum, q_fraction,
            t_dark, t_light, t_dark2, t_light2, t_dark3,
            k2_light=None
        )
        if return_populations:
            pops = []
            for i in range(4):
                pop_i = np.concatenate((
                    sol1.y[i],
                    sol2.y[i][1:],
                    sol3.y[i][1:],
                    sol4.y[i][1:],
                    sol5.y[i][1:],
                    sol6.y[i][1:]
                ))
                if len(pop_i) < len(t):
                    pop_i = np.pad(pop_i, (0, len(t) - len(pop_i)), mode='edge')
                elif len(pop_i) > len(t):
                    pop_i = pop_i[:len(t)]
                pops.append(pop_i)
            return np.array(pops)

        out = np.concatenate((
            sol1.y[2] + sol1.y[3],
            sol2.y[2][1:] + sol2.y[3][1:],
            sol3.y[2][1:] + sol3.y[3][1:],
            sol4.y[2][1:] + sol4.y[3][1:],
            sol5.y[2][1:] + sol5.y[3][1:],
            sol6.y[2][1:] + sol6.y[3][1:]
        ))
        if len(out) < len(t):
            out = np.pad(out, (0, len(t) - len(out)), mode='edge')
        elif len(out) > len(t):
            out = out[:len(t)]
        return out

    bounds = ([0, 0, 0, 0, 0.1, 0], [10, 10, 10, 10, 10, 1])
    popt, pcov = curve_fit(fitfunc_1q, t, norm_pulsephotons, p0=p0, bounds=bounds, verbose=0)

    tau = [1 / popt[i] if popt[i] != 0 else np.nan for i in range(4)]
    q_sum = popt[4]
    q_fraction = popt[5]

    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        tau_err = [perr[i] / (popt[i] ** 2) if popt[i] != 0 else np.nan for i in range(4)]
        q_sum_err = perr[4]
        q_fraction_err = perr[5]
    else:
        perr = [np.nan] * len(popt)
        tau_err = [np.nan] * 4
        q_sum_err = np.nan
        q_fraction_err = np.nan

    print("\n--- Single Quenched State (1Q, single k2) Fit Results ---")
    print(f'Tau1 (1/k1) = {tau[0]:.2f} ± {tau_err[0]:.2f} s  (k1 = {popt[0]:.4f} ± {perr[0]:.4f} s⁻¹)')
    print(f'Tau2 (1/k2) = {tau[1]:.2f} ± {tau_err[1]:.2f} s  (k2 = {popt[1]:.4f} ± {perr[1]:.4f} s⁻¹)')
    print(f'Tau3 (1/k3) = {tau[2]:.2f} ± {tau_err[2]:.2f} s  (k3 = {popt[2]:.4f} ± {perr[2]:.4f} s⁻¹)')
    print(f'Tau4 (1/k4) = {tau[3]:.2f} ± {tau_err[3]:.2f} s  (k4 = {popt[3]:.4f} ± {perr[3]:.4f} s⁻¹)')
    print(f'Q_sum       = {q_sum:.2f} ± {q_sum_err:.2f} cps')
    print(f'Q_fraction  = {q_fraction:.2f} ± {q_fraction_err:.2f}')

    model = fitfunc_1q(t, *popt)
    populations = fitfunc_1q(t, *popt, return_populations=True)
    residuals = norm_pulsephotons - model
    ssr = np.sum(residuals ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f'Sum of Squared Residuals (SSR) = {ssr:.4e}, RMSE = {rmse:.4e}')

    return model, residuals, populations, popt, perr, tau, tau_err



def plot_comparison_residuals(t, norm_pulsephotons, model_1q, res_1q, model_2q, res_2q,
                              phase_times, xlim=(32, 65), save_path=None):
    """Plot side-by-side comparison of 1Q vs 2Q fits and their residuals."""
    t_dark, t_light, t_dark2, t_light2, t_dark3 = phase_times

    fig, axes = plt.subplots(
        2, 2,
        figsize=(150 / 25.4, 75 / 25.4),
        sharex=True,
        sharey='row',
        gridspec_kw={'height_ratios': [3.2, 1]}#, 'hspace': 0.08, 'wspace': 0.05}
    )

    phases = [
        (0, t_dark, 'white'),
        (t_dark, t_light, 'black'),
        (t_light, t_dark2, 'white'),
        (t_dark2, t_light2, 'black'),
        (t_light2, t_dark3, 'white'),
        (t_dark3, t[-1] if len(t) > 0 else 0, 'black')
    ]

    models = [
        (model_1q, res_1q, 'Single Quenched State', 'C0', axes[0, 0], axes[1, 0]),
        (model_2q, res_2q, 'Two Quenched States', 'C3', axes[0, 1], axes[1, 1])
    ]

    for model, res, title, color, ax_top, ax_bot in models:
        # Main trace + fit
        ax_top.plot(t, norm_pulsephotons, '.', color='gray', label=f'{dataset_name} (data)',
                    markersize=1, alpha=0.4)
        ax_top.plot(t, model, '-', color=color, label=f'{dataset_name} (fit)', lw=1)
        for start, end, bar_color in phases:
            ax_top.axvspan(start, end, ymin=0.96, ymax=1.0, facecolor=bar_color,
                           edgecolor='black', linewidth=0.5, transform=ax_top.get_xaxis_transform())

        ax_top.set_title(title, fontsize=7, pad=3)
        if xlim is not None:
            ax_top.set_xlim(xlim)
        # ax_top.legend(frameon=False, loc='lower left', fontsize=6.5)

        # Residuals
        ax_bot.plot(t, res, '.', color=color, markersize=1, alpha=0.4)
        ax_bot.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.7)
        ax_bot.set_xlabel('Time (s)')
        if xlim is not None:
            ax_bot.set_xlim(xlim)
        res_max = np.nanmax(np.abs(res)) * 1.15
        ax_bot.set_ylim(-0.025, 0.025)
        ax_top.set_ylim(0.5, 0.77)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    axes[1, 0].set_ylabel('Residuals')
    axes[0, 0].set_ylabel('Fluorescence (norm.)')

    return fig


def main():
    folder_path = os.path.join(base_dir, dataset_folder)

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return

    h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
    if not h5_files:
        print(f"Error: No measurement files found in {folder_path}")
        return

    print("=" * 80)
    print(f"Processing dataset: {dataset_name} ({folder_path})")
    print("=" * 80)

    norm_pulsephotons, t, timestep, phase_indices = load_and_preprocess_trace(folder_path)
    phase_times = [idx * timestep for idx in phase_indices]

    # 1. Fit Two Quenched States model (2Q)
    model_2q, res_2q, pops_2q, popt_2q, perr_2q, tau_2q, tau_err_2q = fit_trace_2q(
        t, norm_pulsephotons, phase_indices
    )

    # 2. Fit Single Quenched State model (1Q, single k2 value)
    model_1q, res_1q, pops_1q, popt_1q, perr_1q, tau_1q, tau_err_1q = fit_trace_1q(
        t, norm_pulsephotons, phase_indices
    )

    # Output filenames
    save_dir = os.path.dirname(os.path.abspath(__file__))
    out_cmp = os.path.join(save_dir, f"IPKI_fit_comparison_residuals_{dataset_folder.replace(' ', '_')}.png")


    # Plot comparison of 1Q vs 2Q
    print("\nCreating 1Q vs 2Q comparison plot with residuals...")
    fig_cmp = plot_comparison_residuals(
        t, norm_pulsephotons, model_1q, res_1q, model_2q, res_2q,
        phase_times,
        save_path=out_cmp
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

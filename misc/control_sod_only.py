import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from kinetic_models import kinetic_model
import utils

utils.setup_plotting()


# base_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'
base_dir = r'C:\\Users\\bertu\\Desktop'

# Datasets to process (map display names to folder names)
datasets = {
    'LHCII Control': 'LHCII Control 301 mE',
    'LHCII SOD': 'LHCII SOD',
}

# Default parameters for each dataset (can be customized per dataset)
default_partlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
default_p0 = [1 / 4, 1 / 6, 1.5, 1 / 10, 1 / 3, 0.5, 1.0, 0.5]

# Global defaults; these can be overridden per-dataset by a params.json or params.txt
default_startind = 1
default_onlen = 50
default_offlen = 600

onlyplot = False


def fittrace(data_dir, partlist=None, onlen=None, offlen=None, p0=None, startind=None):

    # load per-dataset params; allow partlist in params to override provided partnums
    params = kinetic_model.load_params(data_dir, 
                                       defaults={'startind': default_startind, 
                                                'onlen': default_onlen, 
                                                'offlen': default_offlen, 
                                                'partlist': default_partlist, 
                                                'p0': default_p0})
    startind_param = int(params.get('startind', default_startind))
    onlen_param = int(params.get('onlen', default_onlen))
    offlen_param = int(params.get('offlen', default_offlen))
    partlist_param = params.get('partlist', default_partlist)
    print(partlist_param)
    p0_param = params.get('p0', default_p0)
    
    if startind is None:
        startind = startind_param
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
        print(f"Dataset {data_dir} -> using partnums={partlist}, startind={startind}, onlen={onlen}, offlen={offlen}, p0={p0}")
    except Exception:
        pass

    if onlyplot:
        norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partlist, startind=max(1, startind))
    else:
        norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partlist, startind=max(1, startind))
        norm_pulsephotons = norm_pulsephotons[startind:]
    datapoints = len(norm_pulsephotons)
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.


    def fitfunc(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction):
        sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(t, k1, kr1, kr2, k3, k4, f, 1, 0.23, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[3]+sol1.y[4], sol2.y[3][1:]+sol2.y[4][1:], sol3.y[3][1:]+sol3.y[4][1:],
                               sol4.y[3][1:]+sol4.y[4][1:], sol5.y[3][1:]+sol5.y[4][1:], sol6.y[3][1:]+sol6.y[4][1:]))


    if onlyplot:
        popt = p0
        pcov = None
    else:
        # noinspection PyTupleAssignmentBalance
        popt, pcov = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, ftol=1e-10, xtol=1e-10,
                               bounds=([0, 0, 0, 0, 0, 0, 0, 0], [10, 10, 10, 10, 10, 1, 2, 1]), verbose=2,)

    tau = [1 / popt[i] for i in range(5)]
    f = popt[5]
    q_sum = popt[6]
    q_fraction = popt[7]

    # compute errors if covariance available
    if pcov is not None and np.all(np.isfinite(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
        print(popt, perr)
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

    print(f'Tau1 = {tau[0]:.2g} ± {tau_err[0]:.2g} s')
    print(f'Tau_r1 = {tau[1]:.2g} ± {tau_err[1]:.2g} s')
    print(f'Tau_r2 = {tau[2]:.2g} ± {tau_err[2]:.2g} s')
    print(f'Tau3 = {tau[3]:.2g} ± {tau_err[3]:.2g} s')
    print(f'Tau4 = {tau[4]:.2g} ± {tau_err[4]:.2g} s')
    print(f'f = {f:.2g} ± {f_err:.2g}')
    print(f'Q_sum = {q_sum:.2g} ± {q_sum_err:.2g} cps')
    print(f'Q_fraction = {q_fraction:.2g} ± {q_fraction_err:.2g}')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])

    # Return normalized data, model, time base, taus, errors, etc., AND timing parameters
    return (norm_pulsephotons, model, t_plot, tau, tau_err, q_sum, q_sum_err, q_fraction, q_fraction_err, popt,
            perr, pcov, (t_dark, t_light, t_dark2, t_light2, t_dark3))


# Process all datasets
print("=" * 80)
print("Processing Control and SOD datasets...")
print("=" * 80)

results = []
all_data = {}
covariance_data = {}

for display_name, folder_name in datasets.items():
    folder_path = os.path.join(base_dir, folder_name)

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found for {display_name}: {folder_path}")
        continue

    # Check if measurement files exist
    h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
    if not h5_files:
        print(f"Warning: No measurement files found in {folder_path}")
        continue

    print(f"\nProcessing: {display_name}")
    try:
        # call fittrace without passing the global default_partlist so that any per-dataset
        # 'partlist' in params.json / params.txt will be used automatically
        (norm_pulsephotons, model, t_plot, tau_list, tau_err_list, q_sum, q_sum_err, q_fraction, q_fraction_err, popt,
         perr, pcov, t_params) = fittrace(folder_path)

        # Store data for plotting
        all_data[display_name] = {
            'data': norm_pulsephotons,
            'model': model,
            'time': t_plot,
            'popt': popt,
            'perr': perr,
            'pcov': pcov,
            't_params': t_params
        }

        # Store covariance matrix for correlation analysis
        covariance_data[display_name] = pcov

        # Store results for table (with errors when available)
        def fmt_val_err(val, err):
            try:
                if np.isfinite(err):
                    return f"{val:.2g} ± {err:.2g}"
            except Exception:
                pass
            return f"{val:.2g}"

        results.append({
            'Dataset': display_name,
            'Tau1 (s)': fmt_val_err(tau_list[0], tau_err_list[0]),
            'Tau_r1 (s)': fmt_val_err(tau_list[1], tau_err_list[1]),
            'Tau_r2 (s)': fmt_val_err(tau_list[2], tau_err_list[2]),
            'Tau3 (s)': fmt_val_err(tau_list[3], tau_err_list[3]),
            'Tau4 (s)': fmt_val_err(tau_list[4], tau_err_list[4]),
            'f': fmt_val_err(popt[5], perr[5]),
            'Q_sum': fmt_val_err(q_sum, q_sum_err),
            'Q_fraction': fmt_val_err(q_fraction, q_fraction_err)
        })

        print(f"✓ Successfully processed {display_name}")
    except Exception as e:
        raise e
        print(f"✗ Error processing {display_name}: {str(e)}")
        continue

print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save results to CSV
    output_file = os.path.join(base_dir, 'fitted_parameters_control_sod.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

# Create plot
print("\n" + "=" * 80)
print("Creating plot...")
print("=" * 80)

fig, ax = plt.subplots(figsize=(7, 4))

plot_datasets = ['LHCII Control', 'LHCII SOD']
colors = {'LHCII Control': 'C0', 'LHCII SOD': 'C3'}

for display_name in plot_datasets:
    if display_name in all_data:
        data = all_data[display_name]
        color = colors[display_name]
        # Plot experimental data
        ax.plot(data['time'], data['data'], 'o-', color=color, label=f'{display_name} (data)',
                markersize=3, linewidth=1.5, alpha=0.2)
        # Plot model fit
        if data['model'] is not None:
            ax.plot(data['time'], data['model'], '-', color=color, label=f'{display_name} (fit)',
                    linewidth=2.5, alpha=0.9)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Normalized photon count', fontsize=12)
ax.set_title('Control and SOD Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# Save plot
plot_file = os.path.join(base_dir, 'control_sod_comparison.pdf')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")
plt.close(fig)

# Create population plots
print("\n" + "=" * 80)
print("Creating population plot...")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
plot_datasets = ['LHCII Control', 'LHCII SOD']
state_names = ['Bleached (State 0)', 'Quenched 1 (State 1)', 'Quenched 2 (State 2)', 'Unquenched 2 (State 3)', 'Unquenched (State 4)']

for i, display_name in enumerate(plot_datasets):
    if display_name in all_data:
        ax = axes[i]
        data = all_data[display_name]
        popt = data['popt']
        t = np.append(data['time'], data['time'][-1] + (data['time'][1] - data['time'][0]))
        t_dark, t_light, t_dark2, t_light2, t_dark3 = data['t_params']
        
        # Recalculate model populations using the 2q model
        sols = kinetic_model.modelfunc_2q(t, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7],
                                         t_dark, t_light, t_dark2, t_light2, t_dark3)
        
        # Combine population trajectories
        pop_trajectories = [[] for _ in range(5)]
        for s_idx, sol in enumerate(sols):
            # sol.y has shape (5, points)
            for state_idx in range(5):
                y_vals = sol.y[state_idx]
                if s_idx > 0:
                    y_vals = y_vals[1:]
                pop_trajectories[state_idx].extend(y_vals)
        
        # Plot each population
        for state_idx in range(5):
            ax.plot(data['time'], pop_trajectories[state_idx][:-1], label=state_names[state_idx], linewidth=2)
            
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title(f'Model Populations: {display_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()
# plt.show()

# Save population plot
pop_plot_file = os.path.join(base_dir, 'control_sod_populations.png')
plt.savefig(pop_plot_file, dpi=300, bbox_inches='tight')
print(f"Population plot saved to: {pop_plot_file}")
plt.close(fig)


# Create bar plot of fold-change (SOD / Control) for k1-k4
print("\n" + "=" * 80)
print("Creating fold-change bar plot...")
print("=" * 80)

if 'LHCII Control' in all_data and 'LHCII SOD' in all_data:
    control_popt = all_data['LHCII Control']['popt']
    sod_popt = all_data['LHCII SOD']['popt']

    # Rates: k1=idx 0, kr1=idx 1, kr2=idx 2, k3=idx 3, k4=idx 4, f=idx 5
    # The user wants weighted average of kr1 and kr2: k_r_avg = 1 / (f / kr1 + (1 - f) / kr2)
    
    # Standard errors for individual parameters
    control_perr = all_data['LHCII Control']['perr']
    sod_perr = all_data['LHCII SOD']['perr']

    def get_kr_avg(popt, pcov):
        k1, kr1, kr2, k3, k4, f = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        kr_avg = 1 / (f / kr1 + (1 - f) / kr2)
        
        # Jacobian for kr_avg with respect to [kr1, kr2, f]
        # kr_avg = (f/kr1 + (1-f)/kr2)^-1
        # d(kr_avg)/dkr1 = f * (kr_avg / kr1)**2
        # d(kr_avg)/dkr2 = (1-f) * (kr_avg / kr2)**2
        # d(kr_avg)/df = kr_avg**2 * (1/kr2 - 1/kr1)
        jac = np.array([f * (kr_avg / kr1)**2, (1 - f) * (kr_avg / kr2)**2, kr_avg**2 * (1/kr2 - 1/kr1)])
        
        # Indices of [kr1, kr2, f] in popt are [1, 2, 5]
        indices = [1, 2, 5]
        sub_cov = pcov[np.ix_(indices, indices)]
        
        kr_avg_var = jac @ sub_cov @ jac.T
        kr_avg_err = np.sqrt(max(0, kr_avg_var))
        return kr_avg, kr_avg_err

    control_kr_avg, control_kr_avg_err = get_kr_avg(control_popt, all_data['LHCII Control']['pcov'])
    sod_kr_avg, sod_kr_avg_err = get_kr_avg(sod_popt, all_data['LHCII SOD']['pcov'])

    k_indices = [0, 3, 4]
    k_labels = [r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$']
    
    k_control = [control_popt[0], control_kr_avg, control_popt[3], control_popt[4]]
    k_sod = [sod_popt[0], sod_kr_avg, sod_popt[3], sod_popt[4]]
    
    k_control_err = [control_perr[0], control_kr_avg_err, control_perr[3], control_perr[4]]
    k_sod_err = [sod_perr[0], sod_kr_avg_err, sod_perr[3], sod_perr[4]]

    print('k_control', k_control)
    print('k_control_err', k_control_err)
    print('k_sod', k_sod)
    print('k_sod_err', k_sod_err)

    fold_changes = [sod / ctrl if ctrl != 0 else np.nan for sod, ctrl in zip(k_sod, k_control)]
    
    fold_change_errs = []
    for sod, ctrl, sod_err, ctrl_err in zip(k_sod, k_control, k_sod_err, k_control_err):
        if ctrl != 0 and sod != 0:
            rel_err_sq = (sod_err / sod)**2 + (ctrl_err / ctrl)**2
            err = (1 / np.log(2)) * np.sqrt(rel_err_sq)
            fold_change_errs.append(err)
        else:
            fold_change_errs.append(np.nan)

    fold_changes = np.log2(fold_changes)

    fig, ax = plt.subplots(figsize=(70/25.4, 40/25.4))
    bars = ax.bar(k_labels, fold_changes, yerr=fold_change_errs, capsize=3, error_kw=dict(elinewidth=1),
                  color=['C0', 'C4', 'C1', 'C2'], alpha=0.8, edgecolor='black')
    
    ax.set_ylabel(r'Fold-change $\log_2$(SOD / Control)')

    # Add text labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        err = fold_change_errs[i]
        if height > 0:
            y_pos = height + err + 0.05
            ax.text(bar.get_x() + bar.get_width()/2., y_pos, f'{height:.2f}', ha='center', va='bottom')
        else:
            print(err)
            y_pos = height - err - 0.07
            ax.text(bar.get_x() + bar.get_width()/2., y_pos, f'{height:.2f}', ha='center', va='top')

    # ax.set_ylim([-1, 0.5])
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=True, labelbottom=True)
    ax.spines['top'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Hide the first two labels from the top and the last two from the bottom
    labels = ax.get_xticklabels()
    # Note: With labeltop=True and labelbottom=True, we might have two sets of labels
    # but usually Matplotlib handles them together if not using secondary_xaxis.
    # Let's use a more robust approach:
    
    # Refresh the figure to ensure ticks are populated
    fig.canvas.draw()
    
    # Get all tick objects
    ticks = ax.xaxis.get_major_ticks()
    for i, tick in enumerate(ticks):
        if i < 2:
            # First two labels below the axis
            tick.label1.set_visible(True)
            tick.label2.set_visible(False)
        else:
            # Other labels above the axis
            tick.label1.set_visible(False)
            tick.label2.set_visible(True)

    plt.tight_layout()
    plt.show()


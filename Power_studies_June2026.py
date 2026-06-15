import numpy as np
import h5py
import os
import json
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt

# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'

# Default parameters (used if no config file is found)
default_params = {
    'partlist': [0, 1, 2],
    'p0': [1 / 4, 1 / 6, 1 / 10, 1 / 3, 0.5],
    'onlen': 50,
    'offlen': 200,
    'startind': 6,
    'low_value_threshold': 0.1
}

onlyplot = False


def load_config(folder_path):
    """Load configuration from config.json in the folder, return dict with parameters."""
    config_file = os.path.join(folder_path, 'config.json')
    params = default_params.copy()

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                params.update(loaded_config)
            print(f"  Loaded config from {config_file}")
        except Exception as e:
            print(f"  Warning: Could not load config file: {str(e)}")
            print(f"  Using default parameters")
    else:
        print(f"  No config file found, using default parameters")

    return params


def kinetic(t, y, k1, k2, k3, k4):
    K = np.array([[0,  0,  k4,  k3],  # Bleached
                  [0, -k2, 0, k1],  # Quenced
                  [0, 0, -k4, 0],  # UnQuenched 2
                  [0,  k2, 0, -k1-k3]])  # Unquenched
    return K @ y


def modelfunc(t, k1, k2, k3, k4, q0, t_dark, t_light, t_dark2, t_light2, t_dark3):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark+1]], [0, 0, q0, 1-q0], t_eval=t[0:t_dark+1],
                     args=[k1, k2, k3, k4])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light+1]], sol1.y[:, -1], t_eval=t[t_dark:t_light+1],
                     args=[0, k2, 0, 0])
    sol3 = solve_ivp(kinetic, [t[t_light], t[t_dark2+1]], sol2.y[:, -1], t_eval=t[t_light:t_dark2+1],
                     args=[k1, k2, k3, k4])
    sol4 = solve_ivp(kinetic, [t[t_dark2], t[t_light2+1]], sol3.y[:, -1], t_eval=t[t_dark2:t_light2+1],
                     args=[0, k2, 0, 0])
    sol5 = solve_ivp(kinetic, [t[t_light2], t[t_dark3+1]], sol4.y[:, -1], t_eval=t[t_light2:t_dark3+1],
                     args=[k1, k2, k3, k4])
    sol6 = solve_ivp(kinetic, [t[t_dark3], t[-1]], sol5.y[:, -1], t_eval=t[t_dark3:-1],
                     args=[0, k2, 0, 0])
    return sol1, sol2, sol3, sol4, sol5, sol6


def onetrace(data_dir, partnum, startind, low_value_threshold):
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
    norm_pulsephotons /= np.mean(norm_pulsephotons[startind])

    # Remove infs first
    norm_pulsephotons = norm_pulsephotons[np.isfinite(norm_pulsephotons)]

    # Replace extra-low values with the previous value
    mean_val = np.mean(norm_pulsephotons)
    threshold = float(mean_val * low_value_threshold)
    for i in range(1, len(norm_pulsephotons)):
        if norm_pulsephotons[i] < threshold:
            norm_pulsephotons[i] = norm_pulsephotons[i - 1]

    # norm_pulsephotons = uniform_filter1d(norm_pulsephotons, size=3)
    # norm_pulsephotons = median_filter(norm_pulsephotons, size=4)
    return norm_pulsephotons[:], timestep


def avtrace(data_dir, partnums, startind, low_value_threshold):
    partnums_ = [onetrace(data_dir, partnum, startind, low_value_threshold)[0] for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    timesteps = [onetrace(data_dir, partnum, startind, low_value_threshold)[1] for partnum in partnums]
    return np.mean(partnums_, axis=0), np.mean(timesteps, axis=0)


def fittrace(data_dir, partnums, onlen, offlen, startind, p0, low_value_threshold, k2_fixed):

    if onlyplot:
        norm_pulsephotons, timestep = avtrace(data_dir, partnums, startind, low_value_threshold)
    else:
        norm_pulsephotons, timestep = avtrace(data_dir, partnums, startind, low_value_threshold)
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


    def fitfunc(t, k1, k2, k3, k4, q0):
        sol1, sol2, sol3, sol4, sol5, sol6 = modelfunc(t, k1, k2_fixed, k3, k4, 0.207, t_dark,
        # sol1, sol2, sol3, sol4, sol5, sol6 = modelfunc(t, k1, k2, k3, k4, q0, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
    else:
        popt, pcov, *extra = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0],
                                       [10, 10, 10, 10, 1]), verbose=2)

    k1 = popt[0]
    k2 = popt[1]
    k3 = popt[2]
    k4 = popt[3]
    q0 = popt[4]

    tau1 = 1 / k1
    tau2 = 1 / k2
    tau3 = 1 / k3
    tau4 = 1 / k4

    print(f'K1 = {k1:.4f} s⁻¹, Tau1 = {tau1:.2f} s')
    print(f'K2 = {k2:.4f} s⁻¹, Tau2 = {tau2:.2f} s')
    print(f'K3 = {k3:.4f} s⁻¹, Tau3 = {tau3:.2f} s')
    print(f'K4 = {k4:.4f} s⁻¹, Tau4 = {tau4:.2f} s')
    print(f'Q0 = {q0:.2f} cps')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, k1, k2, k3, k4, q0)

    return norm_pulsephotons, model, t_plot[:-1], tau1, tau2, tau3, tau4, q0, k1, k2, k3, k4


# Loop through all power folders and collect results
results = []

# Get all folders in the base directory
all_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]

# Sort folders for consistent ordering
all_folders = sorted(all_folders)

print("Processing power folders...")
print("=" * 80)

for folder_name in all_folders:
    folder_path = os.path.join(base_data_dir, folder_name)

    # Check if there are measurement files in this folder
    h5_files = [f for f in os.listdir(folder_path) if f.startswith('measurement') and f.endswith('.h5')]
    if not h5_files:
        print(f"Skipping {folder_name} - no measurement files found")
        continue

    # Load configuration for this folder
    print(f"\nProcessing: {folder_name}")
    params = load_config(folder_path)

    # Extract parameters from config
    partlist = params['partlist']
    p0 = params['p0']
    onlen = params['onlen']
    offlen = params['offlen']
    startind = params['startind']
    low_value_threshold = params['low_value_threshold']

    print(f"  Parameters: onlen={onlen}, offlen={offlen}, partlist={partlist}")

    # Extract power and AA info from folder name
    has_aa = folder_name.endswith('AA')
    power_str = folder_name.replace(' AA', '').strip()

    print(f"  Power: {power_str}, AA: {has_aa}")

    if has_aa:
        # k2_fixed = 1.5  # LHCII
        k2_fixed = 2.14  # Thylakoid
    else:
        # k2_fixed = 0.145  # LHCII
        k2_fixed = 0.468  # Thylakoid

    try:
        # Fit the trace and store result tuple
        result = fittrace(folder_path, partlist, onlen, offlen, startind, p0, low_value_threshold, k2_fixed)
        # result = (norm_pulsephotons, model, t_plot, tau1, tau2, tau3, tau4, q0, k1, k2, k3, k4)

        # Store results
        results.append({
            'Power': power_str,
            'AA': 'Yes' if has_aa else 'No',
            'K1 (s⁻¹)': result[8],   # k1
            'K2 (s⁻¹)': result[9],   # k2
            'K3 (s⁻¹)': result[10],  # k3
            'K4 (s⁻¹)': result[11],  # k4
            'Tau1 (s)': result[3],   # tau1
            'Tau2 (s)': result[4],   # tau2
            'Tau3 (s)': result[5],   # tau3
            'Tau4 (s)': result[6],   # tau4
            'Q0': result[7]          # q0
        })

        # Create and save individual trace plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result[2], result[0], 'o-', color='gray', label='Experimental data',
                markersize=4, linewidth=1.5, alpha=0.7)
        if result[1] is not None:
            ax.plot(result[2], result[1], '-', color='red', label='Model fit', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Normalized photon count', fontsize=12)
        aa_label = " (with AA)" if has_aa else ""
        ax.set_title(f'Trace and Fit for {power_str}{aa_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot with descriptive filename
        aa_suffix = '_AA' if has_aa else ''
        plot_filename = f"trace_{power_str.replace(' ', '_')}{aa_suffix}.png"
        plot_filepath = os.path.join(folder_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"✓ Successfully processed {folder_name}")
        print(f"  Trace plot saved to: {plot_filename}")

    except Exception as e:
        print(f"✗ Error processing {folder_name}: {str(e)}")
        continue

print("\n" + "=" * 80)
print("\nSummary of Results:")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save results to CSV
    output_file = os.path.join(base_data_dir, 'analysis_results.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create a plot of k values vs power
    print("\n" + "=" * 80)
    print("Creating k values vs power plot...")
    print("=" * 80)

    print('Average Q0 = ', df['Q0'].mean())

    # Extract power values and convert to numeric (removing 'mE')
    df['Power_numeric'] = df['Power'].str.replace(' mE', '').astype(float)

    # Separate data with and without AA
    df_no_aa = df[df['AA'] == 'No'].sort_values('Power_numeric')
    df_with_aa = df[df['AA'] == 'Yes'].sort_values('Power_numeric')

    # Create figure with k values
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create secondary y-axis for k4
    ax2 = ax.twinx()

    # Plot k values without AA on primary axis
    if len(df_no_aa) > 0:
        ax.plot(df_no_aa['Power_numeric'], df_no_aa['K1 (s⁻¹)'], 'o-', label='K1 (no AA)',
                linewidth=2, markersize=8, color='C0')
        ax.plot(df_no_aa['Power_numeric'], df_no_aa['K3 (s⁻¹)'], '^-', label='K3 (no AA)',
                linewidth=2, markersize=8, color='C1')
        ax2.plot(df_no_aa['Power_numeric'], df_no_aa['K4 (s⁻¹)'], 'v-', label='K4 (no AA)',
                 linewidth=2, markersize=8, color='C2')

    # Plot k values with AA (dashed lines) on primary axis
    if len(df_with_aa) > 0:
        ax.plot(df_with_aa['Power_numeric'], df_with_aa['K1 (s⁻¹)'], 'o--', label='K1 (with AA)',
                linewidth=2, markersize=8, alpha=0.7, color='C3')
        ax.plot(df_with_aa['Power_numeric'], df_with_aa['K3 (s⁻¹)'], '^--', label='K3 (with AA)',
                linewidth=2, markersize=8, alpha=0.7, color='C4')
        ax2.plot(df_with_aa['Power_numeric'], df_with_aa['K4 (s⁻¹)'], 'v--', label='K4 (with AA)',
                 linewidth=2, markersize=8, alpha=0.7, color='C5')

    ax.set_xlabel('Power (mE)', fontsize=12)
    ax.set_ylabel('K1, K3 values (s⁻¹)', fontsize=12)
    ax2.set_ylabel('K4 value (s⁻¹)', fontsize=12)
    ax.set_title('Rate Constants vs Laser Power', fontsize=14, fontweight='bold')

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(base_data_dir, 'k_values_vs_power.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()
else:
    print("No results to display")


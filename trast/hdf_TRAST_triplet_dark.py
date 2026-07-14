import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt
import matplotlib as mpl
import h5py
import os
import re

# Enable LaTeX rendering
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

mpl.rcParams['savefig.dpi'] = 300


class TRASTAnalyzer:

    def __init__(
        self,
        datasets,
        pulse_widths,
        acquisition_times,
        duty_cycles,
        fast_threshold=1e-3
    ):
        """
        Parameters
        ----------
        datasets : list of ndarray
            Photon timestamps for each pulse width.
            Each array should start near t=0.

        pulse_widths : ndarray
            Pulse widths [s]

        acquisition_times : ndarray
            Total acquisition time for each dataset [s]

        duty_cycles : ndarray
            Laser duty cycles

        fast_threshold : float
            Pulse widths below this use averaging method
        """

        self.datasets = datasets
        self.pulse_widths = np.array(pulse_widths)
        self.acquisition_times = np.array(acquisition_times)

        self.duty_cycle = np.array(duty_cycles)
        self.fast_threshold = fast_threshold

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(self):

        signals = []

        for photons, tau, tacq, duty in zip(
            self.datasets,
            self.pulse_widths,
            self.acquisition_times,
            self.duty_cycle
        ):

            if tau < self.fast_threshold:

                signal = self.fast_method(
                    photons,
                    tau,
                    tacq,
                    duty
                )

            else:

                signal = self.slow_method(
                    photons,
                    tau,
                    tacq,
                    duty
                )

            signals.append(signal)

        signals = np.array(signals)

        # normalize to the signal from the shortest pulse
        # find the index of the minimum pulse width
        min_idx = np.argmin(self.pulse_widths)
        signals /= signals[min_idx]

        return self.pulse_widths, signals

    # =========================================================
    # FAST METHOD
    # =========================================================

    def fast_method(
        self,
        photons,
        pulse_width,
        acquisition_time,
        duty
    ):

        rep_period = pulse_width / duty
        n_cycles = acquisition_time / rep_period
        total_on_time = n_cycles * pulse_width
        counts = len(photons)
        intensity = counts / total_on_time

        return intensity

    # =========================================================
    # SLOW METHOD
    # =========================================================

    def slow_method(
            self,
            photons,
            pulse_width,
            acquisition_time,
            duty
    ):

        rep_period = pulse_width / duty
        
        # Trim the acquisition time to an exact multiple of the pulse period
        n_cycles = int(acquisition_time // rep_period)
        trimmed_time = n_cycles * rep_period
        
        # Count photons within the trimmed time
        counts = np.sum(photons < trimmed_time)
        
        # Total laser-on time within the trimmed window
        total_on_time = n_cycles * pulse_width
        
        if total_on_time == 0:
            return 0.0
            
        return counts / total_on_time

    # =========================================================
    # PHASE RECOVERY
    # =========================================================

    def find_pulse_phase(
            self,
            photons,
            pulse_width,
            acquisition_time,
            duty
    ):

        rep_period = (
                pulse_width / duty
        )

        # choose adaptive binning
        bin_width = min(
            pulse_width / 20,
            rep_period / 50
        )

        bins = np.arange(
            0,
            acquisition_time + bin_width,
            bin_width
        )

        counts, edges = np.histogram(
            photons,
            bins
        )

        rates = counts / bin_width

        times = edges[:-1]

        # fold modulo repetition period
        folded_time = (
                times % rep_period
        )

        nbins = 200

        phase_bins = np.linspace(
            0,
            rep_period,
            nbins
        )

        phase_signal = np.zeros(nbins - 1)

        for i in range(nbins - 1):

            mask = (
                    (folded_time >= phase_bins[i])
                    &
                    (folded_time < phase_bins[i + 1])
            )

            if np.any(mask):
                phase_signal[i] = np.mean(
                    rates[mask]
                )

        # Smooth phase signal to get a more robust peak
        window = max(3, nbins // 20)
        if window % 2 == 0: window += 1
        phase_signal = uniform_filter1d(phase_signal, size=window, mode='wrap')

        # pulse starts where folded signal peaks
        # For a better estimate of the START of the pulse, we look for the rising edge
        # instead of the peak.
        
        threshold = (np.max(phase_signal) + np.min(phase_signal)) / 2
        
        # Find indices where signal exceeds threshold
        above_threshold = np.where(phase_signal > threshold)[0]
        
        if len(above_threshold) > 0:
            # Check for wrap around
            diffs = np.diff(above_threshold)
            if np.any(diffs > 1):
                # Wrap around detected. The "start" is the first index after the gap.
                gap_idx = np.where(diffs > 1)[0][0]
                start_idx = above_threshold[gap_idx + 1]
            else:
                start_idx = above_threshold[0]
            
            phase = phase_bins[start_idx]
        else:
            # Fallback to peak method
            peak_idx = np.argmax(phase_signal)
            phase = (phase_bins[peak_idx] - pulse_width / 2) % rep_period
            
        return phase

    # =========================================================
    # RECONSTRUCTION PLOT
    # =========================================================

    def reconstruct_trace(
        self,
        dataset_index,
        bin_width=0.01,
        smooth_bins=5
    ):

        photons = self.datasets[dataset_index]
        tau = self.pulse_widths[dataset_index]
        acquisition_time = (
            self.acquisition_times[dataset_index]
        )
        duty = self.duty_cycle[dataset_index]

        # Find the actual phase to align the laser state
        phase = self.find_pulse_phase(
            photons,
            tau,
            acquisition_time,
            duty
        )

        bins = np.arange(
            0,
            acquisition_time + bin_width,
            bin_width
        )

        counts, edges = np.histogram(
            photons,
            bins
        )

        rates = counts / bin_width
        rates = uniform_filter1d(
            rates,
            size=smooth_bins
        )

        times = edges[:-1]
        rep_period = tau / duty

        # Laser state aligned with the recovered phase
        laser_state = (
            ((times - phase) % rep_period) < tau
        ).astype(float)

        laser_state *= np.max(rates)

        plt.figure(figsize=(10, 4))
        plt.plot(
            times,
            rates,
            label="Fluorescence"
        )

        plt.plot(
            times,
            laser_state,
            alpha=0.5,
            label="Laser ON"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Counts/s")
        plt.title(
            f"Pulse width = {tau:.3e} s"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()


# =============================================================
# EXAMPLE
# =============================================================

if __name__ == "__main__":

    np.random.seed(0)
    folder_path = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/'

    file_list = [
        r'21 June 2026/TRAST 883 uW_new.h5',
        r'21 June 2026/TRAST 340 uW_new.h5',
        r'21 June 2026/TRAST 180 uW_new.h5',
    ]

    # Create a single plot for all datasets
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))

    # To store data for global fitting
    all_taus = []
    all_trast = []
    all_powers = []
    all_colors = []
    all_filenames = []

    for i, file_path in enumerate(file_list):
        color = colors[i % len(colors)]
        print("\n" + "=" * 60)
        print(f"Processing file: {file_path}")
        print("=" * 60)

        # Extract power from filename
        power_match = re.search(r'(\d+)\s*uW', os.path.basename(file_path))
        if power_match:
            power_val = float(power_match.group(1))
        else:
            power_val = None
            print(f"Could not extract power from {file_path}")

        h5data = h5py.File(folder_path + file_path, 'r')

        # Try to get pulse widths from file attributes or define them per file
        if '20 May' in file_path:
            pulse_widths = np.array([
                200e-9, 1e-6, 5e-6, 20e-6, 100e-6, 500e-6,
                2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1
            ])
            particle_nums = np.arange(1, len(pulse_widths) + 1, 1).astype(int)
        else:
            pulse_widths = np.array([
                200e-6, 500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1
            ])
            particle_nums = np.arange(2, len(pulse_widths) + 2, 1).astype(int)
            
        print(f"Particles to process: {particle_nums}")

        datasets = []
        acquisition_times = []
        duty_cycles = []
        for partnum in particle_nums:
            particle_key = f'Particle {partnum}'
            if particle_key not in h5data:
                print(f"Warning: {particle_key} not found in {file_path}")
                continue

            particle = h5data[particle_key]
            abstimes = particle['Absolute Times (ns)'][:] / 1e9
            print(f"  {particle.name}: {particle.attrs.get('Description', 'No description').splitlines()[0] if 'Description' in particle.attrs else ''}")
            datasets.append(abstimes)
            acquisition_time = abstimes[-1]
            acquisition_times.append(acquisition_time)
            duty_cycles.append(0.1)

        # Update pulse_widths if some particles were missing or specifically selected
        pulse_widths = pulse_widths[:len(datasets)]

        analyzer = TRASTAnalyzer(
            datasets=datasets,
            pulse_widths=pulse_widths,
            acquisition_times=acquisition_times,
            duty_cycles=duty_cycles,
            fast_threshold=200e-3
        )

        taus, trast = analyzer.analyze()
        
        all_taus.append(taus)
        all_trast.append(trast)
        all_powers.append(power_val)
        all_colors.append(color)
        all_filenames.append(os.path.basename(file_path))

        # ---------------------------------------------------------
        # TRAST curve
        # ---------------------------------------------------------

        plt.semilogx(
            taus,
            trast,
            'o',
            color=color,
            label=None
        )

        h5data.close()

    # ---------------------------------------------------------
    # Global Fit TRAST curves
    # ---------------------------------------------------------

    def trast_model(tau, tau_T, A, k1, k2, tau_bl, duty=0.1, include_bleaching=True):
        """
        TRAST model for triplet state + another reversible dark state + bleaching.
        tau_T: Triplet lifetime
        A: Triplet amplitude
        k1: Rate into second dark state
        k2: Rate out of second dark state
        tau_bl: Bleaching lifetime
        duty: Duty cycle (fraction)
        include_bleaching: Whether to include bleaching term
        """
        tau_D = 1 / (k1 + k2)
        A_D = k1 / (k1 + k2)

        # Correction factor for duty cycle pile-up
        # Relaxation in the dark happens at rate k2
        tau_relax = 1 / k2
        t_off = tau * (1 - duty) / duty
        correction = (1 - np.exp(-t_off / tau_relax)) / (1 - np.exp(-(tau / tau_D + t_off / tau_relax)))

        triplet = A * (1 - (1 - np.exp(-tau / tau_T)) / (tau / tau_T))
        if include_bleaching:
            bleaching = (1 - np.exp(-tau / tau_bl)) / (tau / tau_bl)
        else:
            bleaching = 1.0

        dark_state2 = A_D * (1 - (1 - np.exp(-tau / tau_D)) / (tau / tau_D) * correction)
        return (1 - triplet - dark_state2) * bleaching

    def global_model(taus_list, *params):
        """
        Global model where k2 is shared across all datasets.
        params: [k2, tau_T1, A1, k1_1, tau_bl1, tau_T2, A2, ...]
        """
        k2 = params[0]
        
        n_datasets = len(taus_list)
        n_params_per_set = 4 if use_bleaching else 3
        
        y_concatenated = []
        for i in range(n_datasets):
            idx = 1 + i * n_params_per_set
            if use_bleaching:
                tau_T, A, k1, tau_bl = params[idx:idx + n_params_per_set]
            else:
                tau_T, A, k1 = params[idx:idx + n_params_per_set]
                tau_bl = 1e9 # Placeholder
            y = trast_model(taus_list[i], tau_T, A, k1, k2, tau_bl, duty=duty_cycle, include_bleaching=use_bleaching)
            y_concatenated = np.concatenate([y_concatenated, y])
        return y_concatenated

    def local_model(taus_list, *params):
        """
        Model where both k1 and k2 are different for each dataset.
        params: [tau_T1, A1, k1_1, k2_1, tau_bl1, tau_T2, A2, ...]
        """
        n_datasets = len(taus_list)
        n_params_per_set = 5 if use_bleaching else 4
        
        y_concatenated = []
        for i in range(n_datasets):
            idx = i * n_params_per_set
            if use_bleaching:
                tau_T, A, k1, k2, tau_bl = params[idx:idx + n_params_per_set]
            else:
                tau_T, A, k1, k2 = params[idx:idx + n_params_per_set]
                tau_bl = 1e9 # Placeholder
            y = trast_model(taus_list[i], tau_T, A, k1, k2, tau_bl, duty=duty_cycle, include_bleaching=use_bleaching)
            y_concatenated = np.concatenate([y_concatenated, y])
        return y_concatenated

    def calculate_aic(y_data, y_fit, n_params):
        n = len(y_data)
        rss = np.sum((y_data - y_fit)**2)
        # Using the standard AIC formula for least squares:
        # AIC = n * ln(RSS/n) + 2k
        aic = n * np.log(rss / n) + 2 * n_params
        return aic

    print("\n" + "=" * 60)
    print("Performing Fits and Comparison")
    use_bleaching = True # Set to False to exclude bleaching
    duty_cycle = 0.1     # Set duty cycle (commonly 0.01 or 0.1)
    if not use_bleaching:
        print("Bleaching is DISABLED")
    print("=" * 60)
    
    y_data = np.concatenate(all_trast)
    n_datasets = len(all_taus)
    
    # ---------------------------------------------------------
    # 1. Global Fit (Shared k2)
    # ---------------------------------------------------------
    n_params_global_per_set = 4 if use_bleaching else 3
    n_shared_global = 1
    p0_global = [1.0] # k2 (initial guess)
    lb_global = [0.1]
    ub_global = [100]
    
    for _ in range(n_datasets):
        # tau_T, A, k1, [tau_bl]
        if use_bleaching:
            p0_global.extend([1e-6, 0.2, 1.0, 2.0])
            lb_global.extend([1e-7, 0.0, 0.0, 1.0])
            ub_global.extend([1e-3, 1.0, 10, 100.0])
        else:
            p0_global.extend([1e-6, 0.2, 1.0])
            lb_global.extend([1e-7, 0.0, 0.0])
            ub_global.extend([1e-3, 1.0, 10])

    # ---------------------------------------------------------
    # 2. Local Fit (Independent k2)
    # ---------------------------------------------------------
    n_params_local_per_set = 5 if use_bleaching else 4
    p0_local = []
    lb_local = []
    ub_local = []
    
    for _ in range(n_datasets):
        # tau_T, A, k1, k2, [tau_bl]
        if use_bleaching:
            p0_local.extend([1e-6, 0.2, 1.0, 30.0, 2.0])
            lb_local.extend([1e-7, 0.0, 0.0, 0.1, 1.0])
            ub_local.extend([1e-3, 1.0, 10, 100, 100.0])
        else:
            p0_local.extend([1e-6, 0.2, 1.0, 30.0])
            lb_local.extend([1e-7, 0.0, 0.0, 0.1])
            ub_local.extend([1e-3, 1.0, 10, 100])

    try:
        # Wrapper for global fit
        def fit_func_global(x_ignored, *params):
            return global_model(all_taus, *params)
            
        popt_g, pcov_g = curve_fit(fit_func_global, np.zeros(len(y_data)), y_data, p0=p0_global, bounds=(lb_global, ub_global))
        y_fit_g = fit_func_global(None, *popt_g.tolist())
        aic_g = calculate_aic(y_data, y_fit_g, len(popt_g))
        
        # Wrapper for local fit
        def fit_func_local(x_ignored, *params):
            return local_model(all_taus, *params)
            
        popt_l, pcov_l = curve_fit(fit_func_local, np.zeros(len(y_data)), y_data, p0=p0_local, bounds=(lb_local, ub_local))
        y_fit_l = fit_func_local(None, *popt_l.tolist())
        aic_l = calculate_aic(y_data, y_fit_l, len(popt_l))
        
        print(f"\nModel Comparison Results:")
        print(f"Global k2 model (n_params={len(popt_g)}): AIC = {aic_g:.2f}")
        print(f"Local k2 model (n_params={len(popt_l)}):  AIC = {aic_l:.2f}")

        # Always print local fit results as requested
        print(f"\n" + "-" * 30)
        print("LOCAL FIT RESULTS (Independent k2):")
        print("-" * 30)
        perr_l = np.sqrt(np.diag(pcov_l))
        for i in range(n_datasets):
            idx = i * n_params_local_per_set
            # tau_T, A, k1, k2, [tau_bl]
            if use_bleaching:
                tau_T_l, A_l, k1_l, k2_l, tau_bl_l = popt_l[idx:idx+n_params_local_per_set]
            else:
                tau_T_l, A_l, k1_l, k2_l = popt_l[idx:idx+n_params_local_per_set]
            k1_l_err, k2_l_err = perr_l[idx+2:idx+4]
            print(f"Dataset {i+1} ({all_filenames[i]}): k1 = {k1_l:.2f} ± {k1_l_err:.2f}, k2 = {k2_l:.2f} ± {k2_l_err:.2f}")
        print("-" * 30)
        
        if aic_g < aic_l:
            print(">>> Global k2 model is preferred (lower AIC).")
            # We will use global fit for plotting
            popt = popt_g
            pcov = pcov_g
            perr = np.sqrt(np.diag(pcov))
            n_shared = n_shared_global
            n_params_per_set = n_params_global_per_set
            using_global = True
        else:
            print(">>> Local k2 model is preferred (lower AIC).")
            popt = popt_l
            pcov = pcov_l
            perr = np.sqrt(np.diag(pcov))
            n_shared = 0
            n_params_per_set = n_params_local_per_set
            using_global = False

        if using_global:
            k2_fit_global = popt[0]
            print(f"\nGlobal Fit Shared Parameters:")
            print(f"Rate out of dark state (k2): {k2_fit_global:.2f} ± {perr[0]:.2f} s^-1")
        
        results = []
        for i in range(n_datasets):
            idx = n_shared + i * n_params_per_set
            dataset_popt = popt[idx:idx+n_params_per_set]
            dataset_perr = perr[idx:idx+n_params_per_set]
            
            if using_global:
                if use_bleaching:
                    tau_T_fit, A_fit, k1_fit, tau_bl_fit = dataset_popt
                    tau_bl_err = dataset_perr[3]
                else:
                    tau_T_fit, A_fit, k1_fit = dataset_popt
                    tau_bl_fit = 1e9
                    tau_bl_err = 0
                k2_fit = popt[0]
                k1_err = dataset_perr[2]
                k2_err = perr[0]
                # Covariance between k2 (index 0) and k1 (index idx+2)
                cov_k1_k2 = pcov[0, idx+2]
                var_k1 = pcov[idx+2, idx+2]
                var_k2 = pcov[0, 0]
            else:
                if use_bleaching:
                    tau_T_fit, A_fit, k1_fit, k2_fit, tau_bl_fit = dataset_popt
                    tau_bl_err = dataset_perr[4]
                else:
                    tau_T_fit, A_fit, k1_fit, k2_fit = dataset_popt
                    tau_bl_fit = 1e9
                    tau_bl_err = 0
                k1_err = dataset_perr[2]
                k2_err = dataset_perr[3]
                # Covariance between k1 (index idx+2) and k2 (index idx+3)
                cov_k1_k2 = pcov[idx+2, idx+3]
                var_k1 = pcov[idx+2, idx+2]
                var_k2 = pcov[idx+3, idx+3]
            
            print(f"\nFit Results for {all_filenames[i]} (Power: {all_powers[i]} uW):")
            print(f"Rate into dark state (k1): {k1_fit:.2f} ± {k1_err:.2f} s^-1")
            print(f"Rate out of dark state (k2): {k2_fit:.2f} ± {k2_err:.2f} s^-1")
            if use_bleaching:
                print(f"Bleaching lifetime (tau_bl): {tau_bl_fit * 1e3:.2g} ± {tau_bl_err * 1e3:.2g} ms")

            # Calculate derived parameters
            tau_D_fit = 1 / (k1_fit + k2_fit)
            A_D_fit = k1_fit / (k1_fit + k2_fit)
            k_bl = 1 / tau_bl_fit
            
            tau_S_dark = 1 / k1_fit
            tau_D_state = 1 / k2_fit
            
            # Error propagation for state lifetimes
            # tau_S_dark = 1 / k1
            tau_S_dark_err = k1_err / (k1_fit**2)
            # tau_D_state = 1 / k2
            tau_D_state_err = k2_err / (k2_fit**2)

            # Error propagation for tau_D and A_D (if needed, but we have k1, k2 now)
            # tau_D = 1 / (k1 + k2)
            grad_tau_D = np.array([-1 / (k1_fit + k2_fit)**2, -1 / (k1_fit + k2_fit)**2])
            # A_D = k1 / (k1 + k2)
            grad_A_D = np.array([k2_fit / (k1_fit + k2_fit)**2, -k1_fit / (k1_fit + k2_fit)**2])
            
            pcov_k = np.array([[var_k1, cov_k1_k2], [cov_k1_k2, var_k2]])
            tau_D_err = np.sqrt(grad_tau_D @ pcov_k @ grad_tau_D)
            A_D_err = np.sqrt(grad_A_D @ pcov_k @ grad_A_D)

            # Plot fit
            tau_fine = np.logspace(np.log10(all_taus[i].min()), np.log10(all_taus[i].max()), 100)
            fit_y = trast_model(tau_fine, tau_T_fit, A_fit, k1_fit, k2_fit, tau_bl_fit, duty=duty_cycle, include_bleaching=use_bleaching)
            
            power_val = all_powers[i]
            power_conv = power_val * 8.44444 if power_val is not None else None
            fit_label = rf"{power_conv:.1f} mmol photons m$^{{-2}}$ s$^{{-1}}$" if power_conv is not None else all_filenames[i].replace('_', r'\_')
            
            plt.semilogx(tau_fine, fit_y, '-', color=all_colors[i], label=fit_label)
            
            if power_val is not None:
                res = {
                    'power': power_val,
                    'power_conv': power_conv,
                    'k_1': k1_fit,
                    'k_2': k2_fit,
                    'k_bl': k_bl,
                    'tau_S_dark': tau_S_dark,
                    'tau_S_dark_err': tau_S_dark_err,
                    'tau_D_state': tau_D_state,
                    'tau_D_state_err': tau_D_state_err
                }
                results.append(res)

        plt.xlabel(r"Pulse width (s)")
        plt.ylabel(r"Normalized TRAST signal")
        plt.grid(False)
        plt.legend(loc='upper right', frameon=False)

        # Add inset for rates vs power
        if results:
            results_sorted = sorted(results, key=lambda x: x['power_conv'] if x['power_conv'] is not None else 0)
            powers_inset = [r['power_conv'] for r in results_sorted if r['power_conv'] is not None]
            k1_vals = [r['k_1'] for r in results_sorted if r['power_conv'] is not None]
            k2_vals = [r['k_2'] for r in results_sorted if r['power_conv'] is not None]
            kbl_vals = [r['k_bl'] for r in results_sorted if r['power_conv'] is not None]

            if powers_inset:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad=4)
                
                # Primary y-axis for k1 and kbl
                ln1 = ax_inset.plot(powers_inset, k1_vals, 'o-', label='$k_1$', markersize=4, color='C3')
                ln3 = ax_inset.plot(powers_inset, kbl_vals, '^-', label='$k_3$', markersize=4, color='C4')
                ax_inset.set_xlabel('Power (mmol photons m$^{-2}$ s$^{-1}$)', fontsize=10)
                ax_inset.set_ylabel('$k_1, k_3$ (s$^{-1}$)', fontsize=10)
                ax_inset.tick_params(axis='y', labelsize=10)
                
                # Secondary y-axis for k2
                ax_inset_k2 = ax_inset.twinx()
                ln2 = ax_inset_k2.plot(powers_inset, k2_vals, 's-', label='$k_2$', markersize=4, color='C5')
                ax_inset_k2.set_ylabel('$k_2$ (s$^{-1}$)', fontsize=10, color='C5')
                ax_inset_k2.tick_params(axis='y', labelcolor='C5', labelsize=10)
                
                # Combine legends
                lns = ln1 + ln2 + ln3
                labs = [l.get_label() for l in lns]
                ax_inset.legend(lns, labs, fontsize=8, loc='best', frameon=False)
                ax_inset.tick_params(axis='x', labelsize=10)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nGlobal fitting failed: {e}")
        import traceback
        traceback.print_exc()

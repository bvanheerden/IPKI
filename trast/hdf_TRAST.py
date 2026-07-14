import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt
import matplotlib as mpl
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
import h5py
import os

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

        duty_cycle : float
            Laser duty cycle

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

        # smooth
        # rates = uniform_filter1d(
        #     rates,
        #     size=5
        # )

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
        # r'14 May 2026/TRAST Control 1.h5',
        # r'14 May 2026/TRAST Control 2.h5',
        # r'14 May 2026/TRAST Control 3.h5',
        # r'14 May 2026/TRAST No Ox 1.h5',
        # r'14 May 2026/TRAST No Ox 2.h5',
        # r'14 May 2026/TRAST AA 2.h5',
        # r'14 May 2026/TRAST Magnet 2.h5',
        # r'14 May 2026/TRAST Magnet 3.h5',
        # r'19 May 2026/TRAST FC 2.h5',
        # r'14 May 2026/TRAST AA 2.h5',
        # r'19 June 2026/TRAST long timescale F4.h5',
        # r'19 June 2026/TRAST long timescale F1.h5',
        # r'19 June 2026/TRAST long timescale F2_reversed.h5',
        r'21 June 2026/TRAST 883 uW_new.h5',
        r'21 June 2026/TRAST 340 uW_new.h5',
        r'21 June 2026/TRAST 180 uW_new.h5',
        # Add more files here
    ]

    # Create a single plot for all datasets
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use a standard color cycle

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
        import re
        power_match = re.search(r'(\d+)\s*uW', os.path.basename(file_path))
        if power_match:
            power_val = float(power_match.group(1))
        else:
            power_val = None
            print(f"Could not extract power from {file_path}")

        h5data = h5py.File(folder_path + file_path, 'r')

        # Try to get pulse widths from file attributes or define them per file
        # For now, using the common one or defining it based on the file
        if 'TRAST Control 1.h5' in file_path:
            pulse_widths = np.array([
                200e-9, 500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6,
                500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            ])
            particle_nums = np.arange(2, len(pulse_widths) + 2, 1).astype(int)

        elif 'TRAST Control 2.h5' in file_path:
            pulse_widths = np.array([
                200e-9, 500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6,
                500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            ])
            particle_nums = np.arange(1, len(pulse_widths) + 1, 1).astype(int)

        elif 'TRAST Control 3.h5' in file_path:
            pulse_widths = np.array([
                200e-9, 500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6,
                500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1,
            ])
            particle_nums = np.arange(1, len(pulse_widths) + 1, 1).astype(int)
        elif 'TRAST No Ox 1.h5' in file_path:
            pulse_widths = np.array([
                200e-9, 500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6,
                500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            ])
            particle_nums = np.arange(2, len(pulse_widths) + 2, 1).astype(int)
        elif 'TRAST No Ox 2.h5' in file_path:
            pulse_widths = np.array([
                200e-9, 500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6,
                500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            ])
            # particle_nums = np.arange(2, len(pulse_widths) + 2, 1).astype(int)
            particle_nums = np.arange(24, len(pulse_widths) + 23, 1).astype(int)  # AA 1
        elif '20 May' in file_path:
            pulse_widths = np.array([
                200e-9, 1e-6, 5e-6, 20e-6, 100e-6, 500e-6,
                2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1
            ])
            particle_nums = np.arange(1, len(pulse_widths) + 1, 1).astype(int)
        elif 'F2' in file_path:
            # Default or other pulse widths
            pulse_widths = np.array([
                20e-3, 40e-3, 100e-3, 200e-3, 400e-3, 1, 2
            ])
            particle_nums = np.arange(1, len(pulse_widths) + 1, 1).astype(int)
        else:
            # Default or other pulse widths
            # pulse_widths = np.array([
            #     1e-3, 2e-3, 4e-3, 10e-3, 20e-3, 40e-3, 100e-3, 200e-3, 400e-3, 1, 2,
            # ])
            pulse_widths = np.array([
                200e-6, 500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1
            ])
            particle_nums = np.arange(2, len(pulse_widths) + 1, 1).astype(int)
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
            if partnum < 19:
                duty_cycles.append(0.1)
            else:
                duty_cycles.append(0.1)

        # Update pulse_widths if some particles were missing or specifically selected
        # The number of datasets collected must match the number of pulse widths used in analysis
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
            label=None  # Remove from legend
        )

        h5data.close()

    # ---------------------------------------------------------
    # Global Fit TRAST curves
    # ---------------------------------------------------------

    def trast_model(tau, tau_T, A, tau_D, A_D, tau_bl, tau_diff=0, A_diff=0, model_type='triplet_dark', include_bleaching=True):
            """
            TRAST model for triplet state + another reversible dark state + bleaching + diffusion.
            tau_T: Triplet lifetime (or the only dark state if model_type is 'one_dark')
            A: Triplet amplitude (or the only dark state amplitude)
            tau_D: Second dark state lifetime
            A_D: Second dark state amplitude
            tau_bl: Bleaching lifetime
            tau_diff: Diffusion time
            A_diff: Diffusion amplitude (fraction of molecules that can diffuse out)
            include_bleaching: Whether to include the bleaching term
            """
            triplet = A * (1 - (1 - np.exp(-tau / tau_T)) / (tau / tau_T))
            
            if include_bleaching:
                bleaching = (1 - np.exp(-tau / tau_bl)) / (tau / tau_bl)
            else:
                bleaching = 1.0
            
            diffusion = 1.0
            if tau_diff > 0:
                # 3D diffusion term averaged over the pulse: 1 / (1 + tau/tau_diff)
                # A simplified version often used in TRAST:
                diffusion = 1 - A_diff * (1 - (1 / (1 + tau / tau_diff)))

            if model_type == 'one_dark':
                return (1 - triplet) * bleaching * diffusion
            
            dark_state2 = A_D * (1 - (1 - np.exp(-tau / tau_D)) / (tau / tau_D))
            return (1 - triplet - dark_state2) * bleaching * diffusion

    def global_model(taus_list, *params):
        """
        Global model where diffusion parameters and tau_D are shared across all datasets.
        params: [tau_diff, A_diff, tau_D, tau_T1, A1, A_D1, tau_bl1, tau_T2, A2, ...]
        """
        tau_diff = params[0]
        A_diff = params[1]
        tau_D = params[2]
        
        include_bleaching = use_bleaching # Use global flag
        n_datasets = len(taus_list)
        n_params_per_set = 4 if include_bleaching else 3
        
        y_concatenated = []
        for i in range(n_datasets):
            idx = 3 + i * n_params_per_set
            if include_bleaching:
                tau_T, A, A_D, tau_bl = params[idx:idx + n_params_per_set]
            else:
                tau_T, A, A_D = params[idx:idx + n_params_per_set]
                tau_bl = 1e9 # Not used but passed
            y = trast_model(taus_list[i], tau_T, A, tau_D, A_D, tau_bl, tau_diff, A_diff, model_type='triplet_dark', include_bleaching=include_bleaching)
            y_concatenated.extend(y)
        return np.array(y_concatenated)

    def global_model_no_diff(taus_list, *params):
        """
        Global model where tau_D is shared across all datasets, NO diffusion.
        params: [tau_D, tau_T1, A1, A_D1, tau_bl1, tau_T2, A2, ...]
        """
        tau_D = params[0]
        
        include_bleaching = use_bleaching # Use global flag
        n_datasets = len(taus_list)
        n_params_per_set = 4 if include_bleaching else 3
        
        y_concatenated = []
        for i in range(n_datasets):
            idx = 1 + i * n_params_per_set
            if include_bleaching:
                tau_T, A, A_D, tau_bl = params[idx:idx + n_params_per_set]
            else:
                tau_T, A, A_D = params[idx:idx + n_params_per_set]
                tau_bl = 1e9 # Not used but passed
            # Call trast_model with tau_diff=0, A_diff=0
            y = trast_model(taus_list[i], tau_T, A, tau_D, A_D, tau_bl, tau_diff=0, A_diff=0, model_type='triplet_dark', include_bleaching=include_bleaching)
            y_concatenated.extend(y)
        return np.array(y_concatenated)

    # Choose model: 'triplet_dark', 'one_dark' or 'triplet_dark_diff'
    use_model = 'triplet_dark_diff'
    use_bleaching = True

    if use_model in ['triplet_dark_diff', 'triplet_dark']:
        print("\n" + "=" * 60)
        if use_model == 'triplet_dark_diff':
            print("Performing Global Fit (Shared Diffusion and tau_D)")
        else:
            print("Performing Global Fit (Shared tau_D, No Diffusion)")
        if not use_bleaching:
            print("Bleaching is DISABLED in the model.")
        print("=" * 60)
        
        n_datasets = len(all_taus)
        n_params_per_set = 4 if use_bleaching else 3
        
        if use_model == 'triplet_dark_diff':
            # Shared params: tau_diff, A_diff, tau_D
            # Per-dataset params: tau_T, A, A_D, tau_bl
            
            # Initial guesses: tau_diff=100ms, A_diff=0.1, tau_D=10ms
            p0 = [100e-3, 0.1, 10e-3]
            lower_bounds = [1e-3, 0, 1e-6]
            upper_bounds = [1e4, 1, 1]
            n_shared = 3
        else:
            # Shared params: tau_D
            # Per-dataset params: tau_T, A, A_D, tau_bl
            
            # Initial guesses: tau_D=10ms
            p0 = [10e-3]
            lower_bounds = [1e-6]
            upper_bounds = [1]
            n_shared = 1
        
        for _ in range(n_datasets):
            # tau_T=1us, AT=0.2, AD=0.1, tbl=2s
            if use_bleaching:
                p0.extend([1e-6, 0.2, 0.1, 2])
                lower_bounds.extend([1e-7, 0, 0, 1])
                upper_bounds.extend([1e-3, 1, 1, 100])
            else:
                p0.extend([1e-6, 0.2, 0.1])
                lower_bounds.extend([1e-7, 0, 0])
                upper_bounds.extend([1e-3, 1, 1])
            
        try:
            # Flatten trast data for curve_fit
            y_data = np.concatenate(all_trast)
            
            # Wrapper for curve_fit
            def fit_func(x_ignored, *params):
                if use_model == 'triplet_dark_diff':
                    return global_model(all_taus, *params)
                else:
                    return global_model_no_diff(all_taus, *params)
                
            popt, pcov = curve_fit(fit_func, np.zeros(len(y_data)), y_data, p0=p0, bounds=(lower_bounds, upper_bounds))
            perr = np.sqrt(np.diag(pcov))
            
            if use_model == 'triplet_dark_diff':
                tau_diff_fit = popt[0]
                A_diff_fit = popt[1]
                tau_D_fit = popt[2]
                
                print(f"\nGlobal Fit Shared Parameters:")
                print(f"Diffusion time (tau_diff): {tau_diff_fit * 1e3:.2g} ± {perr[0] * 1e3:.2g} ms")
                print(f"Diffusion amplitude (A_diff): {A_diff_fit:.2g} ± {perr[1]:.2g}")
                print(f"Dark State 2 relaxation time (tau_D): {tau_D_fit * 1e3:.2g} ± {perr[2] * 1e3:.2g} ms")
            else:
                tau_D_fit = popt[0]
                tau_diff_fit = 0
                A_diff_fit = 0
                print(f"\nGlobal Fit Shared Parameters:")
                print(f"Dark State 2 relaxation time (tau_D): {tau_D_fit * 1e3:.2g} ± {perr[0] * 1e3:.2g} ms")
            
            results = []
            for i in range(n_datasets):
                idx = n_shared + i * n_params_per_set
                dataset_popt = popt[idx:idx+n_params_per_set]
                dataset_perr = perr[idx:idx+n_params_per_set]
                
                if use_bleaching:
                    tau_T_fit, A_fit, A_D_fit, tau_bl_fit = dataset_popt
                    tau_bl_err = dataset_perr[3]
                else:
                    tau_T_fit, A_fit, A_D_fit = dataset_popt
                    tau_bl_fit = 1e9
                    tau_bl_err = 0
                
                print(f"\nFit Results for {all_filenames[i]} (Power: {all_powers[i]} uW):")
                print(f"Dark State 2 Amplitude (A_D): {A_D_fit:.2g} ± {dataset_perr[2]:.2g}")
                if use_bleaching:
                    print(f"Bleaching lifetime (tau_bl): {tau_bl_fit * 1e3:.2g} ± {tau_bl_err * 1e3:.2g} ms")

                # Calculate kinetic rates and state lifetimes for the Dark State 2
                k_SD = A_D_fit / tau_D_fit
                k_DS = (1 - A_D_fit) / tau_D_fit
                k_bl = 1 / tau_bl_fit if use_bleaching else 0
                
                tau_S_dark = 1 / k_SD
                tau_D_state = 1 / k_DS

                # Error propagation for dark state lifetimes
                # We need to consider that tau_D is now shared
                # popt index for shared tau_D is n_shared - 1 (last shared param)
                # popt index for A_D_fit is idx + 2
                grad_S_dark = np.array([1 / A_D_fit, -tau_D_fit / (A_D_fit ** 2)])
                grad_D_state = np.array([1 / (1 - A_D_fit), tau_D_fit / ((1 - A_D_fit) ** 2)])
                
                # Covariance between tau_D and A_D
                shared_tauD_idx = n_shared - 1
                cov_tauD_AD = pcov[shared_tauD_idx, idx+2]
                var_tauD = pcov[shared_tauD_idx, shared_tauD_idx]
                var_AD = pcov[idx+2, idx+2]
                
                pcov_D = np.array([[var_tauD, cov_tauD_AD], [cov_tauD_AD, var_AD]])
                tau_S_dark_err = np.sqrt(grad_S_dark @ pcov_D @ grad_S_dark)
                tau_D_state_err = np.sqrt(grad_D_state @ pcov_D @ grad_D_state)

                # Plot fit
                tau_fine = np.logspace(np.log10(all_taus[i].min()), np.log10(all_taus[i].max()), 100)
                # Params for trast_model: tau_T, A, tau_D, A_D, tau_bl, tau_diff, A_diff
                fit_y = trast_model(tau_fine, tau_T_fit, A_fit, tau_D_fit, A_D_fit, tau_bl_fit, tau_diff_fit, A_diff_fit, model_type='triplet_dark', include_bleaching=use_bleaching)
                
                power_val = all_powers[i]
                power_conv = power_val * 8.44444 if power_val is not None else None
                fit_label = rf"{power_conv:.1f} mmol photons m$^{{-2}}$ s$^{{-1}}$" if power_conv is not None else all_filenames[i].replace('_', '\_')
                
                plt.semilogx(tau_fine, fit_y, '-', color=all_colors[i], label=fit_label)
                
                if power_val is not None:
                    res_popt = [tau_T_fit, A_fit, tau_D_fit, A_D_fit]
                    if use_bleaching: res_popt.append(tau_bl_fit)
                    res_popt.extend([tau_diff_fit, A_diff_fit])

                    res_perr = [dataset_perr[0], dataset_perr[1], perr[shared_tauD_idx], dataset_perr[2]]
                    if use_bleaching: res_perr.append(tau_bl_err)
                    res_perr.extend([perr[0] if use_model=='triplet_dark_diff' else 0, 
                                    perr[1] if use_model=='triplet_dark_diff' else 0])

                    res = {
                        'power': power_val,
                        'power_conv': power_conv,
                        'popt': np.array(res_popt),
                        'perr': np.array(res_perr),
                        'model_type': use_model,
                        'use_bleaching': use_bleaching,
                        'k_1': k_SD,
                        'k_2': k_DS,
                        'k_bl': k_bl,
                        'tau_S_dark': tau_S_dark,
                        'tau_S_dark_err': tau_S_dark_err,
                        'tau_D_state': tau_D_state,
                        'tau_D_state_err': tau_D_state_err,
                        'tau_diff': tau_diff_fit,
                        'A_diff': A_diff_fit
                    }
                    results.append(res)
        except Exception as e:
            print(f"\nGlobal fitting failed: {e}")
            import traceback
            traceback.print_exc()
            results = []

    else:
        # Fallback to original individual fitting for other models if needed
        # (Though the user specifically asked for diffusion constant across all powers)
        results = []
        print("Individual fitting is currently disabled in favor of global fit.")

    plt.xlabel(r"Pulse width (s)")
    plt.ylabel(r"Normalized TRAST signal")
    # plt.title(r"TRAST Curves Comparison")
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
            # ax_inset.set_title(r'Rates vs Power', fontsize=10)

    plt.tight_layout()
    plt.show()
    #
    # # ---------------------------------------------------------
    # # Plot parameters vs power
    # # ---------------------------------------------------------
    # if results:
    #     # Sort results by power
    #     results.sort(key=lambda x: x['power'])
    #
    #     # Table for Notion
    #     print("\n" + "=" * 60)
    #     print("RESULTS SUMMARY TABLE (Markdown/Notion)")
    #     print("=" * 60)
    #
    #     if all(r['model_type'] == 'triplet_dark' for r in results):
    #         header = "| Power (mmol photons m^-2 s^-1) | tau_D (ms) | A_D | tau_bl (ms) | tau_S_dark (ms) | tau_D_state (ms) |"
    #         sep = "|---|---|---|---|---|---|"
    #         print(header)
    #         print(sep)
    #         for r in results:
    #             p = r['power_conv']
    #             popt = r['popt']
    #             tD = popt[2] * 1e3
    #             AD = popt[3]
    #             tbl = popt[4] * 1e3
    #             tS_dark = r['tau_S_dark'] * 1e3
    #             tD_state = r['tau_D_state'] * 1e3
    #             print(f"| {p:.2g} | {tD:.2g} | {AD:.2g} | {tbl:.2g} | {tS_dark:.2g} | {tD_state:.2g} |")
    #     elif all(r['model_type'] == 'one_dark' for r in results):
    #         header = "| Power (mmol photons m^-2 s^-1) | tau_D (ms) | A_D | tau_bl (ms) | tau_bright (ms) | tau_dark (ms) |"
    #         sep = "|---|---|---|---|---|---|"
    #         print(header)
    #         print(sep)
    #         for r in results:
    #             p = r['power_conv']
    #             popt = r['popt']
    #             tD = popt[0] * 1e3
    #             AD = popt[1]
    #             tbl = popt[2] * 1e3
    #             tB = r['tau_bright'] * 1e3
    #             tDk = r['tau_dark'] * 1e3
    #             print(f"| {p:.2g} | {tD:.2g} | {AD:.2g} | {tbl:.2g} | {tB:.2g} | {tDk:.2g} |")
    #     print("=" * 60)
    #
    #     powers = np.array([r['power_conv'] for r in results])
    #     popts = np.array([r['popt'] for r in results])
    #     perrs = np.array([r['perr'] for r in results])
    #     model_types = [r['model_type'] for r in results]
    #
    #     if all(m == 'triplet_dark' for m in model_types):
    #         # Combined figure: original parameters + state lifetimes
    #         fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    #         axes = axes.flatten()
    #
    #         # 1. Original Parameters
    #         param_names = ['tau_T (us)', 'A_T', 'tau_D (ms)', 'A_D', 'tau_bl (ms)']
    #         factors = [1e6, 1, 1e3, 1, 1e3]
    #
    #         for i in range(5):
    #             axes[i].errorbar(powers, popts[:, i] * factors[i], yerr=perrs[:, i] * factors[i], fmt='o-', capsize=2)
    #             axes[i].set_ylabel(param_names[i])
    #
    #         # 2. State Lifetimes
    #         tau_S_triplet = np.array([r['tau_S_triplet'] for r in results])
    #         tau_S_triplet_err = np.array([r['tau_S_triplet_err'] for r in results])
    #         tau_T_state = np.array([r['tau_T_state'] for r in results])
    #         tau_T_state_err = np.array([r['tau_T_state_err'] for r in results])
    #
    #         tau_S_dark = np.array([r['tau_S_dark'] for r in results])
    #         tau_S_dark_err = np.array([r['tau_S_dark_err'] for r in results])
    #         tau_D_state = np.array([r['tau_D_state'] for r in results])
    #         tau_D_state_err = np.array([r['tau_D_state_err'] for r in results])
    #
    #         axes[5].errorbar(powers, tau_S_triplet * 1e6, yerr=tau_S_triplet_err * 1e6, fmt='o-', capsize=2)
    #         axes[5].set_ylabel('tau_S (triplet) (us)')
    #         axes[6].errorbar(powers, tau_T_state * 1e6, yerr=tau_T_state_err * 1e6, fmt='o-', capsize=2)
    #         axes[6].set_ylabel('tau_T (triplet) (us)')
    #
    #         axes[7].errorbar(powers, tau_S_dark * 1e3, yerr=tau_S_dark_err * 1e3, fmt='o-', capsize=2)
    #         axes[7].set_ylabel('tau_S (dark) (ms)')
    #         axes[8].errorbar(powers, tau_D_state * 1e3, yerr=tau_D_state_err * 1e3, fmt='o-', capsize=2)
    #         axes[8].set_ylabel('tau_D (dark) (ms)')
    #
    #         for ax in axes:
    #             ax.set_xlabel('Power (mmol photons m^-2 s^-1)')
    #
    #         plt.suptitle('TRAST Parameters and State Lifetimes vs Laser Power')
    #         plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    #         # plt.show()
    #
    #         # 3. Requested Summary Figure: Tau_D, A_D, tau_bl, tau_S (dark)
    #         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    #         axes = axes.flatten()
    #
    #         # Index 2: tau_D, Index 3: A_D, Index 4: tau_bl
    #         requested_params = [2, 3, 4]
    #         param_names = ['tau_D (ms)', 'A_D', 'tau_bl (ms)']
    #         factors = [1e3, 1, 1e3]
    #
    #         for i, p_idx in enumerate(requested_params):
    #             axes[i].errorbar(powers, popts[:, p_idx] * factors[i], yerr=perrs[:, p_idx] * factors[i], fmt='o-', capsize=2)
    #             axes[i].set_ylabel(param_names[i])
    #             axes[i].set_xlabel('Power (mmol photons m^-2 s^-1)')
    #             axes[i].grid(True)
    #
    #         # tau_S (dark)
    #         axes[3].errorbar(powers, tau_S_dark * 1e3, yerr=tau_S_dark_err * 1e3, fmt='o-', capsize=2)
    #         axes[3].set_ylabel('tau_S (dark) (ms)')
    #         axes[3].set_xlabel('Power (mmol photons m^-2 s^-1)')
    #         axes[3].grid(True)
    #
    #         plt.suptitle('Selected Parameters vs Laser Power')
    #         plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    #         # plt.show()
    #
    #     elif all(m == 'one_dark' for m in model_types):
    #         # Combined figure: original parameters + state lifetimes
    #         fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    #         axes = axes.flatten()
    #
    #         # 1. Original Parameters
    #         param_names = ['tau_D (ms)', 'A_D', 'tau_bl (ms)']
    #         factors = [1e3, 1, 1e3]
    #
    #         for i in range(3):
    #             axes[i].errorbar(powers, popts[:, i] * factors[i], yerr=perrs[:, i] * factors[i], fmt='o-')
    #             axes[i].set_ylabel(param_names[i])
    #             axes[i].grid(True)
    #
    #         # 2. State Lifetimes
    #         tau_bright = np.array([r['tau_bright'] for r in results])
    #         tau_bright_err = np.array([r['tau_bright_err'] for r in results])
    #         tau_dark = np.array([r['tau_dark'] for r in results])
    #         tau_dark_err = np.array([r['tau_dark_err'] for r in results])
    #
    #         axes[3].errorbar(powers, tau_bright * 1e3, yerr=tau_bright_err * 1e3, fmt='o-', capsize=2)
    #         axes[3].set_ylabel('tau_bright (ms)')
    #         axes[4].errorbar(powers, tau_dark * 1e3, yerr=tau_dark_err * 1e3, fmt='o-', capsize=2)
    #         axes[4].set_ylabel('tau_dark (ms)')
    #
    #         axes[5].axis('off')
    #
    #         for i in [3, 4]:
    #             axes[i].set_xlabel('Power (mmol photons m^-2 s^-1)')
    #             axes[i].grid(True)
    #
    #         for i in range(3):
    #             axes[i].set_xlabel('Power (mmol photons m^-2 s^-1)')
    #
    #         plt.suptitle('TRAST Parameters and State Lifetimes vs Laser Power (One Dark State)')
    #         plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    #         # plt.show()
    #     else:
    #         print("Mixed models in results, skipping summary plot.")
    #
    # # ---------------------------------------------------------
    # # Visualize slow pulse reconstruction
    # # ---------------------------------------------------------
    #
    # # analyzer.reconstruct_trace(
    # #      dataset_index=-4,
    # #      bin_width=0.05
    # #  )

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt
import h5py
import os



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

    # To store results for later plotting
    results = []

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

        # ---------------------------------------------------------
        # TRAST curve
        # ---------------------------------------------------------

        plt.semilogx(
            taus,
            trast,
            'o',
            color=color,
            label=f"{os.path.basename(file_path)} (data)"
        )

        # ---------------------------------------------------------
        # Fit TRAST curve
        # ---------------------------------------------------------

        def trast_model(tau, tau_T, A, tau_D, A_D, tau_bl, model_type='triplet_dark'):
            """
            TRAST model for triplet state + another reversible dark state + bleaching.
            tau_T: Triplet lifetime (or the only dark state if model_type is 'one_dark')
            A: Triplet amplitude (or the only dark state amplitude)
            tau_D: Second dark state lifetime
            A_D: Second dark state amplitude
            tau_bl: Bleaching lifetime
            """
            triplet = A * (1 - (1 - np.exp(-tau / tau_T)) / (tau / tau_T))
            bleaching = (1 - np.exp(-tau / tau_bl)) / (tau / tau_bl)

            if model_type == 'one_dark':
                return (1 - triplet) * bleaching
            
            dark_state2 = A_D * (1 - (1 - np.exp(-tau / tau_D)) / (tau / tau_D))
            return (1 - triplet - dark_state2) * bleaching

        # Choose model: 'triplet_dark' or 'one_dark'
        use_model = 'triplet_dark'

        try:
            if use_model == 'one_dark':
                # Initial guesses: tau_D = 10ms, A_D = 0.2, tau_bl = 2s
                p0 = [10e-3, 0.2, 2]
                bounds = ([1e-6, 0, 1], [2, 1, 1000])
                
                # Wrapper to pass model_type to curve_fit if needed, 
                # but curve_fit expects params as args.
                popt, pcov = curve_fit(lambda t, tD, AD, tbl: trast_model(t, tD, AD, 0, 0, tbl, model_type='one_dark'), 
                                       taus, trast, p0=p0, bounds=bounds)
                
                tau_D_fit, A_D_fit, tau_bl_fit = popt
                perr = np.sqrt(np.diag(pcov))

                # Calculate kinetic rates and state lifetimes
                # k_on: Bright -> Dark, k_off: Dark -> Bright
                k_on = A_D_fit / tau_D_fit
                k_off = (1 - A_D_fit) / tau_D_fit
                tau_bright = 1 / k_on
                tau_dark = 1 / k_off

                # Error propagation for lifetimes using covariance matrix
                # f(tau, A) = tau / A => df/dtau = 1/A, df/dA = -tau/A^2
                # f(tau, A) = tau / (1-A) => df/dtau = 1/(1-A), df/dA = tau/(1-A)^2
                grad_bright = np.array([1 / A_D_fit, -tau_D_fit / (A_D_fit ** 2)])
                grad_dark = np.array([1 / (1 - A_D_fit), tau_D_fit / ((1 - A_D_fit) ** 2)])
                
                pcov_sub = pcov[:2, :2]
                tau_bright_err = np.sqrt(grad_bright @ pcov_sub @ grad_bright)
                tau_dark_err = np.sqrt(grad_dark @ pcov_sub @ grad_dark)

                print(f"\nFit Results (One Dark State) for {os.path.basename(file_path)}:")
                print(f"Relaxation time (tau_D): {tau_D_fit * 1e3:.2g} ± {perr[0] * 1e3:.2g} ms")
                print(f"Dark State Amplitude (A_D): {A_D_fit:.2g} ± {perr[1]:.2g}")
                print(f"Bleaching lifetime (tau_bl): {tau_bl_fit * 1e3:.2g} ± {perr[2] * 1e3:.2g} ms")
                print(f"Kinetic rates: k_on = {k_on:.2g} s^-1, k_off = {k_off:.2g} s^-1")
                print(f"State lifetimes: tau_bright = {tau_bright * 1e3:.2g} ± {tau_bright_err * 1e3:.2g} ms, "
                      f"tau_dark = {tau_dark * 1e3:.2g} ± {tau_dark_err * 1e3:.2g} ms")

                fit_curve = lambda t, *p: trast_model(t, p[0], p[1], 0, 0, p[2], model_type='one_dark')
            else:
                # Initial guesses: tau_T = 1ms, A = 0.2, tau_D = 10ms, A_D = 0.1, tau_bl = 100ms
                p0 = [1e-6, 0.2, 10e-3, 0.1, 2]
                bounds = ([1e-7, 0, 1e-6, 0, 1], [1e-3, 1, 1, 1, 100])
                popt, pcov = curve_fit(lambda t, tT, AT, tD, AD, tbl: trast_model(t, tT, AT, tD, AD, tbl, model_type='triplet_dark'), 
                                       taus, trast, p0=p0, bounds=bounds)

                tau_T_fit, A_fit, tau_D_fit, A_D_fit, tau_bl_fit = popt
                perr = np.sqrt(np.diag(pcov))

                # Calculate kinetic rates and state lifetimes
                k_ST = A_fit / tau_T_fit
                k_TS = (1 - A_fit) / tau_T_fit
                k_SD = A_D_fit / tau_D_fit
                k_DS = (1 - A_D_fit) / tau_D_fit
                
                tau_S_triplet = 1 / k_ST
                tau_T_state = 1 / k_TS
                tau_S_dark = 1 / k_SD
                tau_D_state = 1 / k_DS

                # Error propagation for triplet lifetimes
                grad_S_triplet = np.array([1 / A_fit, -tau_T_fit / (A_fit ** 2)])
                grad_T_state = np.array([1 / (1 - A_fit), tau_T_fit / ((1 - A_fit) ** 2)])
                pcov_T = pcov[0:2, 0:2]
                tau_S_triplet_err = np.sqrt(grad_S_triplet @ pcov_T @ grad_S_triplet)
                tau_T_state_err = np.sqrt(grad_T_state @ pcov_T @ grad_T_state)

                # Error propagation for dark state lifetimes
                grad_S_dark = np.array([1 / A_D_fit, -tau_D_fit / (A_D_fit ** 2)])
                grad_D_state = np.array([1 / (1 - A_D_fit), tau_D_fit / ((1 - A_D_fit) ** 2)])
                pcov_D = pcov[2:4, 2:4]
                tau_S_dark_err = np.sqrt(grad_S_dark @ pcov_D @ grad_S_dark)
                tau_D_state_err = np.sqrt(grad_D_state @ pcov_D @ grad_D_state)

                print(f"\nFit Results (Triplet + Dark State) for {os.path.basename(file_path)}:")
                # print(f"Triplet relaxation time (tau_T): {tau_T_fit * 1e6:.2g} ± {perr[0] * 1e6:.2g} µs")
                # print(f"Triplet Amplitude (A): {A_fit:.2g} ± {perr[1]:.2g}")
                print(f"Dark State 2 relaxation time (tau_D): {tau_D_fit * 1e3:.2g} ± {perr[2] * 1e3:.2g} ms")
                print(f"Dark State 2 Amplitude (A_D): {A_D_fit:.2g} ± {perr[3]:.2g}")
                print(f"Bleaching lifetime (tau_bl): {tau_bl_fit * 1e3:.2g} ± {perr[4] * 1e3:.2g} ms")
                # print(f"Triplet rates: k_ST = {k_ST:.2g} s^-1, k_TS = {k_TS:.2g} s^-1")
                print(f"Dark state rates: k_SD = {k_SD:.2g} s^-1, k_DS = {k_DS:.2g} s^-1")
                # print(f"State lifetimes (Triplet): tau_S = {tau_S_triplet * 1e6:.2g} ± {tau_S_triplet_err * 1e6:.2g} µs, "
                #      f"tau_T = {tau_T_state * 1e6:.2g} ± {tau_T_state_err * 1e6:.2g} µs")
                print(f"State lifetimes (Dark): tau_S = {tau_S_dark * 1e3:.2g} ± {tau_S_dark_err * 1e3:.2g} ms, "
                      f"tau_D = {tau_D_state * 1e3:.2g} ± {tau_D_state_err * 1e3:.2g} ms")
                
                fit_curve = lambda t, *p: trast_model(t, p[0], p[1], p[2], p[3], p[4], model_type='triplet_dark')

            # Plot fit
            tau_fine = np.logspace(np.log10(taus.min()), np.log10(taus.max()), 100)
            plt.semilogx(tau_fine, fit_curve(tau_fine, *popt), '-', color=color,
                         label=f'Fit {os.path.basename(file_path)}')

            if power_val is not None:
                res = {
                    'power': power_val,
                    'popt': popt,
                    'perr': perr,
                    'model_type': use_model
                }
                if use_model == 'one_dark':
                    res.update({
                        'tau_bright': tau_bright,
                        'tau_bright_err': tau_bright_err,
                        'tau_dark': tau_dark,
                        'tau_dark_err': tau_dark_err
                    })
                else:
                    res.update({
                        'tau_S_triplet': tau_S_triplet,
                        'tau_S_triplet_err': tau_S_triplet_err,
                        'tau_T_state': tau_T_state,
                        'tau_T_state_err': tau_T_state_err,
                        'tau_S_dark': tau_S_dark,
                        'tau_S_dark_err': tau_S_dark_err,
                        'tau_D_state': tau_D_state,
                        'tau_D_state_err': tau_D_state_err
                    })
                results.append(res)

        except Exception as e:
            print(f"\nFitting failed for {file_path}: {e}")

        h5data.close()

    plt.xlabel("Pulse width (s)")
    plt.ylabel("Normalized TRAST signal")
    plt.title("TRAST Curves Comparison")
    plt.grid(False)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Plot parameters vs power
    # ---------------------------------------------------------
    if results:
        # Sort results by power
        results.sort(key=lambda x: x['power'])
        
        # Table for Notion
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY TABLE (Markdown/Notion)")
        print("=" * 60)
        
        if all(r['model_type'] == 'triplet_dark' for r in results):
            header = "| Power (uW) | tau_D (ms) | A_D | tau_bl (ms) | tau_S_dark (ms) | tau_D_state (ms) |"
            sep = "|---|---|---|---|---|---|"
            print(header)
            print(sep)
            for r in results:
                p = r['power']
                popt = r['popt']
                tD = popt[2] * 1e3
                AD = popt[3]
                tbl = popt[4] * 1e3
                tS_dark = r['tau_S_dark'] * 1e3
                tD_state = r['tau_D_state'] * 1e3
                print(f"| {p:.2g} | {tD:.2g} | {AD:.2g} | {tbl:.2g} | {tS_dark:.2g} | {tD_state:.2g} |")
        elif all(r['model_type'] == 'one_dark' for r in results):
            header = "| Power (uW) | tau_D (ms) | A_D | tau_bl (ms) | tau_bright (ms) | tau_dark (ms) |"
            sep = "|---|---|---|---|---|---|"
            print(header)
            print(sep)
            for r in results:
                p = r['power']
                popt = r['popt']
                tD = popt[0] * 1e3
                AD = popt[1]
                tbl = popt[2] * 1e3
                tB = r['tau_bright'] * 1e3
                tDk = r['tau_dark'] * 1e3
                print(f"| {p:.2g} | {tD:.2g} | {AD:.2g} | {tbl:.2g} | {tB:.2g} | {tDk:.2g} |")
        print("=" * 60)

        powers = np.array([r['power'] for r in results])
        popts = np.array([r['popt'] for r in results])
        perrs = np.array([r['perr'] for r in results])
        model_types = [r['model_type'] for r in results]

        if all(m == 'triplet_dark' for m in model_types):
            # Combined figure: original parameters + state lifetimes
            fig, axes = plt.subplots(3, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # 1. Original Parameters
            param_names = ['tau_T (us)', 'A_T', 'tau_D (ms)', 'A_D', 'tau_bl (ms)']
            factors = [1e6, 1, 1e3, 1, 1e3]
            
            for i in range(5):
                axes[i].errorbar(powers, popts[:, i] * factors[i], yerr=perrs[:, i] * factors[i], fmt='o-', capsize=2)
                axes[i].set_ylabel(param_names[i])
            
            # 2. State Lifetimes
            tau_S_triplet = np.array([r['tau_S_triplet'] for r in results])
            tau_S_triplet_err = np.array([r['tau_S_triplet_err'] for r in results])
            tau_T_state = np.array([r['tau_T_state'] for r in results])
            tau_T_state_err = np.array([r['tau_T_state_err'] for r in results])
            
            tau_S_dark = np.array([r['tau_S_dark'] for r in results])
            tau_S_dark_err = np.array([r['tau_S_dark_err'] for r in results])
            tau_D_state = np.array([r['tau_D_state'] for r in results])
            tau_D_state_err = np.array([r['tau_D_state_err'] for r in results])

            axes[5].errorbar(powers, tau_S_triplet * 1e6, yerr=tau_S_triplet_err * 1e6, fmt='o-', capsize=2)
            axes[5].set_ylabel('tau_S (triplet) (us)')
            axes[6].errorbar(powers, tau_T_state * 1e6, yerr=tau_T_state_err * 1e6, fmt='o-', capsize=2)
            axes[6].set_ylabel('tau_T (triplet) (us)')
            
            axes[7].errorbar(powers, tau_S_dark * 1e3, yerr=tau_S_dark_err * 1e3, fmt='o-', capsize=2)
            axes[7].set_ylabel('tau_S (dark) (ms)')
            axes[8].errorbar(powers, tau_D_state * 1e3, yerr=tau_D_state_err * 1e3, fmt='o-', capsize=2)
            axes[8].set_ylabel('tau_D (dark) (ms)')

            for ax in axes:
                ax.set_xlabel('Power (uW)')
            
            plt.suptitle('TRAST Parameters and State Lifetimes vs Laser Power')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()

            # 3. Requested Summary Figure: Tau_D, A_D, tau_bl, tau_S (dark)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # Index 2: tau_D, Index 3: A_D, Index 4: tau_bl
            requested_params = [2, 3, 4]
            param_names = ['tau_D (ms)', 'A_D', 'tau_bl (ms)']
            factors = [1e3, 1, 1e3]
            
            for i, p_idx in enumerate(requested_params):
                axes[i].errorbar(powers, popts[:, p_idx] * factors[i], yerr=perrs[:, p_idx] * factors[i], fmt='o-', capsize=2)
                axes[i].set_ylabel(param_names[i])
                axes[i].set_xlabel('Power (uW)')
                axes[i].grid(True)
            
            # tau_S (dark)
            axes[3].errorbar(powers, tau_S_dark * 1e3, yerr=tau_S_dark_err * 1e3, fmt='o-', capsize=2)
            axes[3].set_ylabel('tau_S (dark) (ms)')
            axes[3].set_xlabel('Power (uW)')
            axes[3].grid(True)
            
            plt.suptitle('Selected Parameters vs Laser Power')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()

        elif all(m == 'one_dark' for m in model_types):
            # Combined figure: original parameters + state lifetimes
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            # 1. Original Parameters
            param_names = ['tau_D (ms)', 'A_D', 'tau_bl (ms)']
            factors = [1e3, 1, 1e3]
            
            for i in range(3):
                axes[i].errorbar(powers, popts[:, i] * factors[i], yerr=perrs[:, i] * factors[i], fmt='o-')
                axes[i].set_ylabel(param_names[i])
                axes[i].grid(True)
            
            # 2. State Lifetimes
            tau_bright = np.array([r['tau_bright'] for r in results])
            tau_bright_err = np.array([r['tau_bright_err'] for r in results])
            tau_dark = np.array([r['tau_dark'] for r in results])
            tau_dark_err = np.array([r['tau_dark_err'] for r in results])

            axes[3].errorbar(powers, tau_bright * 1e3, yerr=tau_bright_err * 1e3, fmt='o-', capsize=2)
            axes[3].set_ylabel('tau_bright (ms)')
            axes[4].errorbar(powers, tau_dark * 1e3, yerr=tau_dark_err * 1e3, fmt='o-', capsize=2)
            axes[4].set_ylabel('tau_dark (ms)')
            
            axes[5].axis('off')

            for i in [3, 4]:
                axes[i].set_xlabel('Power (uW)')
                axes[i].grid(True)
            
            for i in range(3):
                axes[i].set_xlabel('Power (uW)')

            plt.suptitle('TRAST Parameters and State Lifetimes vs Laser Power (One Dark State)')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()
        else:
            print("Mixed models in results, skipping summary plot.")

    # ---------------------------------------------------------
    # Visualize slow pulse reconstruction
    # ---------------------------------------------------------

    # analyzer.reconstruct_trace(
    #      dataset_index=-4,
    #      bin_width=0.05
    #  )

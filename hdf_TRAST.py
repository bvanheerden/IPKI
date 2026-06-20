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

        # normalize
        signals /= signals[0]

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

        rep_period = (
                pulse_width / duty
        )

        phase = self.find_pulse_phase(
            photons,
            pulse_width,
            acquisition_time,
            duty
        )

        pulse_starts = np.arange(
            phase,
            acquisition_time,
            rep_period
        )
        print(pulse_starts)

        intensities = []

        for start in pulse_starts:

            stop = start + pulse_width
            if stop > acquisition_time:
                continue
            n = np.sum(
                (photons >= start)
                & (photons < stop)
            )
            intensities.append(
                n / pulse_width
            )

        if len(intensities) == 0:
            return np.nan

        return np.mean(intensities)

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

        # pulse starts where folded signal peaks
        peak_idx = np.argmax(
            phase_signal
        )

        phase = phase_bins[peak_idx]

        return phase

    # =========================================================
    # RECONSTRUCTION PLOT
    # =========================================================

    def reconstruct_trace(
        self,
        dataset_index,
        bin_width=0.01,
        smooth_bins=5,
        duty=0.1
    ):

        photons = self.datasets[dataset_index]
        tau = self.pulse_widths[dataset_index]
        acquisition_time = (
            self.acquisition_times[dataset_index]
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

        laser_state = (
            (times % rep_period) < tau
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
        r'19 June 2026/TRAST long timescale F4.h5',
        # Add more files here
    ]

    # Create a single plot for all datasets
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    colors = plt.cm.tab10.colors  # Use a standard color cycle

    for i, file_path in enumerate(file_list):
        color = colors[i % len(colors)]
        print("\n" + "=" * 60)
        print(f"Processing file: {file_path}")
        print("=" * 60)

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
        else:
            # Default or other pulse widths
            pulse_widths = np.array([
                1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1,
            ])
            particle_nums = np.arange(1, len(pulse_widths) + 1, 1).astype(int)
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
            if partnum < 23:
                duty_cycles.append(0.2)
            elif partnum == 23:
                duty_cycles.append(0.2)
            else:
                duty_cycles.append(0.5)

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

        def trast_model(tau, tau_T, A, tau_D, A_D, tau_bl):
            """
            TRAST model for triplet state + another reversible dark state + bleaching.
            tau_T: Triplet lifetime
            A: Triplet amplitude
            tau_D: Second dark state lifetime
            A_D: Second dark state amplitude
            tau_bl: Bleaching lifetime
            """
            triplet = A * (1 - (1 - np.exp(-tau / tau_T)) / (tau / tau_T))
            dark_state2 = A_D * (1 - (1 - np.exp(-tau / tau_D)) / (tau / tau_D))
            bleaching = (1 - np.exp(-tau / tau_bl)) / (tau / tau_bl)

            return (1 - triplet - dark_state2) * bleaching

        try:
            # Initial guesses: tau_T = 1ms, A = 0.2, tau_D = 10ms, A_D = 0.1, tau_bl = 100ms
            p0 = [1e-3, 0.2, 10e-3, 0.1, 2]
            bounds = ([1e-7, 0, 1e-6, 0, 1], [1e-1, 1, 1, 1, 100])
            popt, pcov = curve_fit(trast_model, taus, trast, p0=p0, bounds=bounds)

            tau_T_fit, A_fit, tau_D_fit, A_D_fit, tau_bl_fit = popt
            perr = np.sqrt(np.diag(pcov))

            print(f"\nFit Results for {os.path.basename(file_path)}:")
            print(f"Triplet lifetime (tau_T): {tau_T_fit * 1e6:.2f} ± {perr[0] * 1e6:.2f} µs")
            print(f"Triplet Amplitude (A): {A_fit:.3f} ± {perr[1]:.3f}")
            print(f"Dark State 2 lifetime (tau_D): {tau_D_fit * 1e3:.2f} ± {perr[2] * 1e3:.2f} ms")
            print(f"Dark State 2 Amplitude (A_D): {A_D_fit:.3f} ± {perr[3]:.3f}")
            print(f"Bleaching lifetime (tau_bl): {tau_bl_fit * 1e3:.2f} ± {perr[4] * 1e3:.2f} ms")

            # Plot fit
            tau_fine = np.logspace(np.log10(taus.min()), np.log10(taus.max()), 100)
            plt.semilogx(tau_fine, trast_model(tau_fine, *popt), '-', color=color,
                         label=f'Fit {os.path.basename(file_path)}')

        except Exception as e:
            print(f"\nFitting failed for {file_path}: {e}")

        h5data.close()

    plt.xlabel("Pulse width (s)")
    plt.ylabel("Normalized TRAST signal")
    plt.title("TRAST Curves Comparison")
    plt.grid(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Visualize slow pulse reconstruction
    # ---------------------------------------------------------

    analyzer.reconstruct_trace(
         dataset_index=-1,
         bin_width=0.05,
         duty=0.2
     )

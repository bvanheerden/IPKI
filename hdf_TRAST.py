import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt
import h5py



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

    pulse_widths = np.array([
        200e-9,
        500e-9,
        1e-6,
        2e-6,
        5e-6,
        10e-6,
        20e-6,
        50e-6,
        100e-6,
        200e-6,
        500e-6,
        1e-3,
        2e-3,
        5e-3,
        10e-3,
        20e-3,
        50e-3,
        100e-3,
        200e-3,
        500e-3,
        1,
        2,
        5
    ])

    particle_nums = np.arange(1, 25, 1).astype(int)
    particle_nums = np.delete(particle_nums, 20)
    print(particle_nums)

    h5data = h5py.File(r'E:\SMS\Measurements\Bertus\LHCII\PAM\Modulate and gate\2026\11 May 2026\TRAST.h5', 'r')

    datasets = []
    acquisition_times = []
    duty_cycles = []
    for partnum in particle_nums:
        particle = h5data[f'Particle {partnum}']
        abstimes = particle['Absolute Times (ns)'][:] / 1e9
        print(particle.name)
        print(particle.attrs['Description'].splitlines()[1])
        datasets.append(abstimes)
        acquisition_time = abstimes[-1]
        print(acquisition_time)
        acquisition_times.append(acquisition_time)
        if partnum < 23:
            duty_cycles.append(0.1)
        elif partnum == 23:
            duty_cycles.append(0.2)
        else:
            duty_cycles.append(0.5)


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

    plt.figure(figsize=(6, 5))

    plt.semilogx(
        taus,
        trast,
        'o-'
    )

    plt.xlabel("Pulse width (s)")
    plt.ylabel("Normalized TRAST signal")
    plt.title("TRAST Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Visualize slow pulse reconstruction
    # ---------------------------------------------------------

    analyzer.reconstruct_trace(
        dataset_index=-1,
        bin_width=0.05,
        duty=0.5
    )
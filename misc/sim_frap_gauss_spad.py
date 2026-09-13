import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import utils

utils.setup_plotting()


# ============================================================
# Experimental data loading (`misc/chla.py` logic)
# ============================================================

data_dir = r'C:\\Users\\bertu\\Desktop\\Chl less glycerol'
startind = 0

def load_experimental_trace(partnum=0):
    try:
        h5_path = os.path.join(data_dir, f'measurement {partnum}.h5')
        if not os.path.exists(h5_path):
            print(f"Warning: Experimental data file not found at {h5_path}")
            return None, None
        
        dataset = h5py.File(h5_path, 'r')
        abstimes = dataset['timestamps'][:] * 50

        difftime = np.diff(abstimes)
        boundary_photons = np.where(difftime > 20e6)[0]
        boundary_times = abstimes[boundary_photons]
        boundary_times_start = abstimes[boundary_photons + 1]
        boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
        ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
        timestep = np.mean(np.diff(boundary_times_start) / 1e9)  # timestep in s

        pulsephotons = np.diff(boundary_photons)
        norm_pulsephotons = pulsephotons / ms_pulse[1:]
        norm_pulsephotons /= np.mean(norm_pulsephotons[startind])
        norm_pulsephotons = norm_pulsephotons[np.isfinite(norm_pulsephotons)]
        
        datapoints = len(norm_pulsephotons) + 1
        endpoint = datapoints * timestep
        t_exp_full = np.linspace(0, endpoint, datapoints)
        t_exp = t_exp_full[:-1]
        
        return norm_pulsephotons, t_exp
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None, None

exp_y, exp_t = load_experimental_trace(0)


# ============================================================
# FRAP parameters
# ============================================================

D = 30.0              # diffusion coefficient, µm^2/s

w = 75.0              # Gaussian 1/e^2 radius, µm

F0 = 1.0              # fluorescence before bleaching
Fb = 0.4             # fluorescence at centre immediately after bleach

total_time = 26       # simulation time, s
n_points = 2000


# ============================================================
# Time axis
# ============================================================

t = np.linspace(0, total_time, n_points)


# ============================================================
# Analytical Gaussian-bleach / point-detection model
# ============================================================

# F = F0 - (F0 - Fb) / (
#     1 + 8 * D * t / w**2
# )

# Top hat beam
F = Fb + (F0 - Fb) * np.exp(- w**2 / (4 * D * t))


# ============================================================
# Add optional measurement noise
# ============================================================

noise_sigma = 0.005

F_noisy = F + np.random.normal(
    0,
    noise_sigma,
    size=len(F)
)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(90/25.4, 50/25.4))


if exp_t is not None and exp_y is not None:
    plt.plot(
        exp_t,
        exp_y,
        '.',
        color='gray',
        linewidth=1,
        alpha=0.6,
        label="Experimental data"
    )

plt.plot(
    t,
    F,
    linewidth=2.5,
    label="Analytical model"
)

plt.xlabel("Time (s)")
plt.ylabel("Norm. Fluorescence")

plt.xlim(0, 26)
plt.legend(frameon=False)
# plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# ============================================================
# Characteristic diffusion time
# ============================================================

tau = w**2 / (8 * D)

print(f"D = {D:.2f} µm²/s")
print(f"Gaussian radius w = {w:.1f} µm")
print(f"Characteristic recovery time = {tau:.2f} s")
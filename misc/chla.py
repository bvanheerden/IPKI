import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
import utils

utils.setup_plotting()

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Chl less glycerol'

# timestep = 0.05  # time step of intensity trace in seconds

startind = 0


def onetrace(partnum):
    dataset = h5py.File(os.path.join(data_dir, f'measurement {partnum}.h5'), 'r')
    abstimes = dataset['timestamps'][:] * 50

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
    timestep = np.mean(np.diff(boundary_times_start) / 1e9)  # timestep in s
    # print(ms_pulse)

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    norm_pulsephotons /= np.mean(norm_pulsephotons[startind])
    norm_pulsephotons = norm_pulsephotons[np.isfinite(norm_pulsephotons)]
    return norm_pulsephotons[:], timestep


def avtrace(partnums):
    results = [onetrace(partnum) for partnum in partnums]
    traces = [r[0] for r in results]
    timesteps = [r[1] for r in results]

    minlength = np.min([len(t) for t in traces])
    traces = [t[:minlength] for t in traces]

    avg_trace = np.mean(traces, axis=0)
    avg_timestep = np.mean(timesteps, axis=0)
    return avg_trace, avg_timestep


def fittrace(partnums):

    norm_pulsephotons, timestep = avtrace(partnums)[startind:]
    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    return norm_pulsephotons, t[:-1]


norm_pulsephotons, t = fittrace([0])

plt.figure(figsize = (70/25.4, 50/25.4))
plt.plot(t, norm_pulsephotons[:], '-', color='C0', lw=1.5)

plt.xlabel('Time (s)')
plt.ylabel('Normalized fluorescence')
plt.tight_layout()
plt.xlim(0, 25)
plt.show()


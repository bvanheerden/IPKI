import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
import utils

utils.setup_plotting()

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/27 May 2026'

timestep = 0.05  # time step of intensity trace in seconds

dataset = h5py.File(os.path.join(data_dir, 'Chl a PAM.h5'), 'r')
startind = 0


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    print(particle_.attrs['Description'])

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
    # print(ms_pulse)

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    norm_pulsephotons /= np.mean(norm_pulsephotons[startind])
    norm_pulsephotons = norm_pulsephotons[np.isfinite(norm_pulsephotons)]
    return norm_pulsephotons[:]


def avtrace(dataset, partnums):
    partnums_ = [onetrace(dataset, partnum) for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    return np.mean(partnums_, axis=0)


def fittrace(dataset, partnums):

    norm_pulsephotons = avtrace(dataset, partnums)[startind:]
    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    return norm_pulsephotons, t[:-1]


norm_pulsephotons, t= fittrace(dataset, [11, 12, 13, 14, 15, 16])

plt.figure(figsize = (70/25.4, 50/25.4))
plt.plot(t, norm_pulsephotons[:], '-', color='C0', lw=1.5)

plt.xlabel('Time (s)')
plt.ylabel('Normalized fluorescence')
plt.tight_layout()
plt.xlim(0, 30)
plt.show()


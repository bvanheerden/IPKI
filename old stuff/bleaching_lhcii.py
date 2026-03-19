import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.signal import savgol_filter

# data_dir = 'E:\SMS\Measurements\Bertus\LHCII\PAM\Modulate and gate'
data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/Modulate and gate'

dataset = h5py.File(os.path.join(data_dir, '7 March 2025', 'Bleach kinetic new.h5'), 'r')


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    # print(particle_.attrs['Description'])

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 10e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    # print(ms_pulse)
    # print(pulsephotons[0])
    norm_pulsephotons = savgol_filter(norm_pulsephotons, 100, 3)  # window size 51, polynomial order 3
    norm_pulsephotons /= np.mean(norm_pulsephotons[:6])
    return norm_pulsephotons[1:]


def avtrace(dataset, partnums):
    partnums_ = [onetrace(dataset, partnum) for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = np.array([partnum[:minlength] for partnum in partnums_])
    return np.mean(partnums_, axis=0)


photons_20 = avtrace(dataset, [1, 2])
index_20 = np.abs(photons_20 - 0.92).argmin()
plt.plot(photons_20[:])
photons_45 = avtrace(dataset, [3, 4])
index_45 = np.abs(photons_45 - 0.85).argmin()
plt.plot(photons_45[:])
photons_100 = avtrace(dataset, [5, 6])
index_100 = np.abs(photons_100 - 0.85).argmin()
plt.plot(photons_100[:])
photons_225 = avtrace(dataset, [7, 8])
index_225 = np.abs(photons_225 - 0.85).argmin()
plt.plot(photons_225[:])
photons_506 = avtrace(dataset, [9, 10])
index_506 = np.abs(photons_506 - 0.85).argmin()
plt.plot(photons_506[:])
photons_1140 = avtrace(dataset, [10, 11])
index_1140 = np.abs(photons_1140 - 0.85).argmin()
plt.plot(photons_1140[:])
photons_1700 = avtrace(dataset, [12, 13])
index_1700 = np.abs(photons_1700 - 0.85).argmin()
plt.plot(photons_1700[:])

indices = [index_45, index_100, index_225, index_506, index_1140, index_1700]
powers = [45, 100, 225, 506, 1140, 1700]

plt.grid()
plt.xlabel('Pulse time (125 ms)')
plt.ylabel('Normalized photon count')
# plt.xlim(0, 140)
plt.show()

plt.figure()

plt.plot(powers, [1/index for index in indices])
plt.grid()
plt.show()
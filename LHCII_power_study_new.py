import numpy as np
import h5py
from matplotlib import pyplot as plt
import os

data_dir = 'E:\SMS\Measurements\Bertus\LHCII\PAM\Modulate and gate'

dataset_150 = h5py.File(os.path.join(data_dir, '6 March 2025', '150 uW.h5'), 'r')
dataset = h5py.File(os.path.join(data_dir, '6 March 2025', '225 uW.h5'), 'r')
dataset_338 = h5py.File(os.path.join(data_dir, '7 March 2025', '338 uW.h5'), 'r')
dataset_506 = h5py.File(os.path.join(data_dir, '7 March 2025', '506 uW.h5'), 'r')
dataset_760 = h5py.File(os.path.join(data_dir, '7 March 2025', '760 uW.h5'), 'r')
dataset_1140 = h5py.File(os.path.join(data_dir, '7 March 2025', '1140 uW.h5'), 'r')
dataset_1700 = h5py.File(os.path.join(data_dir, '7 March 2025', '1700 uW.h5'), 'r')


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
    norm_pulsephotons /= np.mean(norm_pulsephotons[:6])
    return norm_pulsephotons[1:]


def avtrace(dataset, partnums):
    partnums_ = [onetrace(dataset, partnum) for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    return np.mean(partnums_, axis=0)


norm_pulsephotons_150 = avtrace(dataset_150, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
norm_pulsephotons = avtrace(dataset, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
norm_pulsephotons_338 = avtrace(dataset_338, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
norm_pulsephotons_506 = avtrace(dataset_506, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
norm_pulsephotons_760 = avtrace(dataset_760, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
norm_pulsephotons_1140 = avtrace(dataset_1140, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])#, 11, 12, 13, 14, 15])
norm_pulsephotons_1700 = avtrace(dataset_1700, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])#, 11, 12, 13, 14, 15])

plt.plot(norm_pulsephotons[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(norm_pulsephotons_150[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(norm_pulsephotons_338[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(norm_pulsephotons_506[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(norm_pulsephotons_760[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(norm_pulsephotons_1140[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(norm_pulsephotons_1700[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')

plt.grid()
plt.xlabel('Pulse time (125 ms)')
plt.ylabel('Normalized photon count')
# plt.xlim(0, 140)
plt.show()
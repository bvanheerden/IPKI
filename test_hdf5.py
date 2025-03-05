import numpy as np
import h5py
from matplotlib import pyplot as plt

# dataset = h5py.File('blinking/Test 3000 mE 2.h5', 'r')
# dataset = h5py.File('blinking/Power study.h5', 'r')
# dataset = h5py.File('blinking/700 mE.h5', 'r')
# dataset_thyl = h5py.File('blinking/12 Feb 2025/Thylakoid intensity study new.h5', 'r')
# dataset = h5py.File('blinking/11 Feb 2025/Ascorbic.h5', 'r')
# dataset_int = h5py.File('blinking/10 Feb 2025/Chemical study.h5', 'r')
# dataset_830 = h5py.File('blinking/11 Feb 2025/830 nm laser.h5', 'r')
dataset = h5py.File('blinking/4 March 2025/test.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/Ascorbic.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/Low pH.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/Azide.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/3000 mE.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/2380 mE.h5', 'r')


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 10e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    norm_pulsephotons /= np.mean(norm_pulsephotons[:6])
    return norm_pulsephotons[1:300]


def avtrace(dataset, partnums):
    return np.mean([onetrace(dataset, partnum) for partnum in partnums], axis=0)


norm_pulsephotons = avtrace(dataset, [3, 4, 5])

# particle_ = dataset['Particle 4']
# abstimes = particle_['Absolute Times (ns)']
# print(particle_.attrs['Description'])
#
# # print(abstimes[0])
# # print((abstimes[28018]-abstimes[0])/1e6)
# # print((abstimes[28019]-abstimes[28018])/1e6)
# difftime = np.diff(abstimes)
# boundary_photons = np.where(difftime > 10e6)[0]
# boundary_times = abstimes[boundary_photons]
# boundary_times_start = abstimes[boundary_photons + 1]
# boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
# # print(boundary_times)
# ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
# # print(ms_pulse)
# # print(boundary_times_start)
#
# pulsephotons = np.diff(boundary_photons)
# norm_pulsephotons = pulsephotons / ms_pulse[1:]
# norm_pulsephotons /= norm_pulsephotons[0]
# plt.plot(norm_pulsephotons[1:-9])
plt.plot(norm_pulsephotons[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')


# particle_ = dataset_thyl['Particle 23']
# abstimes = particle_['Absolute Times (ns)']
# print(particle_.attrs['Description'])
#
# # print(abstimes[0])
# # print((abstimes[28018]-abstimes[0])/1e6)
# # print((abstimes[28019]-abstimes[28018])/1e6)
# difftime = np.diff(abstimes)
# boundary_photons = np.where(difftime > 10e6)[0]
# boundary_times = abstimes[boundary_photons]
# boundary_times_start = abstimes[boundary_photons + 1]
# boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
# # print(boundary_times)
# ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
# # print(ms_pulse)
# # print(boundary_times_start)
#
# pulsephotons = np.diff(boundary_photons)
# norm_pulsephotons = pulsephotons / ms_pulse[1:]
# norm_pulsephotons /= norm_pulsephotons[0]
# # plt.plot(norm_pulsephotons[1:-9])
# # plt.plot(norm_pulsephotons[:], label='500')
#
# particle_ = dataset_thyl['Particle 24']
# abstimes = particle_['Absolute Times (ns)']
# print(particle_.attrs['Description'])
#
# # print(abstimes[0])
# # print((abstimes[28018]-abstimes[0])/1e6)
# # print((abstimes[28019]-abstimes[28018])/1e6)
# difftime = np.diff(abstimes)
# boundary_photons = np.where(difftime > 10e6)[0]
# boundary_times = abstimes[boundary_photons]
# boundary_times_start = abstimes[boundary_photons + 1]
# boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
# # print(boundary_times)
# ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
# # print(ms_pulse)
# # print(boundary_times_start)
#
# pulsephotons = np.diff(boundary_photons)
# norm_pulsephotons = pulsephotons / ms_pulse[1:]
# norm_pulsephotons /= norm_pulsephotons[0]
# # plt.plot(norm_pulsephotons[1:-9])
# # plt.plot(norm_pulsephotons[:], label='181 mmol photons m$^{-2}$ s$^{-1}$')
# # plt.legend()
plt.xlabel('Pulse time (125 ms)')
plt.ylabel('Normalized photon count')
# plt.xlim(0, 140)
plt.show()
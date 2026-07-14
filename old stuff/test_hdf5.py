import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
import os

# data_dir = 'E:\SMS\Measurements\Bertus\LHCII\PAM\Modulate and gate'
# data_dir = 'blinking'
data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek'

# dataset = h5py.File('blinking/Test 3000 mE 2.h5', 'r')
# dataset = h5py.File('blinking/Power study.h5', 'r')
# dataset = h5py.File('blinking/700 mE.h5', 'r')
# dataset_thyl = h5py.File('blinking/12 Feb 2025/Thylakoid intensity study new.h5', 'r')
# dataset = h5py.File('blinking/11 Feb 2025/Ascorbic.h5', 'r')
# dataset_int = h5py.File('blinking/10 Feb 2025/Chemical study.h5', 'r')
# dataset_830 = h5py.File('blinking/11 Feb 2025/830 nm laser.h5', 'r')
# dataset = h5py.File('blinking/4 March 2025/test.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/Low pH.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/Azide.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/3000 mE.h5', 'r')
# dataset = h5py.File('blinking/4 Feb 2025/2380 mE.h5', 'r')

# dataset_150 = h5py.File(os.path.join(data_dir, '6 March 2025', '150 uW.h5'), 'r')
dataset_150 = h5py.File(os.path.join(data_dir, '12 April 2025', '3000uW.h5'), 'r')
# dataset = h5py.File(os.path.join(data_dir, '6 March 2025', '225 uW.h5'), 'r')
dataset = h5py.File(os.path.join(data_dir, '12 April 2025', '2500uW new.h5'), 'r')
# dataset = h5py.File(os.path.join(data_dir, '13 March 2025', 'Ascorbic new.h5'), 'r')


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    print(particle_.attrs['Description'])

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


# norm_pulsephotons = avtrace(dataset, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# norm_pulsephotons = avtrace(dataset, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
# norm_pulsephotons = avtrace(dataset, [27, 28, 29, 30, 31])
# norm_pulsephotons = avtrace(dataset, [31, 32, 33, 34, 35])
# norm_pulsephotons = avtrace(dataset, [38, 39, 40, 41, 42, 43, 44, 45, 46, 47])
norm_pulsephotons_150 = avtrace(dataset_150, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])#, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
norm_pulsephotons = avtrace(dataset, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])#, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# norm_pulsephotons = avtrace(dataset, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# norm_pulsephotons = avtrace(dataset, [26, 27, 28, 29, 30])

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
plt.plot(norm_pulsephotons_150[:], label='725 mmol photons m$^{-2}$ s$^{-1}$')
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
plt.grid()
plt.xlabel('Pulse time (125 ms)')
plt.ylabel('Normalized photon count')
# plt.xlim(0, 140)
plt.show()
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
import os

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/Maart 2026/'

dataset = h5py.File(os.path.join(data_dir, 'Control_switch_off.h5'), 'r')
dataset_AA = h5py.File(os.path.join(data_dir, 'AA_switch_off.h5'), 'r')
dataset_magnet = h5py.File(os.path.join(data_dir, 'Magnet_switch_off_new.h5'), 'r')
dataset_FC = h5py.File(os.path.join(data_dir, 'FC_switch_off.h5'), 'r')
dataset_GOC = h5py.File(os.path.join(data_dir, 'GOC_switch_off.h5'), 'r')
dataset_SOD = h5py.File(os.path.join(data_dir, 'SOD_switch_off.h5'), 'r')
dataset_power = h5py.File(os.path.join(data_dir, 'Power_study_off.h5'), 'r')
dataset_MV = h5py.File(os.path.join(data_dir, 'MV_switch_off_new.h5'), 'r')


def onetrace(dataset, partnum, offset):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    # print(particle_.attrs['Description'])
    start_time = np.min(abstimes)
    end_time = np.max(abstimes)
    bin_edges = np.arange(start_time, end_time + 100e6, 100e6)  # 10e6 ns = 10 ms bins

    # Count photons in each bin
    binned_counts, _ = np.histogram(abstimes, bins=bin_edges)
    binned_counts = binned_counts / np.mean(binned_counts[:50])  # normalize to starting int
    binned_counts += offset
    # print(binned_counts[1478])
    # binned_counts = binned_counts / binned_counts[1500]

    return binned_counts


def avtrace(dataset, partnums, offset=0):
    partnums_ = [onetrace(dataset, partnum, offset) for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    return np.mean(partnums_, axis=0)


# norm_pulsephotons = avtrace(dataset, [3, 4, 5, 8, 10, 11, 12])
# norm_pulsephotons_AA = avtrace(dataset_AA, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -0.28)
# norm_pulsephotons_magnet = avtrace(dataset_magnet, [2, 4, 5, 6, 7, 8, 10])
norm_pulsephotons = avtrace(dataset_magnet, [11, 12, 14, 15, 16, 17, 18, 19, 20])
# norm_pulsephotons_FC = avtrace(dataset_FC, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -0.22)
# norm_pulsephotons_MV = avtrace(dataset_MV, [1, 2, 3, 4, 5, 7, 8, 9, 10], -0.10)
# norm_pulsephotons_GOC = avtrace(dataset_GOC, [10, 12, 14, 15, 16, 17, 18, 19, 20], -0.45)
norm_pulsephotons_SOD = avtrace(dataset_SOD, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -0.02)
# norm_pulsephotons_2 = avtrace(dataset_power, [2, 3, 4, 5, 6, 7, 8, 9, 10])
# norm_pulsephotons_4 = avtrace(dataset_power, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 0.026)
# norm_pulsephotons_5 = avtrace(dataset_power, [21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 0.06)

xvals = np.arange(0, 9.91, 0.01)
plt.plot(norm_pulsephotons[149:245], label='Control')
# plt.plot(xvals, norm_pulsephotons[1484:2475], label='Control')
# plt.plot(xvals, norm_pulsephotons_magnet[1483:2475], label='Magnet', alpha=0.7)
# plt.plot(xvals, norm_pulsephotons_AA[1484:2475], label='AA')
# plt.plot(xvals, norm_pulsephotons_FC[1483:2475], label='FC', alpha=0.5)
# plt.plot(xvals, norm_pulsephotons_MV[1483:2475], label='MV', alpha=0.5)
plt.plot(norm_pulsephotons_SOD[149:245], label='SOD')
# plt.plot(xvals, norm_pulsephotons_SOD[1484:2475], label='SOD')
# plt.plot(xvals, norm_pulsephotons_2[1483:2475], label='140 uE')
# plt.plot(xvals, norm_pulsephotons_4[1483:2475], label='700 uE')
# plt.plot(xvals, norm_pulsephotons_5[1483:2475], label='2600 uE')
# plt.plot(xvals, norm_pulsephotons_GOC[3988:4979], label='GOC')

# plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Norm. Fluorescence (a.u.)')
# plt.xlim(0, 140)
plt.legend()
plt.show()
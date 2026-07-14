import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
import os

# data_dir = r'E:\SMS\Measurements\Bertus\LHCII\PAM\Modulate and gate\10 April 2025\Power study new'

data_dir = 'blinking/10 April 2025/Power study new'

dataset_3550 = h5py.File(os.path.join(data_dir, '3550.h5'), 'r')
dataset_2531 = h5py.File(os.path.join(data_dir, '2531.h5'), 'r')
dataset_1688 = h5py.File(os.path.join(data_dir, '1688.h5'), 'r')
dataset_1125 = h5py.File(os.path.join(data_dir, '1125.h5'), 'r')
dataset_750 = h5py.File(os.path.join(data_dir, '750.h5'), 'r')
dataset_500 = h5py.File(os.path.join(data_dir, '500.h5'), 'r')


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



# -------------------- 500 ------------------------ #
# norm_pulsephotons_500 = avtrace(dataset_500, [1, 2, 3, 4, 5])
# plt.plot(norm_pulsephotons_500[:], label='500 uW')
# plt.axhline(0.803)
# plt.axhline(0.818)
#
# plt.axhline(0.763)
# plt.axhline(0.778)
#
# plt.axhline(0.725)
# plt.axhline(0.747)
#
# plt.axhline(0.703)
# plt.axhline(0.719)

# -------------------- 750 ------------------------ #
# norm_pulsephotons_750 = avtrace(dataset_750, [1, 2, 3, 4, 5])
# plt.plot(norm_pulsephotons_750[:], label='750 uW')
# plt.axhline(0.775, color='C1')
# plt.axhline(0.797, color='C1')
#
# plt.axhline(0.735, color='C2')
# plt.axhline(0.755, color='C2')
#
# plt.axhline(0.700, color='C3')
# plt.axhline(0.724, color='C3')
#
# plt.axhline(0.675, color='C4')
# plt.axhline(0.697, color='C4')

# -------------------- 1125 ------------------------ #
# norm_pulsephotons_1125 = avtrace(dataset_1125, [1, 2, 3, 4, 5])
# plt.plot(norm_pulsephotons_1125[:], label='1125 uW')
# plt.axhline(0.762, color='C1')
# plt.axhline(0.785, color='C1')
#
# plt.axhline(0.715, color='C2')
# plt.axhline(0.740, color='C2')
#
# plt.axhline(0.683, color='C3')
# plt.axhline(0.704, color='C3')
#
# plt.axhline(0.652, color='C4')
# plt.axhline(0.677, color='C4')

# -------------------- 1688 ------------------------ #
# norm_pulsephotons_1688 = avtrace(dataset_1688, [1, 2, 3, 4, 5])
# plt.plot(norm_pulsephotons_1688[:], label='1688 uW')
# plt.axhline(0.783, color='C1')
# plt.axhline(0.808, color='C1')
#
# plt.axhline(0.737, color='C2')
# plt.axhline(0.768, color='C2')
#
# plt.axhline(0.705, color='C3')
# plt.axhline(0.733, color='C3')
#
# plt.axhline(0.679, color='C4')
# plt.axhline(0.708, color='C4')

# -------------------- 2531 ------------------------ #
# norm_pulsephotons_2531 = avtrace(dataset_2531, [1, 2, 3, 4, 5])
# plt.plot(norm_pulsephotons_2531[:], label='2531 uW')
# plt.axhline(0.753, color='C1')
# plt.axhline(0.790, color='C1')
#
# plt.axhline(0.713, color='C2')
# plt.axhline(0.742, color='C2')
#
# plt.axhline(0.672, color='C3')
# plt.axhline(0.711, color='C3')
#
# plt.axhline(0.645, color='C4')
# plt.axhline(0.681, color='C4')

# -------------------- 3550 ------------------------ #
# norm_pulsephotons_3550 = avtrace(dataset_3550, [1, 2, 3, 4, 5])
# plt.plot(norm_pulsephotons_3550[:], label='3550 uW')
# plt.axhline(0.756, color='C1')
# plt.axhline(0.792, color='C1')
#
# plt.axhline(0.713, color='C2')
# plt.axhline(0.750, color='C2')
#
# plt.axhline(0.682, color='C3')
# plt.axhline(0.719, color='C3')
#
# plt.axhline(0.658, color='C4')
# plt.axhline(0.694, color='C4')

plt.legend()
plt.grid()
plt.xlabel('Pulse time (125 ms)')
plt.ylabel('Normalized photon count')
# plt.xlim(0, 140)
plt.show()
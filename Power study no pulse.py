import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns


def boundints(bin_int):
    on_ints = np.where(bin_int> 0.2)[0]  # Find indices where intensity is greater than 0.4
    peakbounds = np.where(np.diff(on_ints) != 1)[0]  # Boundaries of peaks is where there is a gap

    # Each gap is both a start and a stop of a peak, so we need to add the index after for each one in peakbounds.
    result = []
    for value in peakbounds:
        result.append(value)
        result.append(value+1)
    peakbounds = np.array(result)

    # Now we "hem in" a little to ensure we are not in the rise or falling edge of the peak.
    peakbounds[0::2] -= 1
    peakbounds[1::2] += 1
    return np.insert(bin_int[on_ints[peakbounds]], 0, 1)


def blink_bleach(bin_int, plot=False):
    bndints = boundints(bin_int)
    print(bndints)
    blink = np.diff(bndints)[1::2]
    bleach = -np.diff(bndints[::2])
    if plot:
        plt.plot(blink)
        plt.show()
    return np.mean(blink[-8:]), np.std(blink[-8:]) / np.sqrt(8), np.mean(bleach[-5:])


# data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/1 May 2025/Power study traces'
data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/3 June 2025/Power study 1'
# Load the CSV file
# Replace 'file.csv' with the path to your CSV file
data = np.loadtxt(os.path.join(data_dir, 'Particle 1 trace.csv'), delimiter=',', skiprows=1)

def avtrace(particles, stop=-1):
    bin_intensity = None
    parts = []
    for part in particles:
        data = np.loadtxt(os.path.join(data_dir, f'Particle {part} trace.csv'), delimiter=',', skiprows=1)
        parts.append(data)
    stop = min(len(data) for data in parts)
    for data in parts:
        bin_time = data[:stop, 1]    # Second column
        if bin_intensity is None:
            bin_intensity = data[:stop, 2]  # Third column
        else:
            bin_intensity += data[:stop, 2]

    bin_intensity /= bin_intensity.max()
    return bin_time, bin_intensity


# bin_time_3800, bin_intensity_3800 = avtrace([3, 4, 5])
bin_time_3800, bin_intensity_3800 = avtrace([1])
# bin_time_2530, bin_intensity_2530 = avtrace([6, 7, 8, 9])
bin_time_2530, bin_intensity_2530 = avtrace([2])
# bin_time_1720, bin_intensity_1720 = avtrace([10, 11, 12, 13])
bin_time_1720, bin_intensity_1720 = avtrace([3])
bin_time_1140, bin_intensity_1140 = avtrace([5])
# bin_time_750, bin_intensity_750 = avtrace([17, 18, 19])
bin_time_500, bin_intensity_500 = avtrace([6])
# bin_time_350, bin_intensity_350 = avtrace([26, 27])
bin_time_230, bin_intensity_230 = avtrace([8])
# bin_time_150, bin_intensity_150 = avtrace([29, 30, 32])

blink_3800, std_3800, bleach_3800 = blink_bleach(bin_intensity_3800)
blink_2530, std_2530, bleach_2530 = blink_bleach(bin_intensity_2530)
blink_1720, std_1720, bleach_1720 = blink_bleach(bin_intensity_1720)
blink_1140, std_1140, bleach_1140 = blink_bleach(bin_intensity_1140)
# blink_750, std_750, bleach_750 = blink_bleach(bin_intensity_750)
blink_500, std_500, bleach_500 = blink_bleach(bin_intensity_500)
# blink_350, std_350, bleach_350 = blink_bleach(bin_intensity_350)
blink_230, std_230, bleach_230 = blink_bleach(bin_intensity_230)
# blink_150, std_150, bleach_150 = blink_bleach(bin_intensity_150)

# blinks = [blink_3800, blink_2530, blink_1720, blink_1140, blink_750, blink_500, blink_350]#, blink_230, blink_150]
blinks = [blink_3800, blink_2530, blink_1720, blink_1140, blink_500, blink_230]#, blink_150]
stds = [std_3800, std_2530, std_1720, std_1140, std_500, std_230]#, std_150]
# stds = [std_3800, std_2530, std_1720, std_1140, std_750, std_500, std_350]#, std_230, std_150]
bleachs = [bleach_3800, bleach_2530, bleach_1720, bleach_1140, bleach_500, bleach_230]
# bleachs = [bleach_3800, bleach_2530, bleach_1720, bleach_1140, bleach_750, bleach_500, bleach_350]#, bleach_230, bleach_150]
powers = [3800, 2530, 1720, 1140, 500, 230]
mE = [986, 657, 445, 296, 130, 60]

sns.set_context('talk')

plt.figure(figsize=(10, 6))
plt.errorbar(powers[:], blinks[:], stds, None, 'o', capsize=5, label='Blink')
# plt.plot(powers, bleachs, label='Bleach')
plt.xlabel('Photon flux (mE)')
plt.ylabel('Blinking yield')
plt.ylim(0, 0.1)
plt.tight_layout()

# plt.plot(bin_time_3800, bin_intensity_3800)
# plt.plot(bin_time_2530, bin_intensity_2530)
# plt.plot(bin_time_1720, bin_intensity_1720)
# plt.plot(bin_time_1140, bin_intensity_1140)
# plt.plot(bin_time_750, bin_intensity_750)
# plt.plot(bin_time_500, bin_intensity_500)
# plt.plot(bin_time_350, bin_intensity_350)
# plt.plot(bin_time_230, bin_intensity_230)
# plt.plot(bin_time_150, bin_intensity_150)

plt.show()

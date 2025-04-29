import numpy as np
import os
from matplotlib import pyplot as plt

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/12 April 2025/Power study traces'
# Load the CSV file
# Replace 'file.csv' with the path to your CSV file
data = np.loadtxt(os.path.join(data_dir, 'Particle 1 trace.csv'), delimiter=',', skiprows=1)

def avtrace(particles, stop=-1):
    bin_intensity = None
    for part in particles:
        data = np.loadtxt(os.path.join(data_dir, f'Particle {part} trace.csv'), delimiter=',', skiprows=1)

        bin_time = data[:stop, 1]    # Second column
        if bin_intensity is None:
            bin_intensity = data[:stop, 2]  # Third column
        else:
            bin_intensity += data[:stop, 2]

    bin_intensity /= bin_intensity.max()
    return bin_time, bin_intensity


bin_time_3000, bin_intensity_3000 = avtrace([2, 3, 4])
bin_time_2500, bin_intensity_2500 = avtrace([5, 6, 7])
bin_time_2000, bin_intensity_2000 = avtrace([8, 9, 10])
bin_time_1500, bin_intensity_1500 = avtrace([11, 12, 13])
bin_time_1000, bin_intensity_1000 = avtrace([14, 15, 16])
bin_time_500, bin_intensity_500 = avtrace([17, 18, 19])
bin_time_300, bin_intensity_300 = avtrace([20, 21, 22], stop=3000)
bin_time_150, bin_intensity_150 = avtrace([23, 24])

plt.plot(bin_time_3000, bin_intensity_3000)
plt.axhline(0.64, color='k', xmin=0.1, xmax=0.3)
plt.axhline(0.68, color='k', xmin=0.2, xmax=0.4)
plt.axhline(0.53, color='k', xmin=0.25, xmax=0.45)
plt.axhline(0.59, color='k', xmin=0.35, xmax=0.55)
plt.axhline(0.46, color='k', xmin=0.4, xmax=0.6)
plt.axhline(0.53, color='k', xmin=0.5, xmax=0.7)
plt.axhline(0.41, color='k', xmin=0.55, xmax=0.75)
plt.axhline(0.47, color='k', xmin=0.65, xmax=0.85)
plt.axhline(0.37, color='k', xmin=0.7, xmax=0.9)
plt.axhline(0.43, color='k', xmin=0.8, xmax=1)
plt.axhline(0.35, color='k', xmin=0.85, xmax=1)


# plt.plot(bin_time_2500, bin_intensity_2500)
# plt.plot(bin_time_2000, bin_intensity_2000)
# plt.plot(bin_time_1500, bin_intensity_1500)
# plt.plot(bin_time_1000, bin_intensity_1000)
# plt.plot(bin_time_500, bin_intensity_500)
# plt.plot(bin_time_300, bin_intensity_300)
# plt.plot(bin_time_150, bin_intensity_150)
plt.show()

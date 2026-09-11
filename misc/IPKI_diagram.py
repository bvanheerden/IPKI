import sys
import os

# Add the project root (where this file is located) to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import seaborn as sns
import utils

utils.setup_plotting()
sns.set_palette('deep')

# 1. Define the parameters
frequency = 20        # Frequency in Hz
sampling_rate = 2000  # Samples per second
duration_1 = .15         # Duration in seconds
duration_2 = .20         # Duration in seconds
duration_3 = .55         # Duration in seconds
phase_shift = np.pi

# 2. Create the time array
t_1 = np.linspace(0, duration_1, int(sampling_rate*duration_1), endpoint=False)
t_2 = np.linspace(0, duration_2, int(sampling_rate*duration_2), endpoint=False) + t_1[-1]
t_3 = np.linspace(0, duration_3, int(sampling_rate*duration_3), endpoint=False) + t_2[-1]

# 3. Generate the square wave
# The argument for signal.square is the phase (2 * pi * frequency * t)
baseline = 0.5 * signal.square(2 * np.pi * frequency * t_1 + phase_shift, duty=0.1) + 0.5
recovery = 0.5 * signal.square(2 * np.pi * frequency * t_3 + phase_shift, duty=0.1) + 0.5
illum = - 4 * signal.square(2 * np.pi * frequency * t_2 + phase_shift, duty=0.1) + 5

full_trace = np.concatenate([baseline, illum, recovery])
t = np.concatenate([t_1, t_2, t_3])

# 4. Generate fluorescence trace
f_baseline = np.ones_like(t_1) * 8
f_illum = 8 * np.exp(-10 * (t_2 - t_1[-1]))
f_recovery = 5 - (5 - f_illum[-1]) * np.exp(-5 * (t_3 - t_2[-1]))

fluor_trace = np.concatenate([f_baseline, f_illum, f_recovery])

phases = [
    (0, duration_1, 'black'),
    (duration_1, duration_1+duration_2, 'white'),
    (duration_1+duration_2, duration_1+duration_2+duration_3, 'black'),
    ]

plt.figure(figsize=(90/25.4, 40/25.4))
plt.plot(t, full_trace, color='C0', linewidth=1.5, label='Illumination', alpha=0.8)

# Plot fluorescence points above pulses
mask = (full_trace < 8) & (full_trace > 0.6)
plt.scatter(t[mask], fluor_trace[mask], color='C3', s=4, zorder=3, label='Fluorescence')
# plt.plot(t, fluor_trace, color='C3', zorder=3, alpha=0.5)

# for start, end, color in phases:
#     plt.axvspan(start, end, ymin=0.92, ymax=1.0, facecolor=color,
#                 edgecolor='black', linewidth=0.5, transform=plt.gca().get_xaxis_transform())

plt.xlabel('Time')
plt.ylabel('Intensity')
plt.xticks([])
plt.yticks([])
plt.xlim(0, t.max())
plt.ylim(-0.15, None)
plt.legend(frameon=False, loc='upper right')
sns.despine()
plt.tight_layout()
plt.show()

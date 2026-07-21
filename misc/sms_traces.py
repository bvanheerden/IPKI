import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils

utils.setup_plotting()

data_dir = '../blinking/Trace_examples'

part2_trace = pd.read_csv(os.path.join(data_dir, 'Particle 2 trace.csv'))
part2_levels = pd.read_csv(os.path.join(data_dir, 'Particle 2 levels-grouped-plot (ROI).csv'))
part20_trace = pd.read_csv(os.path.join(data_dir, 'Particle 20 trace.csv'))
part20_levels = pd.read_csv(os.path.join(data_dir, 'Particle 20 levels-grouped-plot (ROI).csv'))
part55_trace = pd.read_csv(os.path.join(data_dir, 'Particle 55 trace.csv'))
part55_levels = pd.read_csv(os.path.join(data_dir, 'Particle 55 levels-plot (ROI).csv'))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(90/25.4, 150/25.4), dpi=150)

ax1.plot(part2_trace['Bin Time (s)'], part2_trace['Bin Int (counts/10ms)'], lw=1, alpha=0.6, color='C0')
ax1.plot(part2_levels['Time (s)'], part2_levels['Int (counts/s)']/100, color='C0')
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 70)
ax1.text(12, 61, 'Air', fontsize=10)
ax2.plot(part20_trace['Bin Time (s)'], part20_trace['Bin Int (counts/10ms)'], lw=1, alpha=0.6)
ax2.plot(part20_levels['Time (s)'], part20_levels['Int (counts/s)']/100, color='C0')
ax2.set_xlim(0, 29)
ax2.set_ylim(0, 50)
ax2.text(21, 42, r'10.2% O$_2$', fontsize=10)
ax2.set_ylabel('Binned Intensity (counts/10ms)')
ax3.plot(part55_trace['Bin Time (s)'], part55_trace['Bin Int (counts/10ms)'], lw=1, alpha=0.6)
ax3.plot(part55_levels['Time (s)'], part55_levels['Int (counts/s)']/100, color='C0')
ax3.set_xlim(0, 64)
ax3.set_ylim(0, None)
ax3.text(45.7, 57, '1.2% O$_2$', fontsize=10)
ax3.set_xlabel('Illumination Time (s)')
plt.tight_layout()
plt.show()


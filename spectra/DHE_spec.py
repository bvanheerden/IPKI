import sys
import os

# Add the project root to sys.path to allow imports of utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import utils
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

utils.setup_plotting()

datadir = os.path.join(project_root, 'blinking\\DHE\\')


def loadspec(filename, existing_df, run):
    return utils.load_spectrum(filename, datadir, existing_df, run)

dataset = 'Low power'

# Load DHE data (Control)
dhe_df = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
for rep in [1, 2, 3]:
    for i in [0, 2, 4]:
        dhe_df = loadspec(f'DHE {rep} 410 nm {i} min.asc', dhe_df, f'{i}')

# Load DHE+SOD data
dhe_sod_df = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
for rep in [1, 2, 3]:
    for i in [0, 2, 4, 6]:
        dhe_sod_df = loadspec(f'DHE+SOD {rep} 410 nm {i} min.asc', dhe_sod_df, f'{i}')

# Load DHE_LHCII data (Wait, the original code called DHE_LHCII but used "No LHCII Control")
# Original lines 34-36:
# DHE_LHCII = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
# for i in [0, 2, 4]:
#     DHE_LHCII = loadspec(f'No LHCII Control {i} min.asc', DHE_LHCII, f'{i}')

dhe_lhcii_df = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
for i in [0, 2, 4]:
    dhe_lhcii_df = loadspec(f'No LHCII Control {i} min.asc', dhe_lhcii_df, f'{i}')

# Wavelengths (assumed same for all)
wav = dhe_lhcii_df[dhe_lhcii_df['run'] == '4']['Wavelength (nm)'].unique()[::-1]

# Find index for 620 nm normalization
idx_620 = np.abs(wav - 620).argmin()

# Process DHE_LHCII (Control?)
lhcii_zero = dhe_lhcii_df[dhe_lhcii_df['run'] == '0'].groupby('Wavelength (nm)')['Intensity (counts)'].mean().values / 2.2
lhcii_four = dhe_lhcii_df[dhe_lhcii_df['run'] == '4'].groupby('Wavelength (nm)')['Intensity (counts)'].mean().values / 2.2

# Normalize by 620 nm value of initial spectrum
norm_lhcii = lhcii_zero[idx_620]
lhcii_zero = lhcii_zero / norm_lhcii
lhcii_four = lhcii_four / norm_lhcii
lhcii_diff = lhcii_four - lhcii_zero

# Process DHE
dhe_zero = dhe_df[dhe_df['run'] == '0'].groupby('Wavelength (nm)')['Intensity (counts)'].mean().values.copy()
dhe_four = dhe_df[dhe_df['run'] == '4'].groupby('Wavelength (nm)')['Intensity (counts)'].mean().values.copy()

# Normalize by 620 nm value of initial spectrum
norm_dhe = dhe_zero[idx_620]
dhe_zero = dhe_zero / norm_dhe
dhe_four = dhe_four / norm_dhe
dhe_diff = dhe_four - dhe_zero

# Process DHE+SOD
sod_zero = dhe_sod_df[dhe_sod_df['run'] == '0'].groupby('Wavelength (nm)')['Intensity (counts)'].mean().values.copy()
sod_four = dhe_sod_df[dhe_sod_df['run'] == '4'].groupby('Wavelength (nm)')['Intensity (counts)'].mean().values.copy()

# Normalize by 620 nm value of initial spectrum
norm_sod = sod_zero[idx_620]
sod_zero = sod_zero / norm_sod
sod_four = sod_four / norm_sod
sod_diff = sod_four - sod_zero

fig, axes = plt.subplots(2, 2, figsize=utils.get_figure_size(140, 110),
                         sharex=True)
ax_lhcii, ax_dhe, ax_sod, ax_diff = axes.flatten()

# Plot 1: Difference Spectra
ax_diff.plot(wav, lhcii_diff, label='DHE', lw=2)
ax_diff.plot(wav, dhe_diff, label='DHE+LHCII', lw=2)
ax_diff.plot(wav, sod_diff, label='DHE+LHCII+SOD', lw=2)
ax_diff.legend(title='', frameon=False)
ax_diff.set_ylabel(r'$\Delta$ Fluorescence (norm.)')
# ax_diff.set_ylim((0, 1.1))
ax_diff.set_xlim((550, 710))

# Inset for Plot 1
mask = (wav >= 550) & (wav <= 650)
ax_diff_ins = inset_axes(ax_diff, width="40%", height="40%", loc='upper left', borderpad=4)
ax_diff_ins.plot(wav, lhcii_diff, lw=1)
ax_diff_ins.plot(wav, dhe_diff, lw=1)
ax_diff_ins.plot(wav, sod_diff, lw=1)
ax_diff_ins.set_xlim(550, 650)
diff_min = min(lhcii_diff[mask].min(), dhe_diff[mask].min(), sod_diff[mask].min())
diff_max = max(lhcii_diff[mask].max(), dhe_diff[mask].max(), sod_diff[mask].max())
padding = (diff_max - diff_min) * 0.05
ax_diff_ins.set_ylim(diff_min - padding, diff_max + padding)
# ax_diff_ins.tick_params(labelsize=8)
# mark_inset(ax_diff, ax_diff_ins, loc1=2, loc2=4, fc="none", ec="black")
ax_diff.set_xlabel('Wavelength (nm)')
ax_diff.text(0.05, 0.97, 'd', transform=ax_diff.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')

# Plot 2: DHE
ax_lhcii.plot(wav, lhcii_zero, label='0 min', color='black')
ax_lhcii.plot(wav, lhcii_four, label='4 min', color='red')
# ax_lhcii.set_title('DHE')
ax_lhcii.legend(frameon=False, loc='lower right')
ax_lhcii.text(0.05, 0.97, 'a', transform=ax_lhcii.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')

# Plot 3: DHE + LHCII
ax_dhe.plot(wav, dhe_zero, label='0 min', color='black')
ax_dhe.plot(wav, dhe_four, label='4 min', color='red')
# ax_dhe.set_title('DHE+LHCII')
ax_dhe.set_ylim((0, 40))
ax_dhe.text(0.05, 0.97, 'b', transform=ax_dhe.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')

# Inset for Plot 3
ax_dhe_ins = inset_axes(ax_dhe, width="40%", height="40%", loc='upper left', borderpad=3)
ax_dhe_ins.plot(wav, dhe_zero, color='black', lw=1)
ax_dhe_ins.plot(wav, dhe_four, color='red', lw=1)
ax_dhe_ins.set_xlim(550, 650)
dhe_min = min(dhe_zero[mask].min(), dhe_four[mask].min())
dhe_max = max(dhe_zero[mask].max(), dhe_four[mask].max())
padding_dhe = (dhe_max - dhe_min) * 0.05
ax_dhe_ins.set_ylim(dhe_min - padding_dhe, dhe_max + padding_dhe)
# ax_dhe_ins.tick_params(labelsize=8)
# mark_inset(ax_dhe, ax_dhe_ins, loc1=2, loc2=4, fc="none", ec="black")

# Plot 4: DHE + LHCII + SOD
ax_sod.plot(wav, sod_zero, label='0 min', color='black')
ax_sod.plot(wav, sod_four, label='4 min', color='red')
# ax_sod.set_title('DHE+LHCII+SOD')
ax_sod.set_xlabel('Wavelength (nm)')
ax_sod.set_ylim((0, 40))
ax_sod.text(0.05, 0.97, 'c', transform=ax_sod.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')

# Inset for Plot 4
ax_sod_ins = inset_axes(ax_sod, width="40%", height="40%", loc='upper left', borderpad=3)
ax_sod_ins.plot(wav, sod_zero, color='black', lw=1)
ax_sod_ins.plot(wav, sod_four, color='red', lw=1)
ax_sod_ins.set_xlim(550, 650)
sod_min = min(sod_zero[mask].min(), sod_four[mask].min())
sod_max = max(sod_zero[mask].max(), sod_four[mask].max())
padding_sod = (sod_max - sod_min) * 0.05
ax_sod_ins.set_ylim(sod_min - padding_sod, sod_max + padding_sod)
# ax_sod_ins.tick_params(labelsize=8)
# mark_inset(ax_sod, ax_sod_ins, loc1=2, loc2=4, fc="none", ec="black")

for ax in axes.flatten():
    if ax != ax_diff:
        ax.set_ylabel('Fluorescence (norm.)')

plt.tight_layout()
# plt.savefig('DHE_spec.pdf')
plt.show()

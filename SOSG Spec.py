import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

datadir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/SOSG/'


def loadspec(filename, existing_df, run):
    # filename = datadir + 'SOSG Test 7_' + filename
    filename = datadir + 'SOSG export/' + filename
    new_data = pd.read_csv(filename, names=['Wavelength (nm)', 'Intensity (counts)'], usecols=[0, 1], dtype=np.float64,
                           skiprows=36)
    new_data['run'] = run
    return pd.concat([existing_df, new_data], ignore_index=True)

# dataset = ''
# dataset = ' lower power'
dataset = 'Low power'
# dataset = 'High power'

LHCII = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
# LHCII = loadspec(f'LHCII Before illuminate.asc', LHCII, '0')
LHCII = loadspec(f'LHCII Baseline.asc', LHCII, '0')
# LHCII = loadspec(f'LHCII Baseline High power.asc', LHCII, '0')
LHCII_max = LHCII['Intensity (counts)'][46]
# for i in range(6):
    # LHCII = loadspec(f'LHCII {i+1} min illuminate.asc', LHCII, f'{i+1}')
for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
        LHCII = loadspec(f'LHCII {dataset} {i} min.asc', LHCII, f'{i}')

SOSG_LHCII = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
# SOSG_LHCII = loadspec(f'SOSG LHCII Before illuminate{dataset}.asc', SOSG_LHCII, '0')
SOSG_LHCII = loadspec(f'SOSG LHCII Baseline.asc', SOSG_LHCII, '0')
# SOSG_LHCII = loadspec(f'SOSG LHCII Baseline High power.asc', SOSG_LHCII, '0')
# for i in range(6):
    # SOSG_LHCII = loadspec(f'SOSG LHCII {i+1} min illuminate{dataset}.asc', SOSG_LHCII, f'{i+1}')
for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
#     SOSG_LHCII = loadspec(f'Low power {i} min.asc', SOSG_LHCII, f'{i}')
    SOSG_LHCII = loadspec(f'SOSG LHCII {dataset} {i} min.asc', SOSG_LHCII, f'{i}')

SOSG = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
# SOSG = loadspec(f'SOSG Before illuminate.asc', SOSG, '0')
SOSG = loadspec(f'SOSG Baseline.asc', SOSG, '0')
# SOSG = loadspec(f'SOSG Baseline High power.asc', SOSG, '0')
# for i in range(6):
#     SOSG = loadspec(f'SOSG {i+1} min illuminate.asc', SOSG, f'{i+1}')
for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
    SOSG = loadspec(f'SOSG {dataset} {i} min.asc', SOSG, f'{i}')

# scale = SOSG_LHCII['Intensity (counts)'][46] / LHCII_max
# SOSG_LHCII['Intensity (counts)'] /= scale
#
# SOSG_max = SOSG_LHCII['Intensity (counts)'][220]
# scale = SOSG['Intensity (counts)'][220] / SOSG_max
# SOSG['Intensity (counts)'] /= scale

sns.set_context('notebook', font_scale=2, rc={"lines.linewidth": 3})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(30, 10))
sns.lineplot(LHCII, x='Wavelength (nm)', y='Intensity (counts)', hue='run', ax=ax1)
sns.lineplot(SOSG_LHCII, x='Wavelength (nm)', y='Intensity (counts)', hue='run', ax=ax2, legend=False)
sns.lineplot(SOSG, x='Wavelength (nm)', y='Intensity (counts)', hue='run', ax=ax3, legend=False)
ax1.legend(title='Minutes illuminated')
ax1.set_title('Only LHCII')
ax2.set_title('LHCII and SOSG')
ax3.set_title('Only SOSG')
ax1.set_ylabel('Norm. counts')

plt.tight_layout()
plt.show()

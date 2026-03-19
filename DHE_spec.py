import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

datadir = 'blinking/'


def loadspec(filename, existing_df, run):
    # filename = datadir + 'SOSG Test 7_' + filename
    filename = datadir + 'DHE/' + filename
    new_data = pd.read_csv(filename, names=['Wavelength (nm)', 'Intensity (counts)'], usecols=[0, 1], dtype=np.float64,
                           skiprows=36)
    new_data['run'] = run
    return pd.concat([existing_df, new_data], ignore_index=True)

dataset = 'Low power'

DHE = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
# DHE = loadspec(f'DHE 3 510 nm 0 min.asc', DHE, '0')
# LHCII_max = LHCII['Intensity (counts)'][46]
for i in [0, 2, 4, 6]:
    DHE = loadspec(f'DHE 2 410 nm {i} min.asc', DHE, f'{i}')

DHE_SOD = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
for i in [0, 2, 4, 6]:
    DHE_SOD = loadspec(f'DHE+SOD 2 410 nm {i} min.asc', DHE_SOD, f'{i}')

DHE_LHCII = pd.DataFrame(columns=['Wavelength (nm)', 'Intensity (counts)'])
for i in [0, 2, 4]:
    DHE_LHCII = loadspec(f'No LHCII Control {i} min.asc', DHE_LHCII, f'{i}')

zero_fluo = np.array(DHE_LHCII[DHE_LHCII['run'] == '0']['Intensity (counts)']) / 2.2
four_fluo = np.array(DHE_LHCII[DHE_LHCII['run'] == '4']['Intensity (counts)']) / 2.2
norm_fluo = four_fluo - zero_fluo
norm_wav = DHE_LHCII[DHE_LHCII['run'] == '4']['Wavelength (nm)']

zero_fluo = np.array(DHE[DHE['run'] == '0']['Intensity (counts)'])
four_fluo = np.array(DHE[DHE['run'] == '4']['Intensity (counts)'])
norm_fluo_LHCII = four_fluo - zero_fluo
# norm_fluo_LHCII = zero_fluo

zero_fluo = np.array(DHE_SOD[DHE_SOD['run'] == '0']['Intensity (counts)'])
four_fluo = np.array(DHE_SOD[DHE_SOD['run'] == '4']['Intensity (counts)'])
norm_fluo_SOD = four_fluo - zero_fluo

sns.set_context('notebook', font_scale=1, rc={"lines.linewidth": 3})
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(10, 5))
# sns.lineplot(DHE[DHE['run']=='0'], x='Wavelength (nm)', y='Intensity (counts)', label='0 min', ax=ax1)
# sns.lineplot(DHE[DHE['run']=='4'], x='Wavelength (nm)', y='Intensity (counts)', label='DHE+LHCII 4 min', ax=ax1)
# sns.lineplot(DHE_SOD[DHE_SOD['run'] == '4'], x='Wavelength (nm)', y='Intensity (counts)', label='DHE+LHCII+SOD 4 min', ax=ax1)
# sns.lineplot(DHE_LHCII[DHE_LHCII['run'] == '0'], x='Wavelength (nm)', y='Intensity (counts)', label='DHE 4 min', ax=ax1)
ax1.plot(norm_wav, norm_fluo, label='DHE')
ax1.plot(norm_wav, norm_fluo_LHCII, label='DHE+LHCII')
ax1.plot(norm_wav, norm_fluo_SOD, label='DHE+LHCII+SOD')
ax1.legend(title='')
# ax2.legend(title='Min. illum.')
# ax3.legend(title='Min. illum.')
# ax1.set_title('DHE-SOD')
# ax2.set_title('DHE+SOD')
# ax3.set_title('DHE-LHCII')
# ax1.set_ylabel('Norm. counts')
ax1.set_xlim((550, 635))
# ax2.set_xlim((550, 640))
# ax3.set_xlim((550, 640))
ax1.set_ylim((-900, 17200))
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Fluorescence (a.u.)')

plt.tight_layout()
plt.show()

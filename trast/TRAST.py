import sys
import os

# Add the project root to sys.path to allow imports of utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

utils.setup_plotting()

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/'

data = pd.read_csv(data_dir + 'TRAST.csv')

data['ox'] = data['ox'] / data['ox'].iloc[0]
data['no ox'] = data['no ox'] / data['no ox'].iloc[0]

data = pd.melt(data, id_vars='us', value_vars=['ox', 'no ox'], var_name='Condition',
               value_name='Normalized photon count')

mapping = {'ox': 'High Oxygen', 'no ox': 'Low Oxygen'}
data['Label'] = data['Condition'].map(mapping)

g = sns.lineplot(x="us", data=data, hue="Label", y="Normalized photon count")

# Add labels and title
plt.xlabel("Pulse width (us)")
plt.ylabel("Normalized photon count")
plt.xscale('log')
g.legend_.set_title(None)
# plt.legend(labels=['High Oxygen', 'Low Oxygen'], title='Condition')

# Show the plot
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
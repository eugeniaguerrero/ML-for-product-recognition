from GroupProject.src.PREPROCESSING.frame_differencing_folders import main_diff
import cProfile
import os
import pstats
from pstatsviewer import StatsViewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# inputs to the function you want to test
my_path = os.getcwd()

#my_folders = ['cprofile_graphs']
my_folders = ['sample1']

# edit the string to add your inputs. Do not edit restats
cProfile.run('main_diff(my_folders, my_path)', 'restats')

p = pstats.Stats('restats')
p.sort_stats('tottime').print_stats(10)

my_data = StatsViewer('restats')

# select the top 5 Total Time functions
data = my_data._get_timing_data(5, 'tottime', 'tottime')

my_array = np.asarray(data.axes)
my_labels = my_array.flatten()

my_labels = [w.replace('_', '-') for w in my_labels]


my_values = data.asobject

df = pd.DataFrame({
        'value': my_values,
        'label': my_labels,
        'color': ['g', 'b', 'k', 'y', 'm']})


fig, ax = plt.subplots()

# Plot each bar separately and give it a label.
for index, row in df.iterrows():
    ax.bar([index],
           [row['value']],
           label=row['label'],
           alpha=0.5,
           align='center',
           color=row['color'])

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, -0.05),
          ncol=1)

plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    top='off',
    left='off',
    right='off',
    labelbottom='off')

ax.margins(0.05)
ax.set_ylim(bottom=0)
ax.set(axisbelow=True, xticklabels=[])
plt.ylabel('Time (Seconds)')

fig.savefig('my_plot.png', bbox_inches='tight')




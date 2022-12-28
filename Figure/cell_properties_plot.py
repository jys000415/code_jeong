# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:01:05 2022

@author: yeong
"""
#%% Plotting neuronal properties
# Return : Figure (2d, 3d neuronal properties) / Excitatory, Inhibitory ID

cellproperty = pd.DataFrame(columns=['HalfValleyWidth', 'Peak-Valley Ratio'])
for ijson in batchjson:
    tmpdf = pd.DataFrame({'HalfValleyWidth': ijson['hvwidth'],
                          'Peak-Valley Ratio': ijson['pvratio']})
    cellproperty = cellproperty.append(tmpdf, ignore_index=True)
cellproperty['Firing rate'] = meanfiringrate['Baseline']
cellproperty['Light'] = meanfiringrate['Light']
hue_order = ["Non-respond", "Active", "Inactive"]
fig = plt.figure(figsize=(6, 6))
ax = sns.scatterplot(data=cellproperty, x='HalfValleyWidth', hue_order=hue_order,
                     y='Firing rate', hue='Light', s=70,
                     palette=['lightgrey', 'dodgerblue', 'Orange'], linewidth=0)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper right',
          fontsize=15, borderaxespad=0)
sns.despine()
ax.set_xlabel('Half-Valley Width (us)')
ax.set_ylabel('Firing rate (Hz)')
plt.savefig('Cell classification')
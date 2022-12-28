# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:19:21 2022

@author: yeong
"""
# %% Batch Mean firing rate
# Mean firing rate in the baseline
from scipy.stats import pearsonr
duration = 15
remap_index = []
for ijson in range(len(batchjson)):
    taskname = batchjson[ijson]['spk'].keys()
    meanfiringrate = []    
    batchjson[ijson]['Mean Firingrate'] = {}
    batchjson[ijson]['Mean Firingrate']['Closed'] = {}
    batchjson[ijson]['Mean Firingrate']['Open'] = {}
    if 'L_Firing_rate_%s_2' %(contextname) in batchjson[ijson].keys():
         batchjson[ijson]['Mean Firingrate']['Closed_pre'] = {}
         batchjson[ijson]['Mean Firingrate']['Open_pre'] = {}
         batchjson[ijson]['Mean Firingrate']['Closed_post'] = {}
         batchjson[ijson]['Mean Firingrate']['Open_post'] = {}
         remap_index.append(ijson)
    hvwidth = batchjson[ijson]['hvwidth']
    rsneuron = [list(batchjson[ijson]['spk']['Baseline'].keys())[i]
                for i,v in enumerate(hvwidth) if v>=330]
    rsneuron_index = [i for i,v in enumerate(hvwidth) if v>=330]
    for iname in taskname:
        batchjson[ijson]['Mean Firingrate'][iname] = {}
        time_limit = batchjson[ijson]['time'][iname][0]+duration*60
        array_time = np.array(batchjson[ijson]['time'][iname])
        time = array_time[np.where(
            array_time <= time_limit)[0].tolist()]        
        taskcellname = list(batchjson[ijson]['spk'][iname].keys())
        tmptime = len(time)*0.025
        
        for ind, icell in zip(rsneuron_index, rsneuron):
            array_spk = np.array(batchjson[ijson]['spk'][iname][icell])
            list_spk = array_spk[np.where(array_spk <= time_limit)[0].tolist()]
            tmpspknum = len(list_spk)
            tmpfr = tmpspknum/tmptime
            batchjson[ijson]['Mean Firingrate'][iname][icell] = tmpfr       
            if iname == contextname:
                ttmpspk = np.sum([batchjson[ijson]['L_Spike_EPM'][ind][0:15],
                                batchjson[ijson]['L_Spike_EPM'][ind][21:36]])
                ttmptime = np.sum([batchjson[ijson]['L_Time_EPM'][0:15],
                                 batchjson[ijson]['L_Time_EPM'][21:36]])
                batchjson[ijson]['Mean Firingrate']['Closed'][icell] = ttmpspk/ttmptime
                ttmpspk = np.sum([batchjson[ijson]['L_Spike_EPM'][ind][39:54],
                                batchjson[ijson]['L_Spike_EPM'][ind][60:75]])
                ttmptime = np.sum([batchjson[ijson]['L_Time_EPM'][39:54],
                                 batchjson[ijson]['L_Time_EPM'][60:75]])
                batchjson[ijson]['Mean Firingrate']['Open'][icell] = ttmpspk/ttmptime
             
                if 'L_Firing_rate_%s_2' %(contextname) in batchjson[ijson].keys():
                    ttmpspk = np.sum([batchjson[ijson]['L_Spike_EPM'][ind][0:15],
                                batchjson[ijson]['L_Spike_EPM'][ind][21:36]])
                    ttmptime = np.sum([batchjson[ijson]['L_Time_EPM'][0:15],
                                     batchjson[ijson]['L_Time_EPM'][21:36]])
                    batchjson[ijson]['Mean Firingrate']['Closed_pre'][icell] = ttmpspk/ttmptime
                    ttmpspk = np.sum([batchjson[ijson]['L_Spike_EPM'][ind][39:54],
                                    batchjson[ijson]['L_Spike_EPM'][ind][60:75]])
                    ttmptime = np.sum([batchjson[ijson]['L_Time_EPM'][39:54],
                                     batchjson[ijson]['L_Time_EPM'][60:75]])
                    batchjson[ijson]['Mean Firingrate']['Open_pre'][icell] = ttmpspk/ttmptime
                 
                    ttmpspk = np.sum([batchjson[ijson]['L_Spike_EPM_2'][ind][0:15],
                                    batchjson[ijson]['L_Spike_EPM_2'][ind][21:36]])
                    ttmptime = np.sum([batchjson[ijson]['L_Time_EPM_2'][0:15],
                                     batchjson[ijson]['L_Time_EPM_2'][21:36]])
                    batchjson[ijson]['Mean Firingrate']['Closed_post'][icell] = ttmpspk/ttmptime
                    ttmpspk = np.sum([batchjson[ijson]['L_Spike_EPM_2'][ind][39:54],
                                    batchjson[ijson]['L_Spike_EPM_2'][ind][60:75]])
                    ttmptime = np.sum([batchjson[ijson]['L_Time_EPM_2'][39:54],
                                     batchjson[ijson]['L_Time_EPM_2'][60:75]])
                    batchjson[ijson]['Mean Firingrate']['Open_post'][icell] = ttmpspk/ttmptime

meanFr = pd.DataFrame(columns=['Baseline','EPM','Closed','Open','Postbaseline'])
baseline_Fr = [list(batchjson[ijson]['Mean Firingrate']['Baseline'].values()) 
               for ijson in range(len(batchjson))] 
meanFr['Baseline'] = [np.mean(i) for i in baseline_Fr]
baseline_Fr = np.concatenate(baseline_Fr)

postbaseline_Fr = [list(batchjson[ijson]['Mean Firingrate']['Postbaseline'].values()) 
               for ijson in range(len(batchjson))] 
meanFr['Baseline'] = [np.mean(i) for i in baseline_Fr]
baseline_Fr = np.concatenate(baseline_Fr)

Closed_Fr = [list(batchjson[ijson]['Mean Firingrate']['Closed'].values()) 
               for ijson in range(len(batchjson))] 
meanFr['Closed'] = [np.mean(i) for i in Closed_Fr]
Closed_Fr = np.concatenate(Closed_Fr)

Open_Fr = [list(batchjson[ijson]['Mean Firingrate']['Open'].values()) 
               for ijson in range(len(batchjson))] 
meanFr['Open'] = [np.mean(i) for i in Open_Fr]
Open_Fr = np.concatenate(Open_Fr)

pearsonr(durationInArm['Closed Arm'],meanFr['Baseline'])
plt.scatter(durationInArm['Closed Arm'], meanFr['Baseline'])

Closed_Fr_pre = [list(batchjson[ijson]['Mean Firingrate']['Closed_pre'].values()) 
               for ijson in remap_index] 
Closed_Fr_pre = np.concatenate(Closed_Fr_pre)

Closed_Fr_post = [list(batchjson[ijson]['Mean Firingrate']['Closed_post'].values()) 
               for ijson in remap_index] 
Closed_Fr_post = np.concatenate(Closed_Fr_post)

Open_Fr_pre = [list(batchjson[ijson]['Mean Firingrate']['Open_pre'].values()) 
               for ijson in remap_index] 
Open_Fr_pre = np.concatenate(Open_Fr_pre)

Open_Fr_post = [list(batchjson[ijson]['Mean Firingrate']['Open_post'].values()) 
               for ijson in remap_index] 
Open_Fr_post = np.concatenate(Open_Fr_post)

plt.scatter(baseline_Fr, Open_Fr)
pearsonr(baseline_Fr,Open_Fr)
plt.scatter(baseline_Fr, Closed_Fr)
pearsonr(baseline_Fr,Closed_Fr)

plt.scatter(Closed_Fr_pre, Closed_Fr_post)
pearsonr(Closed_Fr,Open_Fr)

plt.scatter(Open_Fr_pre, Open_Fr_post)
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:19:58 2022

@author: yeong
"""
# %% Duration in arm
from Linearize.areaSetting import areasetting
durationInArm = pd.DataFrame(columns=['Closed Arm', 'Open Arm'])
columns = list(durationInArm)
durationInArm_remap = pd.DataFrame(columns=['Closed Arm1', 'Open Arm1', 'Open Arm2', 'Closed Arm2'])
columns_remap = list(durationInArm_remap)
durationarray, durationarray_remap = [],[]
variablename = 'Time_%s' % (contextname)

for ijson in range(len(batchjson)):   
    tmpjsontime = batchjson[ijson][variablename]
    tmpclosedtime = (len(tmpjsontime[0])+len(tmpjsontime[1]))*0.025
    # tmpcentertime = len(tmpjsontime[2])*0.025
    tmpopentime = (len(tmpjsontime[-2])+len(tmpjsontime[-1]))*0.025
    tmpalltime = sum([len(l) for l in tmpjsontime])*0.025
    value = [tmpclosedtime/tmpalltime*100, tmpopentime/tmpalltime*100]
    tmpdict = dict(zip(columns, value))
    if 'L_Firing_rate_%s_2' %(contextname) in batchjson[ijson].keys():
        tmpjsontime_2 = batchjson[ijson]['%s_2' % (variablename)]
        tmpclosedtime_2 = (len(tmpjsontime_2[0])+len(tmpjsontime_2[1]))*0.025
        # tmpcentertime_2 = len(tmpjsontime_2[2])*0.025
        tmpopentime_2 = (len(tmpjsontime_2[-2])+len(tmpjsontime_2[-1]))*0.025
        # tmpalltime_2 = tmpclosedtime_2+tmpcentertime_2+tmpopentime_2
        tmpalltime_2 = sum([len(l) for l in tmpjsontime_2])*0.025
        value_remap = [tmpclosedtime/tmpalltime*100, tmpopentime/tmpalltime*100,
                 tmpopentime_2/tmpalltime_2*100, tmpclosedtime_2/tmpalltime_2*100]
        tmpdict_remap = dict(zip(columns_remap, value_remap))
        durationarray_remap.append(tmpdict_remap)
    durationarray.append(tmpdict)
durationInArm = durationInArm.append(durationarray, True)
durationInArm_remap = durationInArm_remap.append(durationarray_remap, True)
clrs = ['lightslategrey', 'lightgrey', 'lightgrey', 'lightslategrey']
xylabel = ['','Duration (%)']
yrange = [0, 100]
plot_dataframe(durationInArm_remap, clrs, 
                   xylabel, yrange, 'Duration in arm_remap %s' % (variablename),
                   'Duration in arm (%)', print_val=1)
plot_dataframe(durationInArm, clrs[0:2], 
                   xylabel, yrange, 'Duration in arm %s' % (variablename),
                   'Duration in arm (%)', print_val=1)

if contextname == 'allopenEPM':
    durationInArm = pd.DataFrame(columns=['Arm', 'Center'])
    columns = list(durationInArm)
    durationarray = []
    variablename = 'Time_%s' % (contextname)
    for ijson in range(len(batchjson)):   
        tmpjsontime = batchjson[ijson][variablename]
        tmparmtime = (len(tmpjsontime[0])+len(tmpjsontime[1])+len(tmpjsontime[3])+len(tmpjsontime[4]))*0.025
        tmpcentertime = len(tmpjsontime[2])*0.025
        tmpalltime = tmparmtime+tmpcentertime
        value = [tmparmtime/tmpalltime*100, tmpcentertime/tmpalltime*100]
        tmpdict = dict(zip(columns, value))
        durationarray.append(tmpdict)
    durationInArm = durationInArm.append(durationarray, True)
    clrs = ['lightslategrey', 'lightgrey']
    xylabel = ['','Duration (%)']
    yrange = [0, 100]
    plot_dataframe(durationInArm, clrs[0:2], 
                       xylabel, yrange, 'Duration in arm %s' % (variablename),
                       'Duration in arm (%)', print_val=1)
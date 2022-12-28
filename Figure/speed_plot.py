# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:21:48 2022

@author: yeong
"""
# %% Speed in arm
SpeedInArm = pd.DataFrame(columns=['Closed Arm', 'Open Arm'])
columns = list(SpeedInArm)
SpeedInArm_remap = pd.DataFrame(columns=['Closed Arm1', 'Open Arm1', 'Open Arm2', 'Closed Arm2'])
columns_remap = list(SpeedInArm_remap)
durationarray, durationarray_remap = [],[]
variablename = 'Speed_%s' % (contextname)
threshold = 0
for ijson in range(len(batchjson)):   
    tmpjsonspeed = batchjson[ijson][variablename]
    tmprunningspeed_closed = [i for i in tmpjsonspeed[0] if i > threshold]
    tmprunningspeed_closed2 = [i for i in tmpjsonspeed[1] if i > threshold]
    tmpclosedspeed = (np.mean(tmprunningspeed_closed)+np.mean(tmprunningspeed_closed2))/2
    tmprunningspeed_open = [i for i in tmpjsonspeed[-2] if i > threshold]
    tmprunningspeed_open2 = [i for i in tmpjsonspeed[-1] if i > threshold]
    tmpopenspeed = (np.mean(tmprunningspeed_open)+np.mean(tmprunningspeed_open2))/2
    value = [tmpclosedspeed, tmpopenspeed]
    tmpdict = dict(zip(columns, value))
    if 'L_Firing_rate_%s_2' %(contextname) in batchjson[ijson].keys():
        tmpjsonspeed2 = batchjson[ijson]['%s_2' % (variablename)]
        tmprunningspeed_remap_closed = [i for i in tmpjsonspeed2[0] if i > threshold]
        tmprunningspeed_remap_closed2 = [i for i in tmpjsonspeed2[1] if i > threshold]
        tmpclosedspeed2 = (np.mean(tmprunningspeed_remap_closed)+np.mean(tmprunningspeed_remap_closed2))/2
        tmprunningspeed_remap_open = [i for i in tmpjsonspeed2[-2] if i > threshold]
        tmprunningspeed_remap_open2 = [i for i in tmpjsonspeed2[-1] if i > threshold]
        tmpopenspeed2 = (np.mean(tmprunningspeed_remap_open)+np.mean(tmprunningspeed_remap_open2))/2
        value_remap = [tmpclosedspeed, tmpopenspeed, tmpopenspeed2, tmpclosedspeed2]
        tmpdict_remap = dict(zip(columns_remap, value_remap))
        durationarray_remap.append(tmpdict_remap)
    durationarray.append(tmpdict)
SpeedInArm = SpeedInArm.append(durationarray, True)
SpeedInArm_remap = SpeedInArm_remap.append(durationarray_remap, True)
clrs = ['lightslategrey', 'lightgrey', 'lightgrey', 'lightslategrey']
xylabel = ['','Duration (%)']
yrange = [0,20]
plot_dataframe(SpeedInArm, clrs[0:2], 
                   xylabel, yrange, 'Speed in arm %s' % (variablename),
                   'Speed in arm (cm/s)', print_val=1)
plot_dataframe(SpeedInArm_remap, clrs, 
                   xylabel, yrange, 'Speed in arm_remap_running %s' % (variablename),
                    'Speed in arm (cm/s)', print_val=1)
if contextname == 'allopenEPM':
    SpeedInArm = pd.DataFrame(columns=['Arm', 'Center'])
    columns = list(SpeedInArm)
    durationarray = []
    variablename = 'Speed_%s' % (contextname)
    threshold = 4
    for ijson in range(len(batchjson)):   
        tmpjsonspeed = batchjson[ijson][variablename]
        tmprunningspeed_closed = [i for i in tmpjsonspeed[0] if i > threshold]
        tmprunningspeed_closed2 = [i for i in tmpjsonspeed[1] if i > threshold]
        tmprunningspeed_center = [i for i in tmpjsonspeed[2] if i > threshold]
        tmprunningspeed_open = [i for i in tmpjsonspeed[-2] if i > threshold]
        tmprunningspeed_open2 = [i for i in tmpjsonspeed[-1] if i > threshold]
        tmpclosedspeed = (np.mean(tmprunningspeed_closed)+np.mean(tmprunningspeed_closed2)+
                          np.mean(tmprunningspeed_open)+np.mean(tmprunningspeed_open2))/4
        tmpopenspeed = (np.mean(tmprunningspeed_center))
        value = [tmpclosedspeed, tmpopenspeed]
        tmpdict = dict(zip(columns, value))
        durationarray.append(tmpdict)
    SpeedInArm = SpeedInArm.append(durationarray, True)
    clrs = ['lightslategrey', 'lightgrey']
    xylabel = ['','Duration (%)']
    yrange = [0,10]
    plot_dataframe(SpeedInArm, clrs[0:2], 
                       xylabel, yrange, 'Speed in maze running %s' % (variablename),
                       'Speed in arm (cm/s)', print_val=1)

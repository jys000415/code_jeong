# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:25:08 2021

@author: yeong
"""
# Multiunit activity

import pickle
from Spike_analysis.spikeanalysis import binnedSpike
import numpy as np
import mne


context_name = 'EPM'
f = open('Project1_Baseline_combine.pckl','rb')
spikedata = pickle.load(f)
f.close()

f = open('Project1_spike_EPM_combine.pckl','rb')
epmdata = pickle.load(f)
f.close()

f = open('Project1_position_time_EPM_combine.pckl','rb')
behaviordata = pickle.load(f)
f.close()

f = open('EPM_Spatial_binned_2cm_combine.pckl','rb')
frdata = pickle.load(f)
f.close()

f = open('figure1_EPM_combine.pckl','rb')
speeddata = pickle.load(f)
f.close()

epmspeed = speeddata[6]
for ianimal in range(5):
    del epmspeed[ianimal][-1]
    
basespeed = speeddata[17]
postbasespeed = speeddata[19]
animalgaussfr = frdata[-2]
epmtime = behaviordata[5]
epmspk = epmdata[2]
basetime = spikedata[1]
basespk = spikedata[2]
postbasetime = spikedata[5]
postbasespk = spikedata[6]

binnedEPM_spk = []; binnedEPM_time = []
binnedBase_spk = []; binnedBase_time = []
binnedpostBase_spk = []; binnedpostBase_time = []

for ianimal in range(len(basetime)):    
 
    # EPM task    
    tmpepmspk = binnedSpike(epmspk[ianimal], epmtime[ianimal], 0.005, 40)
    binnedEPM_spk.append(tmpepmspk)    
    binnedEPM_time.append(np.linspace(epmtime[ianimal][0],epmtime[ianimal][-1],
                                      len(epmtime[ianimal])*int((1/40)/0.005)))
    
    print("%s spike done" %(context_name)) 

    # Baseline task
    tmpbasespk = binnedSpike(basespk[ianimal],basetime[ianimal],0.005,40)
    binnedBase_spk.append(tmpbasespk)
    binnedBase_time.append(np.linspace(basetime[ianimal][0],basetime[ianimal][-1],
                                       len(basetime[ianimal])*int((1/40)/0.005)))
    
    print("Baseline spike done")
    if len(postbasetime[ianimal]) > 1:
        tmpbasespk = binnedSpike(postbasespk[ianimal],postbasetime[ianimal],0.005,40)
        binnedpostBase_spk.append(tmpbasespk)
        binnedpostBase_time.append(np.linspace(postbasetime[ianimal][0],postbasetime[ianimal][-1],
                                           len(postbasetime[ianimal])*int((1/40)/0.005)))
        
        print("postBaseline spike done")
    else:
        binnedpostBase_spk.append([])
        binnedpostBase_time.append([])
        print("postBaseline spike epmty")


f = open('EPM_binneddata_combine.pckl','wb')
pickle.dump([binnedEPM_spk,binnedEPM_time,binnedBase_spk,binnedBase_time,
             binnedpostBase_spk,binnedpostBase_time],f)
f.close()


f = open('EPM_binneddata_combine.pckl','rb')
binneddata = pickle.load(f)
f.close()
binnedEPM_spk = binneddata[0]
binnedEPM_time = binneddata[1]
binnedBase_spk = binneddata[2]
binnedBase_time = binneddata[3]
binnedpostBase_spk = binneddata[4]
binnedpostBase_time = binneddata[5]


from ripple_detection_master.ripple_detection import multiunit_HSE_detector

fs = 200
mua_closed = []; mua_rate_closed =[]; mua_open = []; mua_rate_open = []; mua_all =[]; mua_rate_all = []
mua_closed_post = []; mua_rate_closed_post =[]; mua_open_post = []; mua_rate_open_post = []; mua_all_post =[]; mua_rate_all_post = []
epm_closed = []; epm_open = []; epm_all = []; epm_rate_closed = []; epm_rate_open = []; epm_rate_all = []

speed_threshold = 200
close_event_threshold = 1
for ianimal in range(len(animalgaussfr)):
    tmpfr = animalgaussfr[ianimal]
    closedpeak = np.where(np.argmax(tmpfr,axis = 1) < 37)
    openpeak = np.where(np.argmax(tmpfr,axis = 1) > 39)       
        
    speed = np.repeat(epmspeed[ianimal], 5)
    speed = np.concatenate((speed, speed[-6:-1]))
    restingtime = sum(speed<4)*0.005
    
    epm_closed.append(multiunit_HSE_detector(binnedEPM_time[ianimal][0:len(speed)], np.transpose(binnedEPM_spk[ianimal][closedpeak,0:len(speed)][0]), 
                                      speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                      smoothing_sigma=0.015,close_event_threshold = 1))
    
    epm_rate_closed.append([0 if epm_closed[ianimal] is None else len(epm_closed[ianimal])]/restingtime)
    
    epm_open.append(multiunit_HSE_detector(binnedEPM_time[ianimal][0:len(speed)], np.transpose(binnedEPM_spk[ianimal][openpeak,0:len(speed)][0]), 
                                      speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                      smoothing_sigma=0.015, close_event_threshold = 1))
    epm_rate_open.append([0 if epm_open[ianimal] is None else len(epm_open[ianimal])]/restingtime)
    
    epm_all.append(multiunit_HSE_detector(binnedEPM_time[ianimal][0:len(speed)], np.transpose(binnedEPM_spk[ianimal][:,0:len(speed)]), 
                                      speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                      smoothing_sigma=0.015, close_event_threshold = 1))
    epm_rate_all.append([0 if epm_all[ianimal] is None else len(epm_all[ianimal])]/restingtime)
    print('done')
    
    if len(binnedpostBase_spk[ianimal]) > 1:
        
        # Pre base
        speed = np.repeat(basespeed[ianimal], 5)
        speed = np.concatenate((speed, speed[-6:-1]))
        restingtime = sum(speed<4)*0.005
        
        mua_closed.append(multiunit_HSE_detector(binnedBase_time[ianimal], np.transpose(binnedBase_spk[ianimal][closedpeak,:][0]), 
                                          speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                          smoothing_sigma=0.015, close_event_threshold = 1))
        
        mua_rate_closed.append([0 if mua_closed[ianimal] is None else len(mua_closed[ianimal])]/restingtime)
        
        mua_open.append(multiunit_HSE_detector(binnedBase_time[ianimal], np.transpose(binnedBase_spk[ianimal][openpeak,:][0]), 
                                          speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                          smoothing_sigma=0.015, close_event_threshold = 1))
        mua_rate_open.append([0 if mua_open[ianimal] is None else len(mua_open[ianimal])]/restingtime)
        
        mua_all.append(multiunit_HSE_detector(binnedBase_time[ianimal], np.transpose(binnedBase_spk[ianimal]), 
                                          speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                          smoothing_sigma=0.015, close_event_threshold = 1))
        mua_rate_all.append([0 if mua_all[ianimal] is None else len(mua_all[ianimal])]/restingtime)
        
        # Post base
        speed = np.repeat(postbasespeed[ianimal], 5)
        speed = np.concatenate((speed, speed[-6:-1]))
        restingtime = sum(speed<4)*0.005
        
        mua_closed_post.append(multiunit_HSE_detector(binnedpostBase_time[ianimal], np.transpose(binnedpostBase_spk[ianimal][closedpeak,:][0]), 
                                          speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                          smoothing_sigma=0.015, close_event_threshold = 1))
        mua_rate_closed_post.append([0 if mua_closed_post[ianimal] is None else len(mua_closed_post[ianimal])]/restingtime)
        
        mua_open_post.append(multiunit_HSE_detector(binnedpostBase_time[ianimal], np.transpose(binnedpostBase_spk[ianimal][openpeak,:][0]), 
                                          speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                          smoothing_sigma=0.015, close_event_threshold = 1))
        mua_rate_open_post.append([0 if mua_open_post[ianimal] is None else len(mua_open_post[ianimal])]/restingtime)
        
        mua_all_post.append(multiunit_HSE_detector(binnedpostBase_time[ianimal], np.transpose(binnedpostBase_spk[ianimal]), 
                                          speed,fs,speed_threshold, minimum_duration=0.015,zscore_threshold= 3.0, 
                                          smoothing_sigma=0.015, close_event_threshold = 1))
        mua_rate_all_post.append([0 if mua_all_post[ianimal] is None else len(mua_all_post[ianimal])]/restingtime)
    else:
        mua_closed.append([])
        mua_rate_closed.append([])
        mua_open.append([]) 
        mua_rate_open.append([])
        mua_all.append([])
        mua_rate_all.append([])
        mua_closed_post.append([])
        mua_rate_closed_post.append([])
        mua_open_post.append([])
        mua_rate_open_post.append([])
        mua_all_post.append([])
        mua_rate_all_post.append([])
        
# Raster plot 

animal_id = 1
timetemplate = np.random.choice(np.arange(np.shape(mua_all_post[animal_id])[0]),10)
for ind in timetemplate:
    tmpmua = mua_all_post[animal_id]
    tmptime = binnedpostBase_time[animal_id]
    tmpspk = binnedpostBase_spk[animal_id]
    
    startind = np.where(np.in1d(tmptime,tmpmua['start_time']))[0]
    endind = np.where(np.in1d(tmptime,tmpmua['end_time']))[0]
    mua_spk = tmpspk[:,startind[ind]:endind[ind]]
    time_temp = np.arange(np.shape(mua_spk)[1])*0.005
    a = mua_spk*time_temp
    a[a==0] = -1
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6,6))
    plt.eventplot(a,color = 'k',linelengths = .8,linewidth = 2)
    plt.xlim([0, np.max(time_temp)])
    plt.xlabel('Time(s)')


# LFP trace
   
f = open('EPM_cscdata_combine.pckl','rb')
cscdata = pickle.load(f)
f.close()

animal_id = 1
## EEG extraction
animalCsc = cscdata[0][animal_id]
postbasecsc = animalCsc['Postbaseline']
tmpepmcsc = animalCsc['EPM']

epmcsctrace = []
for icsc in range(len(tmpepmcsc)):
    tmpcsc = tmpepmcsc[icsc][0:512,:].reshape(1,tmpepmcsc[icsc][0:512,:].shape[0]*tmpepmcsc[icsc][0:512,:].shape[1],order = 'F')
    if icsc == 0:
        epmcsctrace = tmpcsc
    else:
        epmcsctrace = np.vstack([epmcsctrace,tmpcsc])
csctime = tmpepmcsc[0][512,:]      
tmpcsctrace = epmcsctrace[0,:]
fs = 32000
tmpcsctrace = epmcsctrace
l_freq = 100; h_freq = 300
# mne.filter.filte_data(data,sfreq,l_freq,h_freq)-FIR filter
filteredcsc = mne.filter.filter_data(tmpcsctrace, fs, l_freq, h_freq)
tmp_filtercsc = filteredcsc[0,:]
# Animal trajectory trace


f = open('EPM_trajectory_combine.pckl','rb')
traject_info = pickle.load(f)
f.close()

animal_pos_traject = traject_info[0]
animal_time_traject = traject_info[1]
animal_opendist = traject_info[2]
animal_scale = traject_info[-1]

tmpmua = epm_all[animal_id]
tmpspk = binnedEPM_spk[animal_id]
tmptime = binnedEPM_time[animal_id]
tmpfr = animalgaussfr[animal_id]
closedpeak = np.where(np.argmax(tmpfr,axis = 1) < 37)
openpeak = np.where(np.argmax(tmpfr,axis = 1) > 39)  

import matplotlib.pyplot as plt
startind = np.where(np.in1d(tmptime,tmpmua['start_time']))[0]
endind = np.where(np.in1d(tmptime,tmpmua['end_time']))[0]

cscstartind = []; cscendind = []
for ind in range(len(tmpmua['start_time'])):
    cscstartind.append(np.argmin(np.abs(csctime-tmpmua['start_time'].iloc[ind]))) 
    cscendind.append(np.argmin(np.abs(csctime-tmpmua['end_time'].iloc[ind])))

timetemplate = np.random.choice(np.arange(np.shape(tmpmua)[0]),10)

for ind in timetemplate:    
    mua_spk = tmpspk[:,startind[ind]:endind[ind]]
    time_temp = np.arange(np.shape(mua_spk)[1])*0.005
    a = mua_spk*time_temp
    a[a==0] = -1
    colors = ['k']*np.shape(mua_spk)[0]
    colors = np.array(colors)
    # colors[openpeak[0]] = 'black'
    fig = plt.figure(figsize = (10,6))
    ax_1=fig.add_subplot(223)
    ax_1.eventplot(a,color = colors, linelengths = .8, linewidth = 2)
    ax_1.set_xlim([0, np.max(time_temp)])
    ax_1.set_xlabel('Time(s)')
    ax_1.set_ylabel('Neuron ID')    
    gap = (cscendind[ind]-cscstartind[ind])*512
    tmp_csc = tmp_filtercsc[cscstartind[ind]:cscstartind[ind]+gap]
    tmp_pos = animal_pos_traject[animal_id]
    tmp_time = animal_time_traject[animal_id]
    
    tmpstartind = np.argmin(np.abs(tmp_time-tmpmua['start_time'][ind+1]))
    tmpendind = np.argmin(np.abs(tmp_time-tmpmua['end_time'][ind+1]))
    sortedZ = tmp_pos[tmpendind:tmpendind+400,:]
    restpos = tmp_pos[tmpstartind,:]
    
    opendist = animal_opendist[animal_id]
    scale = animal_scale[animal_id]
    
    width = 45
    mazearm = 40/scale
    maze_coord = [[-width/2,mazearm],[width/2,mazearm],[width/2,width/2],[opendist,width/2],
                  [opendist,-width/2],[width/2,-width/2],[width/2,-mazearm],[-width/2,-mazearm],
                  [-width/2,-width/2],[-opendist,-width/2],[-opendist,width/2],[-width/2,width/2],[-width/2,mazearm]]
    
    maze_coord = np.vstack(maze_coord)
    
    # Plot trajectory & Resting 
    ax_2 = fig.add_subplot(122)
    ax_2.plot(sortedZ[:,0],sortedZ[:,1],color = 'gray',linewidth = 1,alpha = .7, zorder = 0)
    ax_2.plot(maze_coord[:,0],maze_coord[:,1],color = 'black', linewidth = 3)
    
    ax_2.scatter(restpos[0],restpos[1],color = 'red',s =15)      
    ax_2.text(width/2+10,mazearm,'0', fontsize = 13)
    ax_2.text(width/2+10,-mazearm,'35', fontsize = 13)
    ax_2.text(-mazearm,width/2+10,'35', fontsize = 13)
    ax_2.text(mazearm,width/2+10,'70', fontsize = 13)
    ax_2.set_title('Animal Position')         
    ax_2.axis('off')
    
    ax_3 = fig.add_subplot(221)
    ax_3.plot(tmp_csc,color = 'gray',linewidth = 1,alpha = .7, zorder = 0)
    ax_3.set_title('Local field potential')         
    ax_3.axis('off')
        
    plt.show()

# MUA position

animal_pos_traject = traject_info[0]
animal_time_traject = traject_info[1]
animal_opendist = traject_info[2]
animal_scale = traject_info[-1]
width = 45


 # Plot trajectory & Resting 
for ianimal in range(len(animal_pos_traject)):       
    scale = animal_scale[ianimal]
    mazearm = 40/scale
    
    opendist = animal_opendist[ianimal]
    maze_coord = [[-width/2,mazearm],[width/2,mazearm],[width/2,width/2],[opendist,width/2],
                  [opendist,-width/2],[width/2,-width/2],[width/2,-mazearm],[-width/2,-mazearm],
                  [-width/2,-width/2],[-opendist,-width/2],[-opendist,width/2],[-width/2,width/2],[-width/2,mazearm]]
    maze_coord = np.vstack(maze_coord)

    
    tmp_pos = animal_pos_traject[ianimal]
    fig, ax = plt.subplots(figsize = (4,4))    
    ax.plot(animal_pos_traject[ianimal][:,0],animal_pos_traject[ianimal][:,1],color = 'gray',linewidth = 1,alpha = .7, zorder = 0)
    ax.plot(maze_coord[:,0],maze_coord[:,1],color = 'black', linewidth = 3)
    plt.text(width/2+10,mazearm,'0', fontsize = 13)
    plt.text(width/2+10,-mazearm,'35', fontsize = 13)
    plt.text(-mazearm,width/2+10,'35', fontsize = 13)
    plt.text(mazearm,width/2+10,'70', fontsize = 13)
    plt.title('Animal Trajectory & Resting')
    tmp_time = animal_time_traject[ianimal]
    centermua = []; closedmua = []; openmua = []
    for iblob in range(len(epm_all[ianimal])):  
        tmpstartind = np.argmin(np.abs(tmp_time-epm_all[ianimal]['start_time'][iblob+1]))
        ax.scatter(tmp_pos[tmpstartind,0],tmp_pos[tmpstartind,1],color = 'orangered',s =5)    
               
    
    ax.axis('off')
    plt.show()
    
# Data load (position, time)
f = open('EPM_Linearize_combine.pckl','rb')
pos_time_data = pickle.load(f)
f.close()

preferclosedtime = pos_time_data[11]; otherclosedtime = pos_time_data[14]
preferopentime = pos_time_data[17]; otheropentime = pos_time_data[20]
centertime = pos_time_data[23]
  
animal_closed = []; animal_open = []; animal_center = []
for ianimal in range(len(animal_pos_traject)):         
    centermua = []; closedmua = []; openmua = []
    tmp_pos = animal_pos_traject[ianimal]
    tmp_time = animal_time_traject[ianimal]
    
    closedtime = (len(preferclosedtime[ianimal])+len(otherclosedtime[ianimal]))*0.025
    opentime = (len(preferopentime[ianimal])+len(otheropentime[ianimal]))*0.025
    
    for iblob in range(len(epm_all[ianimal])):  
        tmpstartind = np.argmin(np.abs(tmp_time-epm_all[ianimal]['start_time'][iblob+1]))  
        if (tmp_pos[tmpstartind,0] < width/2) & (tmp_pos[tmpstartind,0] > -width/2):
            if (tmp_pos[tmpstartind,1] < width/2) & (tmp_pos[tmpstartind,1] > -width/2):
                centermua.append(iblob)
            else:
                closedmua.append(iblob)
        else:
            openmua.append(iblob)
            
    animal_closed.append(len(closedmua)/closedtime)
    animal_open.append(len(openmua)/closedtime)
    animal_center.append(centermua)
    
import pandas as pd
MUA_rate =pd.DataFrame();
MUA_rate = MUA_rate.append(pd.DataFrame({'Arm':'Closed_arm','MUA rate' : animal_closed}))
MUA_rate = MUA_rate.append(pd.DataFrame({'Arm':'Open_arm','MUA rate' : animal_open}))

import seaborn as sns, matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
sns.set(style = "white",font_scale = 1.6)
clrs = ['paleturquoise','lightsalmon','turquoise','salmon']
ax = sns.barplot(x='Arm',y='MUA rate',dodge=False,data = MUA_rate,
                 palette = clrs, capsize = .1, ci = "sd")
sns.swarmplot(x='Arm',y='MUA rate', data = MUA_rate, color = "0",alpha = .8)
for iline in range(len(animal_closed)):
    tmpdf = MUA_rate.loc[iline]
    sns.lineplot(x = 'Arm',y = 'MUA rate',data = tmpdf,color = 'gray')
plt.ylabel('MUA rate (event/s)')
plt.xlabel('')
plt.title('Multiunit analysis')
plt.savefig('Multiunit analysis_rate')

fig = plt.figure(figsize=(10,8))

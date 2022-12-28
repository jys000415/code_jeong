# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:15:38 2020
LFP data analysis
@author: yeong
"""
import os
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt

## Data load
animalpath = ['J:/Jeong Yeongseok/Project_spatial_coding/Recording/022148_200823/2020-09-29_09-36-32-022148-Base-EPM-OF',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/022346_201116/022346-201119-Base-EPM-OF-Base',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/022347-201116/022347-201124-Base-EPM-OF-Base',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/022348-201116/022348-20201124-Base-EPM-OF-Base',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/022880-201223/022880-Base-EPM-OF-Base-210111',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023274/023274-20210325-Base-EPM-2ndEPM-OF-Base',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023275/023275-20210325-Base_EPM_2ndEPM-OF-Base',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023277/023277-210329-Base-EPM-2ndEPM-OF-Base',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023697/023697-210528-Base-1stEPM-2ndEPM',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023698/023698-20210528-Base-1stEPM-2ndEPM-postBase',
            'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023699/023699-20210528-Base-1stEPM-2ndEPM-postEPM']
remap = 1   

f = open('figure1_EPM_combine.pckl','rb')
behaviordata = pickle.load(f)
f.close()
speeddata = behaviordata[6]

animalcscdata = []
for animal_id in range(len(animalpath)):
    os.chdir(animalpath[animal_id])      
    speed = speeddata[animal_id]
    
    from scipy.io import loadmat
    data = loadmat('animal1.mat')    
    whichcscs = np.where(data['countOfcell'][0] >= 2)
    cellorder = np.argsort(-1*data['countOfcell'][0][data['countOfcell'][0] >= 2])
    import glob
    cscname = []
    for name in glob.glob('CSC?.mat'):
        cscname.append(name)
    cscs = []; chname =[]
    for ind in cellorder:
        chname.append(cscname[whichcscs[0][ind]])
        tmpcsc = loadmat(cscname[whichcscs[0][ind]])
        tmpcsc = tmpcsc['cscdata']
        cscs.append(tmpcsc)
      
    if animal_id > 4:
        context_name = ['Baseline','Openfield','EPM_1','EPM_2','Postbaseline']
    else:
        context_name = ['Baseline','Openfield','EPM','EZM','Postbaseline']
    context_order = data['context_order'][0]-1
    contextind = [i for i in range(len(context_order)) if context_order[i]<255]
    asscontextind = map(context_name.__getitem__, contextind)
    assorderind = map(context_order.__getitem__, contextind)
    incontextorder = list(assorderind)
    incontextname = list(asscontextind)
    taskname =[x for _,x in sorted(zip(incontextorder,incontextname))]
    event = data['Events']['Tasktime'][0].tolist()
    
    
    import pandas as pd
    dftask = pd.DataFrame(data = np.divide(event[0][0:len(taskname)],10**6),index = taskname,columns=['Start','Stop'])
    cscinfo = {'Time':np.divide(data['csctime'][0],10**6),
               'Frequency':data['samplefreq'][0],'Samplenum':data['numofvalidsample'][0]}
    # Divide a poisition,time, spkdata and csc data to each context
    
    animalCsc = {};
    for tname in taskname:
        csctime = cscinfo['Time'][(cscinfo['Time'] >= dftask.loc[tname,'Start']) &
                                  (cscinfo['Time'] <= dftask.loc[tname,'Stop'])]
        tmpcsc = []
        for icsc in cscs:
            tmpcsc.append(np.vstack([icsc[:,(cscinfo['Time'] >= dftask.loc[tname,'Start']) &
                                             (cscinfo['Time'] <= dftask.loc[tname,'Stop'])],csctime]))
        animalCsc[tname] = tmpcsc
    animalcscdata.append(animalCsc)
   
f = open('EPM_cscdata_combine.pckl','wb')
pickle.dump([animalcscdata],f)
f.close()

## EEG extraction

postbasecsc = animalCsc['Postbaseline']
tmpepmcsc = animalCsc['EPM_1']

epmcsctrace = []
for icsc in range(len(tmpepmcsc)):
    tmpcsc = tmpepmcsc[icsc][0:512,:].reshape(1,tmpepmcsc[icsc][0:512,:].shape[0]*tmpepmcsc[icsc][0:512,:].shape[1],order = 'F')
    if icsc == 0:
        epmcsctrace = tmpcsc
    else:
        epmcsctrace = np.vstack([epmcsctrace,tmpcsc])
csctime = tmpepmcsc[0][512,:]      

sfreq = 32000
data = epmcsctrace/10**6
ch_type = ['eeg']*len(chname)
ch_names = list(map(lambda x: x.replace('.mat',''),chname))

info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = ch_type)
raw = mne.io.RawArray(data,info)
raw.crop(0,300).load_data()


# Artifact detection
# save the data before projection

ssp_projectors = raw.info['projs']
raw.del_proj()
fig = raw.plot_psd(tmax = np.inf, fmax = 250, average=True)

# Filtered LFP
fs = 32000
tmpcsctrace = epmcsctrace
l_freq = 100; h_freq = 300
# mne.filter.filte_data(data,sfreq,l_freq,h_freq)-FIR filter
filteredcsc = mne.filter.filter_data(tmpcsctrace, fs, l_freq, h_freq)

from ripple_detection.simulate import simulate_time, brown
from ripple_detection import filter_ripple_band
from ripple_detection.simulate import simulate_LFP
from ripple_detection import Karlsson_ripple_detector,Kay_ripple_detector



time = simulate_time(fs*3, fs)
true_ripple_midtime = [1.1, 2.1]
RIPPLE_DURATION = 0.100
lfps = simulate_LFP(time, true_ripple_midtime,
                    noise_amplitude=1.2,
                    ripple_amplitude=1.5)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15, 3))
plt.plot(time, lfps)

for midtime in true_ripple_midtime:
    plt.axvspan(midtime - RIPPLE_DURATION/2,
                midtime + RIPPLE_DURATION/2, alpha=0.3, color='green', zorder=1000)
filtered_lfps = mne.filter.filter_data(np.transpose(lfps), fs, l_freq, h_freq)

speed = np.ones_like(time)
Karlsson_ripple_times = Karlsson_ripple_detector(time, np.transpose(filtered_lfps), speed, fs)

for midtime in true_ripple_midtime:
    ax.axvspan(midtime - RIPPLE_DURATION/2, midtime + RIPPLE_DURATION/2, alpha=0.3, color='green', zorder=9)
    
for ripple in Karlsson_ripple_times.itertuples():
    ax.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=10)

fig, ax = plt.subplots(figsize=(15, 3))
ax.plot(time, np.transpose(filtered_lfps))


ripple_times = Kay_ripple_detector(time, np.transpose(filtered_lfps), speed,fs,speed_threshold = 4.0, minimum_duration = 0.01, 
                                   zscore_threshold = 2.0, smoothing_sigma = 0.004, close_ripple_threshold = 0.0)
for ripple in ripple_times.itertuples():
    ax.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000)


# Make speed dataset
tmpspeed = speeddata[animal_id]
start_time = 300
end_time = 320
fs = 32000
speed = np.repeat(tmpspeed, 800)
newcsctime = list(np.repeat(csctime,512))

newcsctime = np.array(newcsctime[0:len(speed)])

tmpfilteredcsc = tmpcsctrace[:,0:len(speed)]
filtered_lfps = mne.filter.filter_data(tmpfilteredcsc, fs, l_freq, h_freq)

newfilteredcsc = np.transpose(filtered_lfps[0:2,start_time*fs:end_time*fs])

fig,ax = plt.subplots(figsize=(15,3))
plt.plot(filtered_lfps[5,250000:350000])

fig,ax = plt.subplots(figsize=(15,3))
plt.plot(tmpcsctrace[5,250000:350000])

ripple_times = Kay_ripple_detector(newcsctime[start_time*fs:end_time*fs], newfilteredcsc, speed[start_time*fs:end_time*fs],fs,speed_threshold = 4.0, minimum_duration = 0.01, 
                                   zscore_threshold = 2.0, smoothing_sigma = 0.004, close_ripple_threshold = 0.0)

display(ripple_times)

fig, ax = plt.subplots(figsize = (15,3))
plt.plot(newfilteredcsc[:,0])

import scipy.stats as stat

a = stat.zscore(newfilteredcsc[:,0])

sum(a>0)


eeg_channels = mne.pick_types(raw.info,eeg=True)
raw.plot(duration=60, order=eeg_channels, n_channels=len(eeg_channels),remove_dc=False, scalings = 'auto')

raw_highpass = raw.copy().filter(l_freq = 0.2, h_freq = None)
fig = raw_highpass.plot(duration = 60, order = eeg_channels, proj = False, n_channels = len(eeg_channels), 
                        scalings = 'auto',remove_dc = False)

# Power line artifacts
fig = raw.plot_psd(tmax = np.inf, fmax = 250, average=True)


filteredraw = mne.io.RawArray(filteredcsc,info)
filteredraw.plot(duration = 60, order = eeg_channels, proj = False, n_channels = len(eeg_channels),scalings = 'auto', remove_dc = False)

from numpy.fft import fft, fftfreq
from scipy import signal
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter

sfreq = 1000
f_p = 40
flim = (1,sfreq /2)

n = int(round(.1 * sfreq))
n -= n%2 - 1
t = np.arange(-(n//2), n// 2+1) / sfreq
h = np.sinc(2*200*t)/(4*np.pi)
freq = [0, 200, 200, 100]
gain = [1, 1, 0, 0]
plot_filter(h, sfreq,freq,gain,'Sinc(1s)',flim=flim,compensate = True)
# raw.plot(show_scrollbars = False, show_scalebars = False)
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1.,h_freq=None)


#filtering
# ICA filter
ica = mne.preprocessing.ICA(n_components = len(chname), random_state = 97, max_iter = 200)
ica.fit(filt_raw)

raw.load_data()
ica.plot_sources(raw,show_scrollbars = False)
ica.exclude = [1,2]
# ica.plot_properties(raw,picks = ica.exclude)

raw.load_data()
ica.apply(raw)
chan_idxs = [raw.ch_names.index(ch) for ch in ch_names]
orig_raw.plot(order=chan_idxs,start = 100, duration = 60, scalings = 'auto')
raw.plot(order=chan_idxs,start = 100, duration = 60, scalings = 'auto')

raw.plot_psd(fmin = 0,fmax = 300,average = True,tmin = None, tmax = None)

orig_raw.plot_psd(fmin = 0,fmax = 300,average = True,tmin = None, tmax = None)


l_freq = 100; h_freq = 200
# mne.filter.filte_data(data,sfreq,l_freq,h_freq)-FIR filter
filteredcsc = mne.filter.filter_data(data, sfreq, l_freq, h_freq)
filterdata = mne.io.RawArray(filteredcsc, info)
scalings = 'auto'
filterdata.plot(scalings = scalings)

raw_base.plot_psd(fmin = 0,fmax = 300,average = True,tmin = None, tmax = None)

rawpath = 'J:/Jeong Yeongseok/Recording/Project1'
os.chdir(rawpath) 


f = open('Blob_EPM.pckl','rb')
blobdata = pickle.load(f)
openBlob = blobdata[-4]
closeBlob = blobdata[-3] 
runBlob = blobdata[-2]
restBlob = blobdata[-1]
f.close()

csctime = np.linspace(csctime[0],csctime[-1],num=len(epmcsctrace[0,:]))
runidx = [[(np.abs(runBlob[animal_id][idx][0]-csctime)).argmin(), 
           (np.abs(runBlob[animal_id][idx][-1]-csctime)).argmin()] for idx in range(len(runBlob[animal_id]))]
restidx = [[(np.abs(restBlob[animal_id][idx][0]-csctime)).argmin(), 
            (np.abs(restBlob[animal_id][idx][-1]-csctime)).argmin()] for idx in range(len(restBlob[animal_id]))]
closedidx = [[(np.abs(closeBlob[animal_id][idx][0]-csctime)).argmin(), 
           (np.abs(closeBlob[animal_id][idx][-1]-csctime)).argmin()] for idx in range(len(closeBlob[animal_id]))]
openidx = [[(np.abs(openBlob[animal_id][idx][0]-csctime)).argmin(), 
            (np.abs(openBlob[animal_id][idx][-1]-csctime)).argmin()] for idx in range(len(openBlob[animal_id]))]


events = list()
events = [[closeBlob[animal_id][idx][0], 0, 1] for idx in range(len(closeBlob[animal_id]))]
events.extend([[openBlob[animal_id][idx][0], 0, 2] for idx in range(len(openBlob[animal_id]))])
events.extend([[runBlob[animal_id][idx][0], 0, 3] for idx in range(len(runBlob[animal_id]))])
events.extend([[restBlob[animal_id][idx][0], 0, 4] for idx in range(len(restBlob[animal_id]))])

events = list()
events = [[closedidx[idx][0], 0, 1] for idx in range(len(closeBlob[animal_id]))]
events.extend([[openidx[idx][0], 0, 2] for idx in range(len(openBlob[animal_id]))])
events.extend([runidx[idx][0], 0, 3] for idx in range(len(runBlob[animal_id])))
events.extend([[restidx[idx][0], 0, 4] for idx in range(len(restBlob[animal_id]))])

event_array = np.array(events,dtype=int)
event_dict = {'Closed':1,'Open':2,'Running':3, 'Resting':4}

fig = mne.viz.plot_events(event_array, event_id = event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)
# reject_criteria = dict(eeg=1000e-6)
epochs = mne.Epochs(raw,event_array, event_id=event_dict,event_repeated = 'merge')

run_epochs = epochs['Running']
rest_epochs = epochs['Resting']
closed_epochs = epochs['Closed']
open_epochs = epochs['Open']

rest_epochs.plot_image()
run_epochs.plot_image()
closed_epochs.plot_image()
open_epochs.plot_image()

frequencies = np.arange(100,250,50)
power = mne.time_frequency.tfr_morlet(rest_epochs, n_cycles = 2, return_itc = False, freqs = frequencies, decim = 3)
power.plot()

rest_epochs.plot_psd(fmin=2., fmax=300., average=True, spatial_colors=False)







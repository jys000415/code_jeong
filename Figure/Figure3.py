# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:28:27 2020

@author: yeong
"""
# pickle.dump([0-openpos,1-opentime,2-openspk,3-opencsc,basepos, basetime, basespk, basecsc,
#             epmpos, epmtime, epmspk, epmcsc, cornerpos, cornertime, cornerspk,
#            centerpos, centertime, centerspk, openArmpos, openArmtime, openArmspk,
#            closeArmpos, closeArmtime, closeArmspk],f)

import os 
rawpath = 'J:/Jeong Yeongseok/Recording/Project1'
os.chdir(rawpath) 
import numpy as np
import pickle
import matplotlib.pyplot as plt
f = open('Project1_csc_EPM.pckl','rb')
cscdata = pickle.load(f)
f.close()

## EEG analysis
import mne

# EEG trace data assign
## EEG extraction

epmcsctrace = cscdata[0][2][0:512,:]

epmcsctrace = epmcsctrace.reshape((1,epmcsctrace.shape[0]*epmcsctrace.shape[1]),order = 'F')

sfreq = 32000
data = epmcsctrace/10**6
ch_type = ['eeg']
ch_names = ['epmcsc']

info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = ch_type)
raw = mne.io.RawArray(data,info)
scalings = 'auto'

filteredcsc = mne.filter.filter_data(data,32000,2,300)
filterdata = mne.io.RawArray(filteredcsc,info)

filterdata.plot(scalings = scalings)

# Power spectral density
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

# Extract index of csctime
csctime = np.arange(len(epmcsctrace[0]))*1/32000+project1[11][1][512,0]

centercsc = list()
for itime in range(len(centertime)):
   tmp = np.where((centertime[itime] <= csctime) & (centertime[itime]+0.025 >= csctime))[0]
   centercsc.append(tmp)
cornercsc = list()
for itime in range(len(cornertime)):
   tmp = np.where((cornertime[itime] <= csctime) & (cornertime[itime]+0.025 >= csctime))[0]
   cornercsc.append(tmp)
from etcfunc import blobassign   
centerblob = blobassign(centercsc,1,32000)
cornerblob = blobassign(cornercsc,1,32000)

centerblob = centerBlob[1]; cornerblob = cornerBlob[1]
events = list()
events = [[np.abs(csctime-centerblob[idx][0]).argmin(), 0, 1] for idx in range(len(centerblob))]
events.extend([[np.abs(csctime-cornerblob[idx][0]).argmin(), 0, 2] for idx in range(len(cornerblob))])

event_dict = {'Center':1,'Corner':2}
event_id, tmin, tmax = 1, -2, 2
baseline = (None,0)
epochs = mne.Epochs(raw,events,event_id, tmin, tmax, baseline=baseline, preload=True)
epochs.resample(200., npad = 'auto')

fmin, fmax = 2,300 # look at frequencies between 2 and 300Hz
n_fft = 2**14 # the FFT size (n_fft). Ideally a power of 2
plt.figure()
ax = plt.axes()
filterdata.plot_psd(fmin = fmin, fmax = fmax, n_fft = n_fft,
             n_jobs = 1, proj = False, ax = ax, color = (0,0,1), show=False,average =True)
plt.xlim((0,300))
plt.ylim((-10,60))
plt.savefig('EEGinEPM-3')






raw.plot(n_channels = 1, scalings = scalings, title = 'Data from arrays',
         show = True, block = False)

'''
# Preprocessing
orig_raw = raw.copy()
raw = mne.io.RawArray(data,info)
ica = mne.preprocessing.ICA(n_components = 1, random_state=97, max_iter = 800)
ica.apply(raw)
orig_raw.plot()
raw.plot()
'''


# Event store - Event is numpy [sample number, 0, event ID] // 1-Center, 2-Corner, 3-Running, 4-Resting
events =[]
events = [[centerblob[idx][0], 0, 1] for idx in range(len(centerblob))]
events.extend([[cornerblob[idx][0], 0, 2] for idx in range(len(cornerblob))])

event_dict = {'Center':1,'Corner':2}
fig = mne.viz.plot_events(events, event_id = event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)

epochs = mne.Epochs(raw,events, event_id=event_dict, tmin = -2, tmax = 2)
run_epochs = epochs['Running']
rest_epochs = epochs['Resting']

rest_epochs.plot_image()
run_epochs.plot_image()
# frequency = np.arange(100,300,5)
frequency = np.logspace(*np.log10([10,300]), num=20)
n_cycles = frequency / 2.
power = mne.time_frequency.tfr_morlet(rest_epochs, n_cycles = n_cycles, 
                                      return_itc = False, freqs= frequency, decim = 3)
power.plot(['opencsc'])
power.plot(baseline = (-2,-1.5), mode = 'logratio')
raw.Epochs(raw,events,event_id = event_dict )

# Define annotations
onset = [item[0]*1/32000 for item in restidx]
duration = [(item[-1]-item[0])*1/32000 for item in restidx]
description = ['Resting']*len(restblob)
annot = mne.Annotations(onset, duration, description, orig_time = None)
annotated_center_raw = raw.copy()
annotated_center_raw.set_annotations(annot)
annotated_center_raw.plot()

# Filtered raw data 
filteredcsc = mne.filter.filter_data(data,32000,0.5,300)
filterdata = mne.io.RawArray(filteredcsc,info)

filterdata.plot(scalings = scalings)

# Power spectral density
fmin, fmax = 0.5,300 # look at frequencies between 2 and 300Hz
n_fft = 2**15 # the FFT size (n_fft). Ideally a power of 2
plt.figure()
ax = plt.axes()
filterdata.plot_psd(fmin = fmin, fmax = fmax, n_fft = n_fft,
             n_jobs = 1, proj = False, ax = ax, color = (0,0,1), show=False,average =True)
plt.xlim((0,300))
plt.ylim((-10,60))

"""


"""
# General imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap

# Import some NeuroDSP functions to use with MNE
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.rhythm import compute_lagged_coherence

# Import NeuroDSP plotting functions
from neurodsp.plts import (plot_time_series, plot_power_spectra,
                           plot_bursts, plot_lagged_coherence)

# Settings for exploring an example channel of data
ch_label = 'eeg'
t_start = (1000 * fs)
t_stop = int(t_start + (2 * fs))
sig, times = raw.get_data(start=t_start, stop=t_stop, return_times=True)
sig = np.squeeze(sig)

plot_time_series(times, sig)
freqs, powers = compute_spectrum(sig, fs, method='welch', avg_type='median')
freqs, powers = trim_spectrum(freqs, powers, [60, 300])
peak_cf = freqs[np.argmax(powers)]
print(peak_cf)
plot_power_spectra(freqs, powers)
plt.plot(freqs[np.argmax(powers)], np.max(powers), '.r', ms=12)
amp_dual_thresh = (1., 1.5)
f_range = (peak_cf-2, peak_cf+2)
bursting = detect_bursts_dual_threshold(sig, fs, amp_dual_thresh, f_range)
plot_bursts(times, sig, bursting, labels=['Raw Data', 'Detected Bursts'])




# Time-frequency spectrogram - Context, Zone






# Power spectrogram desity - Context, Zone







# Filtering - Band pass filter








# EEG rate - Across the time







# Sequential activity - combine with single activity (across time)






# Plotting
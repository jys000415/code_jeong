# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:21:45 2022

@author: yeong
"""

import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

datapath = 'J:/Jeong Yeongseok/Project_spatial_coding/Recording/'
regionpath = 'BLA-pIC/'
optopath = 'ChR2/'
animalID = '001500/'
taskname = '001500-220218-optotagging'
contextname = 'EPM'
pathinfo = datapath+regionpath+optopath+animalID+taskname


recording = se.read_neuralynx(pathinfo)
channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()
num_seg = recording.get_num_segments()

recording_cmr = recording
recording_f = st.bandpass_filter(recording, freq_min=300, freq_max=6000)
print(recording_f)
recording_cmr = st.common_reference(recording_f, reference='global', operator='median')
print(recording_cmr)

# this computes and saves the recording after applying the preprocessing chain
recording_preprocessed = recording_cmr.save(format='binary')
print(recording_preprocessed)

print('Available sorters', ss.available_sorters())
print('Installed sorters', ss.installed_sorters())
print(ss.get_default_params('mountainsort4'))
sorting_HS = ss.run_mountainsort4(recording=recording_preprocessed, detect_threshold=4)
print(sorting_HS)
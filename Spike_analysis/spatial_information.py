# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:41:34 2022

@author: yeong
"""
import numpy as np
import random
from Spike_analysis.assign_combined import *
import sys
import time
import threading

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        time.sleep(0.1)
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            print("\b\b\b\r{}".format(next(self.spinner_generator)),end="")
            # sys.stdout.write(next(self.spinner_generator))
            # sys.stdout.flush()
            time.sleep(self.delay)
            # sys.stdout.write('\b')
            # sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        sys.stdout.write('\b')
        if exception is not None:
            return False
        

def cal_spatial_info(combined_Fr, combined_time):
    spatialinfo = []
    binnum =np.shape(combined_Fr)[1]
    avgFr = np.sum(combined_Fr, axis=1)/binnum
    for ibin in range(binnum):
        with np.errstate(divide='ignore', invalid='ignore'):
            tmpinfo = combined_Fr[:, ibin]/avgFr * \
                np.log(combined_Fr[:, ibin]/avgFr)*combined_time[ibin]
        tmpinfo[np.isnan(tmpinfo)] = 0
        tmpinfo[np.isinf(tmpinfo)] = 0
        spatialinfo.append(tmpinfo)
    spatial_Info = sum(spatialinfo)
    return spatial_Info


def shuffle_place(linear, lineartime, spk,
                  shuffle_num, num_bin, binsize, sigma):
    shuffle_spatial_info = []
    counter = 0
    # with Spinner():
    print('Calculate shuffling dataset: ')
    for ishuffle in range(shuffle_num):
        shufflespk = {}
        for idx, icell in enumerate(spk):
            if len(spk[icell]) > 0:
                tmpdiff = np.diff(spk[icell])
                random.shuffle(tmpdiff)
                tmpisi = np.cumsum(tmpdiff)
                shufflespk[icell] = np.append(
                    spk[icell][0], (spk[icell][0])+tmpisi)
            else:
                shufflespk[icell] = []
        linear_spk, linear_fr, linear_time = assign_spike(
            linear, lineartime, num_bin, binsize,
            shufflespk, sigma)
        combined_fr = combine_dataset(linear_fr)
        combined_time = combine_dataset(linear_time)
        spatial_Info = cal_spatial_info(combined_fr, combined_time)
        shuffle_spatial_info.append(spatial_Info)
    
        counter += 1 
        if counter == 49:
            print('Calculate shuffling dataset: Round %d' % (ishuffle+1))
            counter = 0
    return shuffle_spatial_info


def spatial_info_variable(origin_shuffle_spatial_info, shuffle_spatial_info):
    shuffle_array = np.array(shuffle_spatial_info)
    mean_shuffle = sum(shuffle_spatial_info)/len(shuffle_array)
    std_shuffle = np.std(shuffle_array, axis=0)
    z_origin = (origin_shuffle_spatial_info-mean_shuffle)/std_shuffle
    sig_ind = np.where(z_origin > 1.96)
    return shuffle_array, sig_ind

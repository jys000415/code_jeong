# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:26:07 2022

@author: yeong
"""
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
import weakref
# from memory_profiler import profile
import gc

# @profile
def assign_spatial_spk(linearized_dist, linearized_time, num_spatial_bin, binsize, spk, sigma):

    # Assign position and time according to Spatial bin (2cm) - EPM
    binned_time = np.zeros(int(num_spatial_bin/binsize))
    binnum = int(num_spatial_bin/binsize)
    bins = np.linspace(0, num_spatial_bin, binnum, endpoint=False)
    tmp_binning_Linear = np.digitize(linearized_dist, bins)
    tmptime = linearized_time
    # binned_time = [sum(tmp_binning_Linear == ibin+1)*0.025
    #                for ibin in range(0, binnum)]
    for ibin in range(0, binnum):
        binned_time[ibin] = sum(
            tmp_binning_Linear == ibin+1)*0.025
    # timebin = []
    arr = np.array(tmptime)
    a = min(np.diff(arr))
    # for i in tmptime:
    #     timebin.append((i, i+a))
    timebin = [(i, i+a) for i in tmptime]
    tmpbin = pd.IntervalIndex.from_tuples(timebin)
    tmpspk = spk
    binned_Fr = np.zeros((len(tmpspk.keys()), binnum))
    binned_Spk = np.zeros((len(spk.keys()), binnum))
    np.seterr(divide='ignore', invalid='ignore')
    for idx, (icell, _) in enumerate(tmpspk.items()):
        df = pd.DataFrame(tmpspk[icell], columns=['spktime'])
        if not df.empty: 
            spkAssign = pd.Series(df.groupby(pd.cut(df['spktime'], tmpbin)).size()).values   
            binned_Spk[idx, :] = [sum(spkAssign[tmp_binning_Linear == ibin+1]) 
                                  for ibin in range(0, binnum)]
            binned_Fr[idx, :] = [sum(spkAssign[tmp_binning_Linear == ibin+1])/(sum(tmp_binning_Linear == ibin+1)*0.025)
                                  for ibin in range(0, binnum)]
            binned_Fr[np.isnan(binned_Fr)] = 0
            # for ibin in range(0, binnum):
            #     if sum(tmp_binning_Linear == ibin+1):
            #         binned_Spk[idx, ibin] = sum(
            #             spkAssign[tmp_binning_Linear == ibin+1])
            #         binned_Fr[idx, ibin] = (sum(
            #             spkAssign[tmp_binning_Linear == ibin+1])/(sum(tmp_binning_Linear == ibin+1)*0.025))
            # gc.collect()
            del df, spkAssign
    binned_Fr = gaussian_filter1d(binned_Fr, sigma)
    gc.collect()
    return binned_Spk, binned_Fr, binned_time


def assign_spike(linear, time, num_bin, binsize, spk, sigma):
    assignspk = list(range(len(num_bin)))
    assignfr = list(range(len(num_bin)))
    assigntime = list(range(len(num_bin)))
    for itask in range(len(num_bin)):
        tmpspk, tmpfr, tmptime = assign_spatial_spk(
            linear[itask], time[itask], num_bin[itask],
            binsize, spk, sigma)
        assignspk[itask] = tmpspk
        assignfr[itask] = tmpfr
        assigntime[itask] = tmptime
        # weakref.ref(assign_spatial_spk)
    return assignspk, assignfr, assigntime


def find_maze_order(assigntime):
    closedtime = [len(x) for x in assigntime[0:2]]
    closed_order = np.argsort(closedtime[::-1]).tolist()
    opentime = [len(x) for x in assigntime[-2::]]
    open_order = np.argsort(opentime[::-1]).tolist()
    open_order = [x-2 for x in open_order]
    maze_order = [closed_order, open_order]
    return maze_order


def reorder_dataset(dataset, maze_order):
    reorder_dataset = deepcopy(dataset)
    reorder_dataset[0], reorder_dataset[1] = reorder_dataset[maze_order[0]
                                                             [0]], reorder_dataset[maze_order[0][1]]
    reorder_dataset[-2], reorder_dataset[-1] = reorder_dataset[maze_order[1]
                                                               [0]], reorder_dataset[maze_order[1][1]]
    return reorder_dataset


def combine_dataset(combined_dataset):
    if len(combined_dataset[0].shape) < 2:
        combined_dataset = np.concatenate(combined_dataset)
    else:
        combined_dataset = np.concatenate(combined_dataset, axis=1)
    return combined_dataset

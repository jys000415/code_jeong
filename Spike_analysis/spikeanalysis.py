# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:47:25 2021

@author: yeong
"""

# Spike with binning window


def binnedSpike(spk, time, binsize, fs):
    import numpy as np
    tmpspk = np.zeros((len(spk.keys()), len(time)*int((1/fs)/binsize)))
    cellkey = list(spk)
    for itime, icell, ibin in [(itime, icell, ibin) for itime in range(len(time))
                               for icell in range(len(cellkey)) for ibin in range(5)]:
        if any((time[itime]+binsize*ibin <= spk[cellkey[icell]]) & (spk[cellkey[icell]] <= time[itime] + binsize*(ibin+1))):
            if sum((time[itime]+binsize*ibin <= spk[cellkey[icell]]) & (spk[cellkey[icell]] <= time[itime] + binsize*(ibin+1)) > 1):
                print('Warning! there are more than 2 spikes in a timebin')
            tmpspk[icell, int((1/fs)/binsize)*itime+ibin] = 1
    return tmpspk


# Calculate mean firing rate
def cal_mean_spk(spk, time, duration):
    import numpy as np
    meanfr = []
    for ianimal in range(len(spk)):
        for key in spk[ianimal].keys():
            tmp = spk[ianimal][key]
            tmp = tmp[np.where(tmp <= time[ianimal][0] +
                               duration*60)[0].tolist()]
            meanfr.append(len(tmp)/(duration*60))
    return meanfr


def cal_mean_spk_zone(spk, time, fs):
    meanfr = []
    for ianimal in range(len(spk)):
        for i in range(len(spk[ianimal])):
            tmp = spk[ianimal][i]
            meanfr.append(sum(tmp)/(len(time[0])*1/fs))
    return meanfr


def assign_spatial_bin(linearized_dist, linearized_time, num_spatial_bin, binsize, spk, sigma):

    # Assign position and time according to Spatial bin (2cm) - EPM
    import pandas as pd
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    binned_Fr_Animal = []
    binned_Spk_Animal = []
    binned_time = np.zeros(
        (len(linearized_dist), int(num_spatial_bin/binsize)))

    for ianimal in range(len(linearized_dist)):
        binnum = int(num_spatial_bin/binsize)
        bins = np.linspace(0, num_spatial_bin, binnum, endpoint=False)
        tmp_binning_Linear = np.digitize(linearized_dist[ianimal], bins)
        tmptime = linearized_time[ianimal]

        for ibin in range(0, binnum):
            binned_time[ianimal, ibin] = sum(
                tmp_binning_Linear == ibin+1)*0.025

        timebin = []
        arr = np.array(tmptime)
        a = min(np.diff(arr))
        for i in tmptime:
            timebin.append((i, i+a))
        tmpbin = pd.IntervalIndex.from_tuples(timebin)

        tmpspk = spk[ianimal]

        tmpfr = np.zeros((len(tmpspk.keys()), binnum))
        tmp_binned_spk = np.zeros((len(spk[ianimal].keys()), binnum))

        for idx, (icell, value) in enumerate(tmpspk.items()):
            df = pd.DataFrame(tmpspk[icell], columns=['spktime'])
            if not df.empty:
                s = df.groupby(pd.cut(df['spktime'], tmpbin)).size()
                spkAssign = pd.Series(s).values

                for ibin in range(0, binnum):
                    if sum(tmp_binning_Linear == ibin+1):
                        tmp_binned_spk[idx, ibin] = sum(
                            spkAssign[tmp_binning_Linear == ibin+1])
                        tmpfr[idx, ibin] = (sum(
                            spkAssign[tmp_binning_Linear == ibin+1])/(sum(tmp_binning_Linear == ibin+1)*0.025))

        binned_Spk_Animal.append(tmp_binned_spk)
        binned_Fr_Animal.append(gaussian_filter1d(tmpfr, sigma))

    return binned_Spk_Animal, binned_Fr_Animal, binned_time


def time_across_activity(event, spk, time, cellname, timewindow, timebin_size,
                         smoothing):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    base_bin_num = int(1/timebin_size)
    before_entry = timewindow[0]
    after_entry = timewindow[1]
    whole_event = []
    whole_event_fr = []
    for idx, icell in enumerate(cellname):
        tmpcell = spk[cellname[idx]]
        my_rows, my_cols = (len(event[0]),
                            len(np.arange(-before_entry, after_entry, timebin_size))-1)
        eventtrial = [[0]*my_cols]*my_rows
        eventtrialfr = [[0]*my_cols]*my_rows
        for itrial in range(len(event[0])):
            tmpbin = np.arange(event[0][itrial][0]-before_entry,
                                  event[0][itrial][0]+after_entry, timebin_size)
            tmptrial, _ = np.histogram(tmpcell, bins=tmpbin)
            normtrial = tmptrial - np.mean(tmptrial[0:base_bin_num])
            eventtrial[itrial] = normtrial
            eventtrialfr[itrial] = tmptrial
        eventtrial = np.vstack(eventtrial)
        eventtrialfr = np.vstack(eventtrialfr)
        eventtrial = np.sum(eventtrial, axis=0)/(timebin_size*len(event[0]))
        eventtrialfr = np.sum(eventtrialfr, axis=0)/(timebin_size*len(event[0]))
        eventtrial = gaussian_filter1d(eventtrial, smoothing)
        eventtrialfr = gaussian_filter1d(eventtrialfr, smoothing)
        whole_event.append(eventtrial)
        whole_event_fr.append(eventtrialfr)
    whole_event = np.vstack(whole_event)
    whole_event_fr = np.vstack(whole_event_fr)
    return whole_event, whole_event_fr

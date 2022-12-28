# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:54:10 2021

@author: yeong
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from scipy.stats.stats import pearsonr
# SALT analysis (for increasing activity)


def salt_test(windowsize, salt_bin, light, tmpcell, lightwidth, dysynapse):
    from opto.salt import salt
    import numpy as np
    from etc.range_func import crange
    # SALT baseline data generation
    binnum = int(windowsize/salt_bin)
    salt_test = np.zeros((len(light), binnum))
    salt_base = np.zeros((len(light), int(0.1/salt_bin)))
    for ilight in range(len(light)):
        basebin = crange(light[ilight]-0.1, light[ilight], salt_bin)
        salt_base[ilight, :], t = np.histogram(tmpcell,
                                               bins=basebin.tolist())
        testbin = crange(light[ilight]+lightwidth+dysynapse,
                         light[ilight]+lightwidth+dysynapse+windowsize,
                         salt_bin)
        salt_test[ilight, :], t = np.histogram(tmpcell,
                                               bins=testbin.tolist())
    salt_p, _, salt_p_dec = salt(salt_base, salt_test, salt_bin, windowsize)
    if salt_p < 0.01:
        salt_result = 1
    else:
        salt_result = 0
    return salt_result


def log_rank_test(windowsize, light, log_bin_size, tmpcell, lightwidth,
                  tmpcellwv):
    import numpy as np
    import random
    from lifelines.statistics import logrank_test
    stimwv = []
    zerocellwv = np.random.rand(np.shape(tmpcellwv)[0], np.shape(tmpcellwv)[1])
    spk_latency = np.zeros((2, len(light)))+windowsize
    event_array = np.zeros((2, len(light)))
    light_gap = np.diff(light).tolist()
    light_gap.insert(-1, light_gap[-1])
    for ilight in range(len(light)):
        random_candidate = np.arange(light[ilight]-light_gap[ilight],
                                     light[ilight]-windowsize, log_bin_size)
        randomtime = random.choice(random_candidate)
        random_ind = [idx for idx, element in enumerate(
            tmpcell) if tmpcell[idx] > randomtime]
        light_ind = [idx for idx, element in enumerate(
            tmpcell) if tmpcell[idx] > light[ilight]]
        if not len(np.where(tmpcell > randomtime)[0]):
            random_latency = windowsize + lightwidth
        else:
            random_latency = tmpcell[random_ind[0]]-randomtime
        if not len(np.where(tmpcell > light[ilight])[0]):
            first_latency = windowsize + lightwidth
        else:
            first_latency = tmpcell[light_ind[0]]-light[ilight]
        if first_latency < windowsize + lightwidth:
            spk_latency[0, ilight] = first_latency
            event_array[0, ilight] = 1
            stimwv.append(
                np.squeeze(tmpcellwv[np.where((tmpcell >= light[ilight]) & (tmpcell < light[ilight]+windowsize+lightwidth)), :], 0))
        if random_latency < windowsize + lightwidth:
            spk_latency[1, ilight] = random_latency
            event_array[1, ilight] = 1
        results = logrank_test(spk_latency[0, :], spk_latency[1, :],
                               event_observed_A=event_array[0, :],
                               event_observed_B=event_array[1, :])
    log_rank_test = results.p_value
    if len(stimwv) < 1:
        stimwv = zerocellwv
    else:
        stimwv = np.vstack(np.array(stimwv))
    spontaneous_wv = np.mean(tmpcellwv, axis=0)
    stim_wv = np.mean(stimwv, axis=0)
    waveform_corr, _ = pearsonr(spontaneous_wv, np.mean(stimwv, axis=0))
    return log_rank_test, waveform_corr, spontaneous_wv, stim_wv


def wavform_corr(windowsize, wv, light, tmpcell, lightwidth):
    stimwv = []
    zerocellwv = np.random.rand(np.shape(wv)[0], np.shape(wv)[1])
    # tmpcell = np.array(tmpcell)
    for ilight in range(len(light)):
        light_ind = [idx for idx, element in enumerate(
            tmpcell) if tmpcell[idx] > light[ilight]]
        if not len(np.where(tmpcell > light[ilight])[0]):
            first_latency = windowsize + lightwidth
        else:
            first_latency = tmpcell[light_ind[0]]-light[ilight]

        if first_latency < windowsize + lightwidth:
            stimwv.append(
                np.squeeze(
                    wv[np.where((tmpcell >= light[ilight])
                                & (tmpcell < light[ilight]+windowsize+lightwidth)), :], 0))
    if len(stimwv) < 1:
        stimwv = zerocellwv
    else:
        stimwv = np.vstack(np.array(stimwv))
    spontaneous_wv = np.mean(wv, axis=0)
    stim_wv = np.mean(stimwv, axis=0)
    w_corr, _ = pearsonr(spontaneous_wv, np.mean(stimwv, axis=0))
    return w_corr


def inactive_test(inactivewn, light, binsize, tmpcell, sigma):
    import numpy as np
    from etc.range_func import crange
    from scipy.stats import mannwhitneyu
    import random
    from scipy.signal.windows import gaussian
    base = []
    test = []
    gauss_win = gaussian(sigma, 3)
    for ilight in range(len(light)):
        gbasebin = crange(light[ilight]-inactivewn, light[ilight]+inactivewn,
                          binsize)
        tmpbase, t = np.histogram(tmpcell,
                                  bins=gbasebin.tolist())
        tmpbase_smooth = np.convolve(tmpbase, gauss_win, mode='same')
        mean_base = np.mean(tmpbase_smooth[0:int(len(tmpbase_smooth)/2)])
        tmptest_smooth = tmpbase_smooth[int(len(tmpbase_smooth)/2)+15:]

        if (np.min(tmptest_smooth) < mean_base*0.5) & (mean_base > 0.1):
            # print(ilight)
            minindex = [idx+1 for idx, element in enumerate(tmptest_smooth)
                        if element < mean_base*0.5]
            minindex.insert(0, 0)
            firstbelow = [0, len(tmptest_smooth)-1]
            transition = np.where(np.diff(minindex) != 1)[0]
            if (len(transition) == 1) & (minindex[1] == 1):
                firstbelow[1] = transition[0]
            elif (len(transition) == 1) & (minindex[1] != 1):
                firstbelow[0] = transition[0]
            elif len(transition) > 1:
                firstbelow = [minindex[x] for x in transition]
            test.append(tmptest_smooth[firstbelow[0]:firstbelow[1]+1])
            binnum = firstbelow[1]+1-firstbelow[0]
            base.append(random.sample(np.ndarray.tolist(tmpbase_smooth),
                                      binnum))
        else:
            base.append(tmpbase_smooth[0:int(len(tmpbase_smooth)/2)])
            test.append(tmpbase_smooth[int(len(tmpbase_smooth)/2):])
    base = np.concatenate(base, axis=0)
    test = np.concatenate(test, axis=0)
    if (np.ndarray.tolist(base) == np.ndarray.tolist(test)):
        p_value = 1
    else:
        _, p_value = mannwhitneyu(base, test, alternative='greater')
    return p_value


def inhibit_test(inactivewn, light, lightwidth, binsize, tmpcell, sigma):
    import numpy as np
    from etc.range_func import crange
    from scipy.stats import mannwhitneyu
    from scipy.signal.windows import gaussian
    tmpsmooth = []
    base = []
    test = []
    gauss_win = gaussian(sigma, 3)
    for ilight in range(len(light)):
        gbasebin = crange(light[ilight]-inactivewn, light[ilight]+inactivewn,
                          binsize)
        tmpbase, t = np.histogram(tmpcell, bins=gbasebin.tolist())
        tmpbase_smooth = np.convolve(tmpbase, gauss_win, mode='same')
        tmpsmooth.append(tmpbase_smooth)
        base.append(tmpbase_smooth[0:int(len(tmpbase_smooth)/2)])
        test.append(
            tmpbase_smooth[int(len(tmpbase_smooth)/2+lightwidth/binsize):])
    base = np.concatenate(base, axis=0)
    test = np.concatenate(test, axis=0)
    basemean = np.sum(base)/(binsize*len(base))
    testmean = np.sum(test)/(binsize*len(test))
    if (basemean > .5):
        _, p_value = mannwhitneyu(base, test, alternative='greater')
        _, p_value_inhibit = mannwhitneyu(base, test, alternative='less')
    else:
        p_value = 1
        p_value_inhibit = 1
    return p_value, p_value_inhibit


def inhibit_test_NpHR(inactivewn, light, binsize, tmpcell, sigma):
    import numpy as np
    from etc.range_func import crange
    from scipy.stats import ttest_rel
    from scipy.signal.windows import gaussian
    tmpsmooth = []
    gauss_win = gaussian(sigma, 3)
    for ilight in range(len(light)):
        gbasebin = crange(light[ilight]-inactivewn, light[ilight]+2*inactivewn,
                          binsize)
        tmpbase, t = np.histogram(tmpcell, bins=gbasebin.tolist())
        # tmpbase_smooth = np.convolve(tmpbase, gauss_win, mode='same')
        tmpsmooth.append(tmpbase)
    tmpbinned_spk = np.sum(tmpsmooth, axis = 0)
    tmpbinned_fr = tmpbinned_spk/(binsize*len(light))
    movbin = int(1/binsize)
    comparebin = int(10/binsize)
    
    N=movbin # number of points to test on each side of point of interest, best if even
    x = tmpbinned_fr
    padded_x = np.insert(np.insert( np.insert(x, len(x), np.empty(int(N/2))*np.nan), 0, np.empty(int(N/2))*np.nan ),0,0)
    n_nan = np.cumsum(np.isnan(padded_x))
    cumsum = np.nancumsum(padded_x) 
    window_sum = cumsum[N+1:] - cumsum[:-(N+1)] - x # subtract value of interest from sum of all values within window
    window_n_nan = n_nan[N+1:] - n_nan[:-(N+1)] - np.isnan(x)
    window_n_values = (N - window_n_nan)
    movavg = (window_sum) / (window_n_values)
    basemean = np.mean(tmpbinned_fr[0:comparebin])    
    testmean = np.mean(tmpbinned_fr[comparebin:2*comparebin])
    if (basemean > .3):
        _, p_value = ttest_rel(tmpbinned_fr[0:comparebin],
                               tmpbinned_fr[comparebin:2*comparebin])
    else:
        p_value = 1
    return movavg, p_value, basemean, testmean


def inactive_test_eOPN(inactivewn, light, binsize, tmpcell, sigma):
    import numpy as np
    from etc.range_func import crange
    from scipy.stats import mannwhitneyu
    from scipy.stats import wilcoxon
    from scipy.signal.windows import gaussian
    from scipy.stats import zscore
    gauss_win = gaussian(sigma, 3)
    stimtime = light[0]
    window = int(inactivewn/binsize)
    gbasebin = crange(stimtime-inactivewn, stimtime+inactivewn*3,
                      binsize)
    tmpbase, t = np.histogram(tmpcell,
                              bins=gbasebin.tolist())
    tmpbase = tmpbase/binsize
    tmpbase_smooth = np.convolve(tmpbase, gauss_win, mode='same')
    # N = int(1/binsize)
    # tmpbase_smooth = np.convolve(tmpbase, np.ones((N,))/N, mode='valid')
    # tmpbase_smooth = zscore(tmpbase_smooth)
    base = tmpbase_smooth[0:window]
    test = tmpbase_smooth[window:window*2]
    active_p_value = 1
    inactive_p_value = 1
    if (np.ndarray.tolist(base) == np.ndarray.tolist(test)):
        active_p_value = 1
        inactive_p_value = 1
    elif np.mean(base)*0.5 > np.mean(test):
        _, inactive_p_value = wilcoxon(base, test, alternative='greater')
        active_p_value = 1
    elif np.mean(base)*1.5 < np.mean(test):
        _, active_p_value = wilcoxon(base, test, alternative='less')
        inactive_p_value = 1
    return active_p_value, inactive_p_value, tmpbase


def tagged_neuron(light, wv, spk, draw=0, **kargs):
    # Make a binned data onset to light stimulation
    import warnings
    warnings.filterwarnings("ignore")
    windowsize = 0.01
    inactivewn = 0.05
    salt_bin = 0.001
    lightwidth = 0.005
    log_bin_size = 0.002
    sigma = 10
    binsize = 0.005
    timewindow = 0.06
    num_bin = int(timewindow/log_bin_size*2-1)
    tmplight = light
    tmpwv = wv['waveform']
    tmpspk = spk
    tmpkey = list(tmpspk.keys())
    tag_inhibit = []
    tag_activate = []
    for icell in range(len(tmpkey)):
        tmpbinned_tag = np.zeros((len(tmplight), num_bin))
        tmpcell = tmpspk[tmpkey[icell]]
        tmpcell = np.array(tmpcell)
        tmpcellwv = np.array(tmpwv[icell])
        tmpinactive, _ = inhibit_test(
            inactivewn, tmplight, lightwidth, binsize, tmpcell, sigma)
        tmpsalt = salt_test(windowsize, salt_bin, tmplight,
                            tmpcell, lightwidth, 0)
        tmpwvcorr = wavform_corr(
            windowsize, tmpcellwv, tmplight, tmpcell, lightwidth)
        # tmplog, tmpwvcorr, spontwv, stimwv = log_rank_test(
        #     windowsize, tmplight, log_bin_size, tmpcell, lightwidth, tmpcellwv)
        if (tmpsalt & (tmpwvcorr >= 0.8)):
            for ilight in range(len(tmplight)):
                a = np.arange(tmplight[ilight], tmplight[ilight]+timewindow,
                              log_bin_size)
                b = np.arange(tmplight[ilight]-timewindow, tmplight[ilight],
                              log_bin_size)
                tmpbin = np.concatenate((b, a))
                a, b = np.histogram(tmpcell, tmpbin)
                tmpbinned_tag[ilight, :] = a
            if np.sum(tmpbinned_tag[:, int(len(a)/2):int(len(a)/2+windowsize/log_bin_size)]) > len(tmplight)*0.01:
                tag_activate.append(tmpkey[icell])
                if draw:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    time_temp = np.arange(num_bin)*log_bin_size
                    tmpbinned_tag = tmpbinned_tag*time_temp
                    tmpbinned_tag[tmpbinned_tag == 0] = -1
                    ax.eventplot(tmpbinned_tag, linelength=4,
                                 linewidth=5, color='black')
                    ax.add_patch(rect((timewindow, -2), log_bin_size, len(tmplight)+2,
                                      color='dodgerblue'))
                    ax.set_xlim([0.03, np.max(time_temp)])
                    ax.set_ylim([-2, len(tmplight)+2])
                    ax.set_xticks([0.03, 0.06, 0.09, 0.12])
                    ax.set_xticklabels(
                        ['-30ms', '0', '30ms', '60ms'], fontsize=15)
                    ax.set_yticks([len(tmplight)+2])
                    ax.set_yticklabels(['600'], fontsize=15)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.ylabel('Trials', fontsize=20)
                    plt.title(tmpkey[icell], fontsize=20)        
                    plt.savefig('Active_neuron_%d' %(icell))
        elif tmpinactive < 0.05:
            for ilight in range(len(tmplight)):
                a = np.arange(tmplight[ilight], tmplight[ilight]+timewindow,
                              log_bin_size)
                b = np.arange(tmplight[ilight]-timewindow, tmplight[ilight],
                              log_bin_size)
                tmpbin = np.concatenate((b, a))
                a, b = np.histogram(tmpcell, tmpbin)
                tmpbinned_tag[ilight, :] = a
            tag_inhibit.append(tmpkey[icell])
            if draw:
                fig, ax = plt.subplots(figsize=(8, 6))
                time_temp = np.arange(num_bin)*log_bin_size
                tmpbinned_time_tag = tmpbinned_tag*time_temp
                tmpbinned_time_tag[tmpbinned_time_tag == 0] = -1
                ax.eventplot(tmpbinned_time_tag, linelength=5,
                             linewidth=5, color='black')
                ax.add_patch(rect((timewindow, -2), log_bin_size,
                                  len(tmplight)+2, color='dodgerblue'))
                ax.set_xlim([0, np.max(time_temp)])
                ax.set_ylim([-2, len(tmplight)+2])
                ax.set_xticks([0, 0.03, 0.06, 0.09, 0.12])
                ax.set_xticklabels(
                    ['-60ms', '-30ms', '0', '30ms', '60ms'], fontsize=15)
                # ax.set_xlim([0.03, np.max(time_temp)])
                ax.set_ylim([-2, len(tmplight)+2])
                # ax.set_xticks([0.03, 0.06, 0.09, 0.12])
                # ax.set_xticklabels(['-30ms', '0', '30ms', '60ms'], fontsize=15)
                ax.set_yticks([602])
                ax.set_yticklabels(['600'], fontsize=15)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.ylabel('Trials', fontsize=20)
                plt.title(tmpkey[icell], fontsize=20)
                plt.savefig('Inhibit_neuron_%d' %(icell))
       
    return tag_activate, tag_inhibit


def tagged_neuron_eOPN(light, spk, draw=0, **kargs):
    from etc.range_func import crange
    from scipy.signal.windows import gaussian
    inactivewn = 300
    sigma = 10
    binsize = 0.005
    baseline_t = 300
    tmplight = light['Postbaseline']
    tmpspk = spk
    tmpkey = list(tmpspk.keys())
    tag_inhibit = []
    tag_activate = []
    for icell in range(len(tmpkey)):
        tmpcell = tmpspk[tmpkey[icell]]
        active_p, inactive_p, spk_smooth = inactive_test_eOPN(inactivewn,
                                                              tmplight, binsize, tmpcell, sigma)
        tmpbinsize = 5
        tmpwindow = int(inactivewn/tmpbinsize)
        stimtime = tmplight[0]
        gbasebin = crange(stimtime-inactivewn, stimtime+inactivewn*3,
                          tmpbinsize)
        tmpbase, t = np.histogram(tmpcell,
                                  bins=gbasebin.tolist())
        gauss_win = gaussian(10, 3)
        tmp_smooth = np.convolve(tmpbase, gauss_win, mode='same')
        baseline = tmp_smooth[tmpwindow-int(baseline_t/tmpbinsize):tmpwindow]
        tmp_smooth = tmp_smooth[tmpwindow -
                                int(baseline_t/tmpbinsize):tmpwindow*3]
        z_baseline = (tmp_smooth-np.mean(baseline)) / \
            (np.max(baseline)-np.min(baseline))
        # fig, ax = plt.subplots(figsize=(8, 6))
        # plt.plot(z_baseline)
        if inactive_p < 0.05:
            tag_inhibit.append(tmpkey[icell])
            if draw:
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(z_baseline, color='lightcoral')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlim([0, len(z_baseline)])
                maxy = np.ceil(np.max(z_baseline))
                ax.set_ylim([-1, maxy])
                ax.set_xticks([0, int(baseline_t/tmpbinsize),
                               int(baseline_t/tmpbinsize)*2, int(baseline_t/tmpbinsize)*3])
                ax.set_yticks([0, maxy])
                ax.add_patch(rect((int(baseline_t/tmpbinsize), -1), tmpwindow,
                                  4, color='lemonchiffon'))
                ax.add_patch(rect((int(baseline_t/tmpbinsize), maxy -
                             0.05), tmpwindow, 0.05, color='yellow'))
                ax.set_xticklabels(['-5min', '0', '5min', '10min'], fontsize=15)
                ax.set_yticklabels(['0', maxy], fontsize=15)
                plt.ylabel('Normalized firing rate', fontsize=15)
             
            # plt.savefig('Inhibit_neuron_%d' %(icell))
        elif active_p < 0.05:
            tag_activate.append(tmpkey[icell])
            if draw:
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(z_baseline, color='lightcoral')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlim([0, len(z_baseline)])
                maxy = np.ceil(np.max(z_baseline))
                ax.set_ylim([-1, maxy])
                ax.set_xticks([0, int(baseline_t/tmpbinsize),
                              int(baseline_t/tmpbinsize)*2, int(baseline_t/tmpbinsize)*3])
                ax.set_yticks([0, maxy])
                ax.add_patch(rect((int(baseline_t/tmpbinsize), -1),
                             tmpwindow, 4, color='lemonchiffon'))
                ax.add_patch(rect((int(baseline_t/tmpbinsize), maxy -
                             0.05), tmpwindow, 0.05, color='yellow'))
                ax.set_xticklabels(['-5min', '0', '5min', '10min'], fontsize=15)
                ax.set_yticklabels(['0', maxy], fontsize=15)
                plt.ylabel('Normalized firing rate', fontsize=15)
                # plt.savefig('Activate_neuron_%d' % (icell))
    return tag_activate, tag_inhibit


def tagged_neuron_NpHR(light, spk, draw=0, **kargs):
    import warnings
    warnings.filterwarnings("ignore")
    inactivewn = 10
    sigma = 10
    binsize = 0.1
    tmplight = light
    tmpspk = spk
    tmpkey = list(tmpspk.keys())
    tag_inhibit = []
    tag_activate = []
    for icell in range(len(tmpkey)):
        tmpcell = tmpspk[tmpkey[icell]]
        tmpcell = np.array(tmpcell)
        tmpfr, tmpinactive, tmpbase, tmptest = inhibit_test_NpHR(
            inactivewn, tmplight, binsize, tmpcell, sigma)

        if tmpinactive < 0.05 and tmpbase > tmptest:
            tag_inhibit.append(tmpkey[icell])
            if draw:
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(tmpfr)
                ax.add_patch(rect((int(inactivewn/binsize), np.max(tmpfr)*1.2), 
                                  int(inactivewn/binsize), np.max(tmpfr)*.03,
                                  color='darkorange'))  
                ax.add_patch(rect((int(inactivewn/binsize), 0), int(inactivewn/binsize),
                                  np.max(tmpfr)*1.2, color='navajowhite'))  
                ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
                ax.set_xticklabels(
                    ['-10', '-5', '0', '5', '10', '15', '20'], fontsize=20)
                plt.yticks(fontsize = 20)
                plt.xlabel('Time (s)', fontsize = 20)
                # ax.set_xlim([0.03, np.max(time_temp)])
               
                # ax.set_xticks([0.03, 0.06, 0.09, 0.12])
                # ax.set_xticklabels(['-30ms', '0', '30ms', '60ms'], fontsize=15)
                # ax.set_yticklabels(['%d' % (len(tmplight))], fontsize=15)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.ylabel('Firing rate (Hz)', fontsize=20)
                plt.title(tmpkey[icell], fontsize=20)
                plt.savefig('Inhibit_neuron_%d' %(icell))
                
        elif tmpinactive < 0.05 and tmpbase < tmptest:
            tag_activate.append(tmpkey[icell])
            if draw:
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(tmpfr)
                ax.add_patch(rect((int(inactivewn/binsize), np.max(tmpfr)*1.2), 
                                  int(inactivewn/binsize), np.max(tmpfr)*.03,
                                  color='darkorange'))  
                ax.add_patch(rect((int(inactivewn/binsize), 0), int(inactivewn/binsize),
                                  np.max(tmpfr)*1.2, color='navajowhite'))  
                ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
                plt.xlabel('Time (s)', fontsize = 20)
                ax.set_xticklabels(
                    ['-10', '-5', '0', '5', '10', '15', '20'], fontsize=20)
                # ax.set_xlim([0.03, np.max(time_temp)])
               
                # ax.set_xticks([0.03, 0.06, 0.09, 0.12])
                # ax.set_xticklabels(['-30ms', '0', '30ms', '60ms'], fontsize=15)
                # ax.set_yticklabels(['%d' % (len(tmplight))], fontsize=15)
                plt.yticks(fontsize = 20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.ylabel('Firing rate (Hz)', fontsize=20)
                plt.title(tmpkey[icell], fontsize=20)
                plt.savefig('Activate_neuron_%d' %(icell))       
    return tag_activate, tag_inhibit

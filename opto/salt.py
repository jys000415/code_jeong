# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:15:48 2021

@author: yeong
"""
import numpy as np
from etc.range_func import crange


def salt(spt_baseline, spt_test, dt, wn):
    # =============================================================================
    # Input arguments:
    #       SPT_BASELINE - Discretized spike raster for stimulus-free baseline
    #           period. N x M binary matrix with N rows for trials and M
    #           columns for spikes. Spike times have to be converted to a
    #           binary matrix with a temporal resolution provided in DT. The
    #           baseline segment has to excede the window size (WN) multiple
    #           times, as the length of the baseline segment divided by the
    #           window size determines the sample size of the null
    #           distribution (see below).
    #       SPT_TEST - Discretized spike raster for test period, i.e. after
    #           stimulus. N x M binary matrix with N rows for trials and M
    #           columns for spikes. Spike times have to be converted to a
    #           binary matrix with a temporal resolution provided in DT. The
    #           test segment has to excede the window size (WN). Spikes out of
    #           the window are disregarded.
    #       DT - Time resolution of the discretized spike rasters in seconds.
    #       WN - Window size for baseline and test windows in seconds
    #           (optional; default, 0.001 s).
    # =============================================================================
    wn = wn * 1000
    dt = dt * 1000
    # Latency histogram - baseline
    [tno, st] = np.shape(spt_baseline)
    nmbn = int(np.round(wn/dt))
    edges = crange(0, nmbn+2, 1)
    nm = int(np.floor(st/nmbn))
    lsi, slsi = np.zeros((tno, nm)), np.zeros((tno, nm))
    hlsi, nhlsi = np.zeros((nmbn+1, nm+1)), np.zeros((nmbn+1, nm+1))
    next_iter = 0
    for i in np.arange(0, st, nmbn):
        for k in range(tno):
            cspt = spt_baseline[k, int(i):int(i+nmbn)]
            pki = [ii for ii in range(len(cspt)) if cspt[ii] != 0]
            if pki:
                lsi[k, next_iter] = pki[0]+1
            else:
                lsi[k, next_iter] = 0
        slsi[:, next_iter] = np.sort(lsi[:, next_iter])
        hst, _ = np.histogram(slsi[:, next_iter], edges)
        hlsi[:, next_iter] = hst[0:-1]
        nhlsi[:, next_iter] = hlsi[:, next_iter] / np.sum(hlsi[:, next_iter])
        next_iter = next_iter+1
    # ISI histogram - test
    tno_test = np.shape(spt_test)[0]
    lsi_tt = np.empty((tno_test, 1))
    lsi_tt[:] = np.NaN
    for k in range(tno_test):
        cspt = spt_test[k, 0:nmbn]
        pki = [i for i in range(len(cspt)) if cspt[i] != 0]
        if pki:
            lsi_tt[k, 0] = pki[0]+1
        else:
            lsi_tt[k, 0] = 0
    slsi_tt = np.sort(lsi_tt[:])
    hst, _ = np.histogram(slsi_tt, edges)
    hlsi[:, next_iter] = hst[0:-1]
    nhlsi[:, next_iter] = hlsi[:, next_iter] / np.sum(hlsi[:, next_iter])
    # JS-divergence
    kn = nm+1
    jsd = np.empty((kn, kn))
    jsd[:] = np.NAN
    for k1 in range(kn):
        D1 = nhlsi[:, k1]
        for k2 in np.arange(k1+1, kn):
            D2 = nhlsi[:, k2]
            jsd[k1, k2] = np.sqrt(JSdiv(D1, D2)*2)
    p, I, p_dec = makep(jsd, kn)
    return p, I, p_dec


def makep(kld, kn):
    pnhk = kld[0:kn-1, 0:kn-1]
    nullhypkld = []
    for idata in range(kn-1):
        if ~np.isnan(pnhk[:, idata]).all():
            nullhypkld.extend(pnhk[:, idata][~np.isnan(pnhk[:, idata])])
    testkld = np.median(kld[0:kn-1, kn-1])
    sno = len(nullhypkld[:])
    p_value = np.sum([True for x in nullhypkld if x >= testkld])/sno
    p_value_dec = np.sum([True for x in nullhypkld if x < testkld])/sno
    Idiff = testkld - np.median(nullhypkld)
    return p_value, Idiff, p_value_dec


def JSdiv(P, Q):
    if (np.abs(np.sum(P[:])-1) > 0.00001) | (np.abs(np.sum(Q[:])-1) > 0.00001):
        print('Input arguments must be probability distributions.')
    if not np.shape(P) == np.shape(Q):
        print('Input distributions must be of the same size')
    # JS-divergence
    M = (P+Q)/2
    D1 = KLdist(P, M)
    D2 = KLdist(Q, M)
    D = (D1+D2)/2
    return D


def KLdist(P, Q):
    if (np.abs(np.sum(P[:])-1) > 0.00001) | (np.abs(np.sum(Q[:])-1) > 0.00001):
        print('Input arguments must be probability distributions.')
    if not np.shape(P) == np.shape(Q):
        print('Input distributions must be of the same size')
    # KL-distance
    P2 = P[np.multiply(P, Q) > 0]
    Q2 = Q[np.multiply(P, Q) > 0]
    P2 = P2/np.sum(P2)
    Q2 = Q2/np.sum(Q2)
    D = np.sum(np.multiply(P2, np.log(np.divide(P2, Q2))))
    return D

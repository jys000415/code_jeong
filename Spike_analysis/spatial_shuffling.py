# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:05:11 2021

@author: yeong
"""


def make_FR_Time_template(prefercloseFrAnimal, othercloseFrAnimal, centerFrAnimal, preferopenFrAnimal, otheropenFrAnimal,
                          prefercloseTime, othercloseTime, centerTime, preferopenTime, otheropenTime):
    import numpy as np

    EPM_Fr = []
    for ianimal in range(len(prefercloseFrAnimal)):
        EPM_Fr.append(np.concatenate((prefercloseFrAnimal[ianimal], othercloseFrAnimal[ianimal],
                                      centerFrAnimal[ianimal], preferopenFrAnimal[ianimal], otheropenFrAnimal[ianimal]), axis=1))
    EPM_FrAnimal = []
    for ianimal in range(len(prefercloseFrAnimal)):
        if len(EPM_FrAnimal) == 0:
            EPM_FrAnimal = EPM_Fr[0]
        else:
            EPM_FrAnimal = np.concatenate(
                (EPM_FrAnimal, EPM_Fr[ianimal]), axis=0)
    cellnum = [len(prefercloseFrAnimal[i])
               for i in range(len(prefercloseFrAnimal))]
    EPM_Time = np.concatenate(
        (prefercloseTime, othercloseTime, centerTime, preferopenTime, otheropenTime), axis=1)
    EPM_Time / np.sum(EPM_Time, axis=1)[:, np.newaxis]
    EPM_TimeAnimal = []
    for ianimal in range(len(prefercloseFrAnimal)):
        if len(EPM_TimeAnimal) == 0:
            EPM_TimeAnimal = np.tile(np.array(EPM_Time[0, :]), (cellnum[0], 1))
        else:
            EPM_TimeAnimal = np.concatenate((EPM_TimeAnimal, np.tile(
                np.array(EPM_Time[ianimal, :]), (cellnum[ianimal], 1))), axis=0)

    EPM_avgFr = np.sum(EPM_FrAnimal, 1)/len(EPM_FrAnimal[0, :])
    return EPM_FrAnimal, EPM_TimeAnimal, EPM_avgFr


def make_FR_Time_template_EZM(prefercloseFrAnimal, othercloseFrAnimal, preferopenFrAnimal, otheropenFrAnimal,
                              prefercloseTime, othercloseTime, preferopenTime, otheropenTime):
    import numpy as np

    EPM_Fr = []
    for ianimal in range(len(prefercloseFrAnimal)):
        EPM_Fr.append(np.concatenate((prefercloseFrAnimal[ianimal], othercloseFrAnimal[ianimal],
                                      preferopenFrAnimal[ianimal], otheropenFrAnimal[ianimal]), axis=1))
    EPM_FrAnimal = []
    for ianimal in range(len(prefercloseFrAnimal)):
        if len(EPM_FrAnimal) == 0:
            EPM_FrAnimal = EPM_Fr[0]
        else:
            EPM_FrAnimal = np.concatenate(
                (EPM_FrAnimal, EPM_Fr[ianimal]), axis=0)
    cellnum = [len(prefercloseFrAnimal[i])
               for i in range(len(prefercloseFrAnimal))]
    EPM_Time = np.concatenate(
        (prefercloseTime, othercloseTime, preferopenTime, otheropenTime), axis=1)
    EPM_Time / np.sum(EPM_Time, axis=1)[:, np.newaxis]
    EPM_TimeAnimal = []
    for ianimal in range(len(prefercloseFrAnimal)):
        if len(EPM_TimeAnimal) == 0:
            EPM_TimeAnimal = np.tile(np.array(EPM_Time[0, :]), (cellnum[0], 1))
        else:
            EPM_TimeAnimal = np.concatenate((EPM_TimeAnimal, np.tile(
                np.array(EPM_Time[ianimal, :]), (cellnum[ianimal], 1))), axis=0)

    EPM_avgFr = np.sum(EPM_FrAnimal, 1)/len(EPM_FrAnimal[0, :])
    return EPM_FrAnimal, EPM_TimeAnimal, EPM_avgFr


def cal_spatial_info(EPM_FrAnimal, EPM_TimeAnimal, EPM_avgFr, num_bin):
    import numpy as np
    spatialinfo = []
    for ibin in range(num_bin):
        with np.errstate(divide='ignore', invalid='ignore'):
            tmpinfo = EPM_FrAnimal[:, ibin]/EPM_avgFr * \
                np.log(EPM_FrAnimal[:, ibin]/EPM_avgFr)*EPM_TimeAnimal[:, ibin]
        tmpinfo[np.isnan(tmpinfo)] = 0
        tmpinfo[np.isinf(tmpinfo)] = 0
        spatialinfo.append(tmpinfo)
    EPM_spatial_Info = sum(spatialinfo)
    return EPM_spatial_Info


def shuffle_place(preferclosedLinear, preferclosedLinearTime, otherclosedLinear, otherclosedLinearTime, preferopenLinear,
                  preferopenLinearTime, otheropenLinear, otheropenLinearTime, centerLinear, centerLinearTime, epmspk, shuffle_num, arm_spatial_bin, center_spatial_bin, binsize, sigma):
    import numpy as np
    import random
    from Spike_analysis.spatial_shuffling import make_FR_Time_template, cal_spatial_info
    # shuffle_num = 100
    # Make shuffled spk
    shuffle_spatial_info = []
    for ishuffle in range(shuffle_num):
        epmshufflespk = []
        for ianimal in range(len(preferclosedLinear)):
            tmpspk = epmspk[ianimal]
            shufflespk = {}
            for idx, icell in enumerate(tmpspk):
                if len(tmpspk[icell]) > 0:
                    tmpdiff = np.diff(tmpspk[icell])
                    random.shuffle(tmpdiff)
                    tmpisi = np.cumsum(tmpdiff)
                    shufflespk[icell] = np.append(
                        tmpspk[icell][0], (tmpspk[icell][0])+tmpisi)
                else:
                    shufflespk[icell] = []
            epmshufflespk.append(shufflespk)

        from Spike_analysis.spikeanalysis import assign_spatial_bin

        prefercloseSpkAnimal, prefercloseFrAnimal, prefercloseTime = assign_spatial_bin(preferclosedLinear, preferclosedLinearTime,
                                                                                        arm_spatial_bin, binsize, epmshufflespk, sigma)
        othercloseSpkAnimal, othercloseFrAnimal, othercloseTime = assign_spatial_bin(otherclosedLinear, otherclosedLinearTime,
                                                                                     arm_spatial_bin, binsize, epmshufflespk, sigma)
        preferopenSpkAnimal, preferopenFrAnimal, preferopenTime = assign_spatial_bin(preferopenLinear, preferopenLinearTime,
                                                                                     arm_spatial_bin, binsize, epmshufflespk, sigma)
        otheropenSpkAnimal, otheropenFrAnimal, otheropenTime = assign_spatial_bin(otheropenLinear, otheropenLinearTime,
                                                                                  arm_spatial_bin, binsize, epmshufflespk, sigma)

        centerSpkAnimal, centerFrAnimal, centerTime = assign_spatial_bin(centerLinear, centerLinearTime,
                                                                         center_spatial_bin, binsize, epmshufflespk, sigma)

        EPM_FrAnimal, EPM_TimeAnimal, EPM_avgFr = make_FR_Time_template(prefercloseFrAnimal, othercloseFrAnimal, centerFrAnimal, preferopenFrAnimal, otheropenFrAnimal,
                                                                        prefercloseTime, othercloseTime, centerTime, preferopenTime, otheropenTime)

        spatial_Info = cal_spatial_info(
            EPM_FrAnimal, EPM_TimeAnimal, EPM_avgFr, len(EPM_FrAnimal[0, :]))
        shuffle_spatial_info.append(spatial_Info)
        print(ishuffle)
    return shuffle_spatial_info


def shuffle_place_EZM(preferclosedLinear, preferclosedLinearTime, otherclosedLinear, otherclosedLinearTime, preferopenLinear,
                      preferopenLinearTime, otheropenLinear, otheropenLinearTime, ezmspk, shuffle_num, arm_spatial_bin, binsize, sigma):
    import numpy as np
    import random
    from Spike_analysis.spatial_shuffling import make_FR_Time_template_EZM, cal_spatial_info
    # shuffle_num = 100
    # Make shuffled spk
    shuffle_spatial_info = []
    for ishuffle in range(shuffle_num):
        ezmshufflespk = []
        for ianimal in range(len(ezmspk)):
            tmpspk = ezmspk[ianimal]
            shufflespk = {}
            for idx, icell in enumerate(tmpspk):
                if len(tmpspk[icell]) > 0:
                    tmpdiff = np.diff(tmpspk[icell])
                    random.shuffle(tmpdiff)
                    tmpisi = np.cumsum(tmpdiff)
                    shufflespk[icell] = np.append(
                        tmpspk[icell][0], (tmpspk[icell][0])+tmpisi)
                else:
                    shufflespk[icell] = []
            ezmshufflespk.append(shufflespk)

        from Spike_analysis.spikeanalysis import assign_spatial_bin
        prefercloseSpkAnimal, prefercloseFrAnimal, prefercloseTime = assign_spatial_bin(preferclosedLinear, preferclosedLinearTime,
                                                                                        arm_spatial_bin, binsize, ezmshufflespk, sigma)
        othercloseSpkAnimal, othercloseFrAnimal, othercloseTime = assign_spatial_bin(otherclosedLinear, otherclosedLinearTime,
                                                                                     arm_spatial_bin, binsize, ezmshufflespk, sigma)
        preferopenSpkAnimal, preferopenFrAnimal, preferopenTime = assign_spatial_bin(preferopenLinear, preferopenLinearTime,
                                                                                     arm_spatial_bin, binsize, ezmshufflespk, sigma)
        otheropenSpkAnimal, otheropenFrAnimal, otheropenTime = assign_spatial_bin(otheropenLinear, otheropenLinearTime,
                                                                                  arm_spatial_bin, binsize, ezmshufflespk, sigma)

        EZM_FrAnimal, EZM_TimeAnimal, EZM_avgFr = make_FR_Time_template_EZM(prefercloseFrAnimal, othercloseFrAnimal, preferopenFrAnimal, otheropenFrAnimal,
                                                                            prefercloseTime, othercloseTime, preferopenTime, otheropenTime)

        spatial_Info = cal_spatial_info(
            EZM_FrAnimal, EZM_TimeAnimal, EZM_avgFr, len(EZM_FrAnimal[0, :]))
        shuffle_spatial_info.append(spatial_Info)
        print(ishuffle)
    return shuffle_spatial_info

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:15:38 2020
Data reorganization in each animal
@author: yeong
"""
# I can make this to function (input : path, output : dataframe)


def dataorganize(context_name, contextn_name_order, remap):
    import pandas as pd
    import os
    import numpy as np

    # Remove unnecessary keys in data
    from scipy.io import loadmat
    data = loadmat('animal1.mat')
    keys_to_remove = ["__globals__", "__header__", "__version__"]
    for key in keys_to_remove:
        del data[key]

    # DataFrame of Task time
    sorted(data.keys())
    context_name = context_name
    # context_order = data['context_order'][0]-1
    context_order = contextn_name_order
    contextind = [i for i in range(
        len(context_order)) if context_order[i] < 255]
    asscontextind = map(context_name.__getitem__, contextind)
    assorderind = map(context_order.__getitem__, contextind)
    incontextorder = list(assorderind)
    incontextname = list(asscontextind)
    taskname = [x for _, x in sorted(zip(incontextorder, incontextname))]
    event = data['Events']['Tasktime'][0].tolist()
    event_laser = []
    dflaser = {}
    dftask = pd.DataFrame(data=np.divide(event[0][0:len(taskname)], 10**6),
                          index=taskname, columns=['Start', 'Stop'])
    if len(data['Events'][0][0])>1:
        if len(data['Events'][0][0][1])>0:
            event_laser = data['Events']['lasertime'][0].tolist()
            tmpevent_laser = np.divide(event_laser[0][0], 10**6)
            for icontext in taskname:
                tmpdata = [x for x in tmpevent_laser
                           if (x >= dftask.loc[icontext]['Start']) & (x < dftask.loc[icontext]['Stop'])]
                dflaser[icontext] = tmpdata

    # DataFrame of XY position
    position = data['Xpoints'][0].tolist()
    yposition = data['Ypoints'][0].tolist()
    time = data['Time'][0].tolist()
    dfx = pd.DataFrame(data=position, columns=['Xpoint'])
    dfy = pd.DataFrame(data=yposition, columns=['Ypoint'])
    dftime = pd.DataFrame(data=time, columns=['Time'])
    xyframe = [dfx, dfy, dftime]
    dfxytime = pd.concat(xyframe, axis=1, sort=False)
    xyposition = pd.concat([dfxytime['Xpoint'],
                            dfxytime['Ypoint']], axis=1).to_numpy()
    xydtime = dfxytime['Time'].to_numpy()

    # DataFrame of Neuronal data
    cellname = data['spkT'].dtype.names
    spkdata = {}
    halfwidth = []
    halfspkwidth = []
    spknum = []
    pvratio = []
    spkwv = {}
    for cname in cellname:
        spkdata[cname] = data['spkT'][cname][0][0]
        spkwv[cname] = data['wvform'][cname][0][0]
        halfwidth.append(data['halfWidth'][cname][0][0][0][0])
        halfspkwidth.append(data['halfSpikeWidth'][cname][0][0][0][0])
        pvratio.append(data['pvratio'][cname][0][0][0][0])
        spknum.append(data['spknum'][cname][0][0][0])
    dfspkdata = pd.DataFrame(list(spkdata.items()),
                             columns=['Cell ID', 'spkTime'])
    dfwvdata = pd.DataFrame(list(spkwv.items()), columns=[
                            'Cell ID', 'waveform'])
    dfspkdata['spkwidth'] = halfspkwidth
    dfspkdata['spknum'] = spknum
    
    # CSC data load
    whichcscs = np.where(data['countOfcell'][0] ==
                         np.amax(data['countOfcell'][0]))
    # whichcscs = np.where(data['countOfcell'][0] >= 2)
    import glob
    cscname = []
    for name in glob.glob('CSC?.mat'):
        cscname.append(name)
    # cscs = []
    # for ind in whichcscs[0]:
        # tmpcsc = loadmat(cscname[ind])
        # tmpcsc = tmpcsc['cscdata']
        # cscs.append(tmpcsc)
    csc = loadmat(cscname[whichcscs[0][0]])
    keys_to_remove = ["__globals__", "__header__", "__version__"]
    for key in keys_to_remove:
        del csc[key]
    cscinfo = {'Trace': csc['cscdata'][0],
               'Time': np.divide(data['csctime'][0], 10**6),
               'Frequency': data['samplefreq'][0],
               'Samplenum': data['numofvalidsample'][0]}

    # Exclude the data that losing animal's position
    position = xyposition[np.all(xyposition > 0, axis=1), 0:]
    wholetime = np.divide(xydtime[np.all(xyposition > 0, axis=1)], 10**6)

    # Divide a poisition,time, spkdata and csc data to each context
    animalPosition = {}
    animalTime = {}
    animalSpk = {}
    animalCsc = {}
    animallight = {}
    animalWv = {}
    for tname in taskname:
        animalPosition[tname] = position[(wholetime >= dftask.loc[tname, 'Start']) &
                                         (wholetime <= dftask.loc[tname, 'Stop']), 0:]
        animalTime[tname] = wholetime[(wholetime >= dftask.loc[tname, 'Start']) &
                                      (wholetime <= dftask.loc[tname, 'Stop'])]
        tmpSpk = {}
        for cname, i in zip(dfspkdata['Cell ID'], range(len(dfspkdata['Cell ID']))):
            tmpSpk[cname] = dfspkdata['spkTime'][i][(dfspkdata['spkTime'][i] >= dftask.loc[tname, 'Start']) &
                                                    (dfspkdata['spkTime'][i] <= dftask.loc[tname, 'Stop'])]
        animalSpk.update(dict(zip([tname], [tmpSpk])))
        csctime = cscinfo['Time'][(cscinfo['Time'] >= dftask.loc[tname, 'Start']) &
                                  (cscinfo['Time'] <= dftask.loc[tname, 'Stop'])]
        animalCsc[tname] = np.vstack([csc['cscdata'][:, (cscinfo['Time'] >= dftask.loc[tname, 'Start']) &
                                                     (cscinfo['Time'] <= dftask.loc[tname, 'Stop'])], csctime])
        # animallight[tname] = dflaser
    animalWv = dfwvdata
    animalhalfvalley = halfwidth
    animalspikewidth = halfspkwidth
    animalpvratio = pvratio
    return animalPosition, animalTime, animalSpk, animalCsc, dflaser, animalWv, animalspikewidth, animalhalfvalley, animalpvratio

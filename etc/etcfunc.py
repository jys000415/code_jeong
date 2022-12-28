# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:51:32 2020

@author: yeong
"""

# Make the consecutive blocks (within 2s)


def blobassign(time, iti, fs, duration):
    import numpy as np
    blob = list()
    blobind = list()
    tmpblob = list()
    tmpblobind = list()
    blobnum = 1
    if isinstance(time[0], np.floating):
        tmpblob.append(time[0])
        tmpblobind.append(0)
        for i in range(len(time)-1):
            # if true, combine those two index into one
            if (time[i+1]-tmpblob[-1]) <= iti/(1/fs):
                # print('These two are a same blob')
                tmpblob.append(time[i+1])
                tmpblobind.append(i+1)
                if (i == len(time) - duration) and tmpblob[-1]-tmpblob[0] > duration*fs:
                    blob.append(tmpblob)
                    blobind.append(tmpblobind)
                   
            # if false, set as a block and pass to next one
            elif ((time[i+1]-tmpblob[-1]) > iti/(1/fs)) or (i == len(time)-1):
                if tmpblob[-1]-tmpblob[0] > duration*fs:
                    blob.append(tmpblob)
                    blobind.append(tmpblobind)
                    blobnum += 1
                # Set new start point
                tmpblob = list()
                tmpblobind = list()
                tmpblob.append(time[i+1])
                tmpblobind.append(i)
        print('Blob number is %d' % (blobnum))
    else:
        tmpblob = time[0].tolist()
        tmpblobind = 0
        for i in range(len(time)-1):
            # if true, combine those two index into one
            if (time[i+1][0]-tmpblob[-1]) <= iti/(1/fs):
                # print('These two are a same blob')
                tmpblob.extend(time[i+1].tolist())
                tmpblobind.extend(i+1)
                if i == len(time)-duration:
                    blob.append(tmpblob)
                    blobind.append(tmpblobind)
            # if false, set as a block and pass to next one
            elif ((time[i+1][0]-tmpblob[-1]) > iti/(1/fs)):
                if tmpblob[-1]-tmpblob[0] > duration*fs:
                    blob.append(tmpblob)
                    blobind.append(tmpblobind)
                    blobnum += 1
                # Set new start point
                tmpblob = list()
                tmpblobind = list()
                tmpblob = time[i+1].tolist()
                tmpblobind = i+1
        print('Blob number is %d' % (blobnum))
    return blob, blobind


def calculateDistance(x1, y1, x2, y2):
    import math
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist


def getAngle(a, b, c):
    import math
    ang = math.degrees(math.atan2(
        c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

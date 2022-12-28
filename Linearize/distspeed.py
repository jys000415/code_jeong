# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:50:02 2021

@author: yeong
"""

# Scale 
# output - Scale/ input - path info, smoothing, 
def scaleCal(mazelen,position,mask):
    from Linearize.areaSetting import areasetting
    from etc.etcfunc import calculateDistance
    # import os 
    # os.chdir(inpath)    
    edge = areasetting(position,mask)
    tmpp1 = edge.coord[0]; tmpp2 = edge.coord[1]
    scale = mazelen/calculateDistance(tmpp1[0],tmpp1[1],tmpp2[0],tmpp2[1])
    return scale

# Speed & Distance
def distSpeed(position,time,smoothinfo,scale): 
    from scipy.ndimage import gaussian_filter1d
    from etc.etcfunc import calculateDistance
    import numpy as np
    smoothx = gaussian_filter1d(position[:,0],smoothinfo)
    smoothy = gaussian_filter1d(position[:,1],smoothinfo)
    smoothpos = np.concatenate((smoothx.reshape(len(smoothx),1), smoothy.reshape(len(smoothy),1)),axis = 1)
    dist = []; speed = []
    diff_time = np.diff(time)
    for ind,(p1, p2) in enumerate(zip(smoothpos, smoothpos[1:])):
        dist.append(calculateDistance(p1[0],p1[1],p2[0],p2[1])*scale)
        speed.append(calculateDistance(p1[0],p1[1],p2[0],p2[1])*scale/diff_time[ind])
    return dist, speed

# Speed & Distance
def linearSpeed(position,time,smoothinfo): 
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    smoothpos = gaussian_filter1d(position,smoothinfo)
    speed = []
    diff_time = np.diff(time)
    diff_pos = np.abs(np.diff(smoothpos))
    speed = diff_pos/diff_time
    return speed


def areaspeed(position, time, linear, scale, smoothinfo, speedthreshold): 
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    from etc.etcfunc import calculateDistance
    speed = list(range(len(position)))
    speed_run = list(range(len(position)))
    position_speed = list(range(len(position)))
    time_speed = list(range(len(position)))
    linear_speed = list(range(len(position)))
    
    for iarea in range(len(position)):
        diff_time = np.diff(time[iarea])
        tmpposition = np.array(position[iarea])
        tmpx = gaussian_filter1d(tmpposition[:,0], smoothinfo)
        tmpy = gaussian_filter1d(tmpposition[:,1], smoothinfo)
        tmpspeed = [calculateDistance(tmpx[ipoint], tmpy[ipoint],
                          tmpx[ipoint+1], tmpy[ipoint+1])*scale/diff_time[ipoint]
                    for ipoint in range(len(diff_time))]
        # tmpspeed = gaussian_filter1d(tmpspeed, smoothinfo)
        speed[iarea] = tmpspeed
        speed_run[iarea] = [i for i in tmpspeed if i > speedthreshold]
        position_speed[iarea] = [element for idx, element in enumerate(
            position[iarea]) if tmpspeed[idx-1] > speedthreshold]
        time_speed[iarea] = [element for idx, element in enumerate(
            time[iarea]) if tmpspeed[idx-1] > speedthreshold]
        linear_speed[iarea] = [element for idx, element in enumerate(
            linear[iarea]) if tmpspeed[idx-1] > speedthreshold]
    return speed, position_speed, time_speed, linear_speed, speed_run


def speedDistDivTime(time,lowAnxTime,highAnxTime,dist,speed,duration,divtime,fs):
    import numpy as np
    divtime = 3
    DivDist = []; DivSpeed = []; lowAnxDivTime = []; highAnxDivTime = []
    for ibin in range(int(duration / divtime)):
        sp =  [speed[i-1] for i in np.where((time<time[0]+(divtime*(ibin+1))*60)&
                                                                    (time>=time[0]+(divtime*ibin)*60))[0]]
        ddist = [dist[i-1] for i in np.where((time<time[0]+(divtime*(ibin+1))*60)&
                                                                   (time>=time[0]+(divtime*ibin)*60))[0]]
        del sp[0]; del ddist[0]; 

        DivSpeed.append(sum(sp)/len(sp))
        DivDist.append(sum(ddist)/len(ddist))
        lowAnxDivTime.append(sum((lowAnxTime<time[0]+(divtime*(ibin+1))*60)&
                                             (lowAnxTime>=time[0]+(divtime*ibin)*60))*1/fs)
        highAnxDivTime.append(sum((highAnxTime<time[0]+(divtime*(ibin+1))*60)&
                                             (highAnxTime>=time[0]+(divtime*ibin)*60))*1/fs)
       
        
    return DivSpeed, DivDist, lowAnxDivTime, highAnxDivTime
        


# Extract position and time (Speed > 4cm/s)
def extract_pos_time(linear,time,running_speed,smooth_val):
    from Linearize.distspeed import linearSpeed
    import numpy as np
    speed = []; speed_linear = []; speed_time = []
    for ianimal, (ipos, itime) in enumerate(zip(linear,time)):
        speed.append(linearSpeed(ipos,itime,smooth_val)[linearSpeed(ipos,itime,smooth_val)>running_speed])    
        speed_linear.append(np.array(ipos[0:-1])[linearSpeed(ipos,itime,smooth_val)>running_speed])
        speed_time.append(np.array(itime[0:-1])[linearSpeed(ipos,itime,smooth_val)>running_speed])
    return speed,speed_linear,speed_time



# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:04:23 2021

@author: yeong
"""

# Spatial Decoding

# Required variable
# 1. Spatial tuning curve of each neuron - rate maps
# 2. Number of spike with temporal binning (5ms and 25ms boxcar filtering)
# 3. Animal's actual position in a given time window (5 ms)

import pickle
import numpy as np
context_name = 'EPM'
f = open('%s_Spatial_binned_speed_2cm.pckl' %(context_name),'rb')
epmdata = pickle.load(f)

prefercloseSpkAnimal_speed = epmdata[0]; prefercloseFrAnimal_speed = epmdata[1]; prefercloseTime_speed = epmdata[2];
othercloseSpkAnimal_speed = epmdata[3]; othercloseFrAnimal_speed = epmdata[4]; othercloseTime_speed = epmdata[5]
preferopenSpkAnimal_speed = epmdata[6]; preferopenFrAnimal_speed = epmdata[7]; preferopenTime_speed = epmdata[8]
otheropenSpkAnimal_speed = epmdata[9]; otheropenFrAnimal_speed = epmdata[10];otheropenTime_speed = epmdata[11]
gausscloseopenFrAnimal = epmdata[12]; centerSpkAnimal_speed = epmdata[13]; centerFrAnimal_speed = epmdata[14]
centerTime_speed = epmdata[-1]

f = open('%s_Linearize.pckl' %(context_name),'rb')
epm_data = pickle.load(f)
f.close()

f = open('EPM_Linearize_remap_1_2cm.pckl','rb')
remap_epm = pickle.load(f)
f.close()

preferclosedLinear = epm_data[0]+remap_epm[0]
otherclosedLinear = epm_data[1]+remap_epm[1]
preferopenLinear = epm_data[2]+remap_epm[2]
otheropenLinear = epm_data[3]+remap_epm[3]
centerLinear = epm_data[-2] + remap_epm[8]
preferclosedTime = epm_data[4]+remap_epm[4]
otherclosedTime = epm_data[5]+remap_epm[5]
preferopenTime = epm_data[6]+remap_epm[6]
otheropenTime = epm_data[7]+remap_epm[7]
centerTime = epm_data[-1]+remap_epm[9]


gausscloseFrAnimal = []; gaussopenFrAnimal = []
for ianimal in range(len(prefercloseFrAnimal_speed)):    
   tmpcenter =  np.mean(centerFrAnimal_speed[ianimal],axis = 1)
   tmpcenter = tmpcenter.reshape((len(tmpcenter),1))
   gausscloseFrAnimal.append(np.concatenate((prefercloseFrAnimal_speed[ianimal],tmpcenter,
                                             othercloseFrAnimal_speed[ianimal]),axis = 1))
   gaussopenFrAnimal.append(np.concatenate((preferopenFrAnimal_speed[ianimal],tmpcenter,
                                             otheropenFrAnimal_speed[ianimal]),axis = 1))
   
a = np.sum(gausscloseopenFrAnimal[0],axis = 1)/len(gausscloseopenFrAnimal[0][0,:])

num_spatial_bin = 36
center_spatial_bin = 6
binsize = 2
arm_length = 18

wholelinear = len(preferclosedLinear)*[None]
wholelineartime = len(preferclosedLinear)*[None]
closedlinear = len(preferclosedLinear)*[None]
closedlineartime = len(preferclosedLinear)*[None]
openlinear = len(preferclosedLinear)*[None]
openlineartime = len(preferclosedLinear)*[None]

binnum = int(num_spatial_bin/binsize)
bins = np.linspace(0,num_spatial_bin,binnum,endpoint = False)

centerbinnum = int(center_spatial_bin/binsize)
centerbins = np.linspace(0,center_spatial_bin,centerbinnum,endpoint = False)

for ianimal in range(len(preferclosedLinear)):
    preferclosed = [x for x in np.digitize(preferclosedLinear[ianimal],bins)]
    otherclosed = [x+arm_length for x in np.digitize(otherclosedLinear[ianimal],bins)]
    preferopen = [x+arm_length*2 for x in np.digitize(preferopenLinear[ianimal],bins)]
    otheropen = [x+arm_length*3 for x in np.digitize(otheropenLinear[ianimal],bins)]
    center = [x+arm_length*4 for x in np.digitize(centerLinear[ianimal],centerbins)]   
    
    tmptime = preferclosedTime[ianimal] + otherclosedTime[ianimal] + preferopenTime[ianimal]+ otheropenTime[ianimal] + centerTime[ianimal]
    timeorder = np.argsort(tmptime)
    
    tmppos = preferclosed + otherclosed + preferopen + otheropen + center
    wholelinear[ianimal] = [tmppos[x] for x in timeorder]
    wholelineartime[ianimal] = [tmptime[x] for x in timeorder]
    
    preferclosed = [abs(x-arm_length) for x in np.digitize(preferclosedLinear[ianimal],bins)]
    otherclosed = [-x for x in np.digitize(otherclosedLinear[ianimal],bins)]
    preferopen = [abs(x-arm_length) for x in np.digitize(preferopenLinear[ianimal],bins)]
    otheropen = [-x for x in np.digitize(otheropenLinear[ianimal],bins)]
   
    center = [0 for x in np.digitize(centerLinear[ianimal],centerbins)]   
    
    tmppos = preferclosed + center + otherclosed
    tmptime = preferclosedTime[ianimal] + centerTime[ianimal] + otherclosedTime[ianimal]
    timeorder = np.argsort(tmptime)    
    
    closedlinear[ianimal] = [tmppos[x] for x in timeorder]
    closedlineartime[ianimal] = [tmptime[x] for x in timeorder]

    tmppos = preferopen + center + otheropen
    tmptime = preferopenTime[ianimal] + centerTime[ianimal] + otheropenTime[ianimal]
    timeorder = np.argsort(tmptime)    
    
    openlinear[ianimal] = [tmppos[x] for x in timeorder]
    openlineartime[ianimal] = [tmptime[x] for x in timeorder]
    
    
f = open('Project1_spike_%s.pckl' %(context_name),'rb')
spk_data = pickle.load(f)
f.close()
f = open('Project1_spike_EPM_remap_1.pckl','rb')
spk_data_remap = pickle.load(f)
f.close()

taskspk = spk_data[2]+spk_data_remap[2]

f = open('Project1_position_time_EPM.pckl','rb')
position_time = pickle.load(f)
f.close()
f = open('Project1_position_time_EPM_remap_1.pckl','rb')
position_time_remap = pickle.load(f)
f.close()

f = open('figure1_%s.pckl' %(context_name),'rb')
figure1 = pickle.load(f)
f.close()
speedepm = figure1[6]
[speedepm[ianimal].pop(-1) for ianimal in range(len(speedepm))]

f = open('figure1_EPM_remap_1.pckl','rb')
figure1_remap = pickle.load(f)
f.close()
speed_remap = figure1_remap[6]
epmspeed = speedepm + speed_remap

epmtime = position_time[5]+position_time_remap[5]
duration = 15
for ianimal in range(len(epmtime)):  
    epmtime[ianimal] = epmtime[ianimal][np.where(epmtime[ianimal]<=epmtime[ianimal][0]+duration*60)[0].tolist()]
    epmspeed[ianimal] = epmspeed[ianimal][np.where(epmtime[ianimal]<=epmtime[ianimal][0]+duration*60)[0]]
runspeed = 2

speedtime = epmtime[0][np.where(np.array(epmspeed[0])>runspeed)]

# Speed Running 2 cm/s 
# Making Spike - Time template
def binning_spk(spk,time,speedtime,contexttime,binwidth,filter_length):    
    import numpy as np
    from scipy import signal
    timedata = np.arange(time[0],time[-1],binwidth)
    
    tmparmidx = []
    for idx in range(len(contexttime)):
        tmparmidx.append(np.argmin(abs(speedtime-contexttime[idx])))
    speedtime = speedtime[tmparmidx]
    
    tmpspeedidx = []
    for idx in range(len(speedtime)):
        tmpspeedidx.append(np.argmin(abs(timedata-speedtime[idx])))
    
    # rest_ind = [x for x in range(len(timedata)) if x not in tmpspeedidx]
    tmp_mat = np.zeros((len(spk),len(timedata)))
    filter_box = np.ones(filter_length)/filter_length
    fr_cell = []    
    for icell,idk in enumerate(spk):
        tmpspk = spk[idk]       
        tmpspk = tmpspk[tmpspk <= timedata[-1]]
        fr_cell.append(len(tmpspk)/(timedata[-1]-timedata[0]))
        tmp_cell = np.digitize(tmpspk,bins = timedata)
        tmp_mat[icell,tmp_cell-1] = 1
        tmp_mat[icell,:] = signal.convolve(tmp_mat[icell,:],filter_box,mode = 'same')
    run_spk = tmp_mat[:,tmpspeedidx]
    return tmp_mat, run_spk, speedtime, fr_cell

def binning_pos(pos,time,postime,speedtime,binwidth):    
    import numpy as np
    timedata = np.arange(time[0],time[-1],binwidth)
    timeposition = []
    for idx in range(len(speedtime)):
        tmpidx = np.argmin(abs(timedata-speedtime[idx]))
        timeposition.append(pos[np.argmin(abs(postime-timedata[tmpidx]))])
    tmp_mat = np.zeros(len(timedata))
    for idx in range(len(timedata)):        
        tmpidx = np.argmin(abs(postime-timedata[idx]))
        tmp_mat[idx] = pos[tmpidx]        
    return tmp_mat, timeposition

binned_spk,run_spk, armspeedtime, fr_cell = binning_spk(taskspk[0],epmtime[0],speedtime,closedlineartime[0],0.005, 5)
binned_pos,run_pos = binning_pos(closedlinear[0],epmtime[0],closedlineartime[0],armspeedtime,0.005)
tmptc = gausscloseFrAnimal[0]

# Divide Open and Closed arm and compare !!!
# Decoding calculation
def decoding_error(rate_map,armspeedtime,run_spk,pos):
    from math import exp
    import math
    occspatial = np.ones(len(rate_map[0,:]))/len(rate_map[0,:])
    p = np.zeros((len(armspeedtime),len(rate_map[0,:])))
    for ispatial in range(len(rate_map[0,:])):
        a = np.matlib.repmat(rate_map[:,ispatial],len(armspeedtime),1)
        b = a.transpose()
        c = run_spk*np.log(b)
        c = np.nansum(c,axis =0)
        tcsum = exp(-0.005*np.nansum(rate_map[:,ispatial]))
        p[:,ispatial] = np.exp(c)*tcsum*occspatial[ispatial]
    nActiveneurons = np.sum(run_spk,axis = 0)
    tmpsum = np.matlib.repmat(np.nansum(p,axis = 1),len(rate_map[0,:]),1)
    b = tmpsum.transpose()
    norm_p = p/tmpsum.transpose()
    norm_p[np.where(nActiveneurons < 0.4),:] = math.nan
    active_p = norm_p[np.where(nActiveneurons >= 0.4)]
    decoded_pos = (np.argmax(active_p,axis = 1)-18)*(-1)    
    active_linear = [pos[idx] for idx in np.where(nActiveneurons >= 0.4)[0]]
    err_pos = active_linear-decoded_pos
    return active_linear, decoded_pos, err_pos

active_linear, decoded_pos, err_pos = decoding_error(tmptc,armspeedtime,run_spk,closedlinear[0])

tmptc = gaussopenFrAnimal[0]
binned_spk,run_spk, armspeedtime, fr_cell = binning_spk(taskspk[0],epmtime[0],speedtime,openlineartime[0],0.005, 5)
active_linear_open, decoded_pos_open, err_pos_open = decoding_error(tmptc,armspeedtime,run_spk,openlinear[0])

decoded_err_closed = []; decoded_err_open = []
actual_pos_closed = []; actual_pos_open = []
decoded_pos_closed = []; decoded_pos_open = []

for ianimal in range(len(gausscloseopenFrAnimal)):
    closetc = gausscloseFrAnimal[ianimal]
    opentc = gaussopenFrAnimal[ianimal]
    speedtime = epmtime[ianimal][np.where(np.array(epmspeed[ianimal])>runspeed)]
    
    binned_spk,run_spk, armspeedtime, fr_cell = binning_spk(taskspk[ianimal],epmtime[ianimal],speedtime,openlineartime[ianimal],0.005, 5)
    active_linear, decoded_pos, err_pos = decoding_error(opentc,armspeedtime,run_spk,openlinear[ianimal])
    
    actual_pos_open.append(active_linear)
    decoded_pos_open.append(decoded_pos)
    decoded_err_open.append(err_pos)
    
    binned_spk,run_spk, armspeedtime, fr_cell = binning_spk(taskspk[ianimal],epmtime[ianimal],speedtime,closedlineartime[ianimal],0.005, 5)
    active_linear, decoded_pos, err_pos = decoding_error(closetc,armspeedtime,run_spk,closedlinear[ianimal])

    actual_pos_closed.append(active_linear)
    decoded_pos_closed.append(decoded_pos)
    decoded_err_closed.append(err_pos)
    
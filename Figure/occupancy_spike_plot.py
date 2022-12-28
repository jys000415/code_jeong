# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:18:08 2022

@author: yeong
"""
# %% Occupancy
import scipy.ndimage
from matplotlib.gridspec import GridSpec
for ijson in range(len(batchjson)):
  
    tmpname = batchjson[ijson]['ID']
    tmppathinfo = datapath+targetsite+optopath+batchname+contextname + "/" + tmpname
    os.chdir(tmppathinfo)
    tmpbatchjson = batchjson[ijson]
    occupancy = tmpbatchjson['L_Time_%s' % (contextname)]
    occupancy_speed = tmpbatchjson['L_Time_Speed_%s' % (contextname)]
    spike = tmpbatchjson['L_Spike_%s' % (contextname)]
    fr = tmpbatchjson['L_Firing_rate_%s' % (contextname)]
    # occupancy = gaussian_filter1d(occupancy, 2)
    
    tmppos = tmpbatchjson['pos_2d_%s' % (contextname)]
    tmptime = tmpbatchjson['time_2d_%s' % (contextname)]
    timearray = np.array(tmptime)
    
    tmpmazetime = tmpbatchjson['Time_%s' % (contextname)]
    tmpspk = tmpbatchjson['spk'][contextname]
    tmpscale = tmpbatchjson['Scale']
    ttmpclosedtime = np.concatenate((tmpmazetime[0], tmpmazetime[1]), axis=0)
    ttmpopentime = np.concatenate((tmpmazetime[3], tmpmazetime[4]), axis=0)
    centertime = tmpmazetime[2]
    tmppos3d = np.array(tmppos)
    tmptime3d = np.linspace(0, 100, num=len(tmppos3d))
    _, ind_closed, _ = np.intersect1d(tmptime, ttmpclosedtime,
                                              return_indices=True)
    closedind = ind_closed
    ttmpclosed_2d = np.array([tmppos[i] for i in ind_closed])
    
    _, ind_open, _ = np.intersect1d(tmptime, ttmpopentime,
                                                return_indices=True)
    openind = ind_open
    ttmpopen_2d = np.array([tmppos[i] for i in ind_open])
    
    _, ind_center, _ = np.intersect1d(tmptime, centertime,
                                                return_indices=True)
    centerind = ind_center
    ttmpcenter_2d = np.array([tmppos[i] for i in centerind])
    if np.max(ttmpclosed_2d[:, 1])-np.min(ttmpclosed_2d[:, 1]) > np.max(ttmpclosed_2d[:, 0])-np.min(ttmpclosed_2d[:, 0]):
        closedaxis = 1
        openaxis = 0
        project_closed = [1,0]
        project_open = [0,1]
    else:
        closedaxis = 0
        openaxis = 1
        project_closed = [0,1]
        project_open = [1,0]
        
    pos_2d = np.array(tmppos)
    smoothposclosed = gaussian_filter1d(pos_2d[:,closedaxis], 5)
    smoothposopen = gaussian_filter1d(pos_2d[:,openaxis], 5)
    closedpos_smooth = smoothposclosed[closedind]
    closedpos = pos_2d[closedind, closedaxis]
    speedrange = np.arange(0,38,2)
    minclosed = np.min(closedpos)
    maxclosed = np.max(closedpos)
    closedtime = tmptime3d[closedind]
    
    speedclosedtime = timearray[closedind]
    
    tmpclosedspeed  = []
    for idist in range(len(speedrange)):
        if idist > 0:
            tmpind = np.where((np.abs(closedpos_smooth*tmpscale) > speedrange[idist-1]) 
                              & (np.abs(closedpos_smooth*tmpscale) < speedrange[idist]))
            tmptime = np.diff(speedclosedtime[tmpind])        
            tmpdiff = np.diff(np.abs(closedpos_smooth[tmpind]*tmpscale))
            tmptime = tmptime[tmpdiff>0]
            tmpdiff = tmpdiff[tmpdiff>0]
            tmpclosedspeed.append(np.mean([i/j for i,j in zip(tmpdiff, tmptime)]))
    
    openpos_smooth = smoothposopen[openind]
    openpos = pos_2d[openind,openaxis]
    minopen = np.min(openpos)
    maxopen = np.max(openpos)
    opentime = tmptime3d[openind]
    speedopentime = timearray[openind]
    
    tmpopenspeed  = []
    for idist in range(len(speedrange)):
        if idist > 0:
            tmpind = np.where((np.abs(openpos_smooth*tmpscale) > speedrange[idist-1]) 
                              & (np.abs(openpos_smooth*tmpscale) < speedrange[idist]))
            tmptime = np.diff(speedopentime[tmpind])        
            tmpdiff = np.diff(np.abs(openpos_smooth[tmpind]*tmpscale))
            tmptime = tmptime[tmpdiff>0]
            tmpdiff = tmpdiff[tmpdiff>0]
            tmpopenspeed.append(np.mean([i/j for i,j in zip(tmpdiff, tmptime)]))
    
    
    centerpos = pos_2d[centerind, closedaxis]
    centertime = tmptime3d[centerind]
    overlap_closed = [i for i in range(len(closedpos)) if closedpos[i] in pos_2d[centerind, closedaxis]]
    overlap_open = [i for i in range(len(openpos)) if openpos[i] in pos_2d[centerind, openaxis]]
    closedpos[overlap_closed] = np.NaN
    openpos[overlap_open] = np.NaN
    if len(np.where(closedpos<0)[0]) > len(np.where(closedpos>0)[0]):
        newclosedpos = (closedpos - minclosed)*tmpscale/2   
    else:
        newclosedpos = np.abs(closedpos - maxclosed)*tmpscale/2 
    if len(np.where(openpos<0)[0]) > len(np.where(openpos>0)[0]):
        newopenpos = (openpos - minopen)*tmpscale/2+39
    else:
        newopenpos = np.abs(openpos - maxopen)*tmpscale/2+39
      
    newcenterpos = (centerpos - np.min(centerpos))*tmpscale/2+36    
    tmptime = tmpbatchjson['time_2d_%s' % contextname]
    
    pos_3d = (np.concatenate(
                (tmppos3d, tmptime3d.reshape(len(tmptime), 1)), axis=1))
    openscale = tmpbatchjson['Scale']
    tmpaxis = tmpbatchjson['Axis_%s' % (contextname)]
    opendist = calculateDistance(tmpaxis[3][0][0], tmpaxis[3][0][1],
                                 tmpaxis[4][0][0], tmpaxis[4][0][1])
    width = 45
    mazearm = 40/openscale
    closed_coord = [[-width/2, width/2], [-width/2, mazearm], [width/2, mazearm],
                    [width/2, width/2], [width/2, -
                                         width / 2], [width/2, -mazearm],
                    [-width/2, -mazearm], [-width/2, -width/2]]
    closed_coord = np.vstack(closed_coord)
    open_coord = [[width/2, width/2], [opendist, width/2],
                  [opendist, -width/2], [width/2, -width / 2],
                  [-width/2, -width/2], [-opendist, -width/2],
                  [-opendist, width/2], [-width/2, width/2]]
    open_coord = np.vstack(open_coord)
    
    for icell, iname in enumerate(tmpspk):
        tmpspkpos = []
        for ispk in tmpspk[iname]:
            tmpspkpos.append(np.argmin(abs(np.array(tmptime)-ispk)))
        fig = plt.figure(figsize = (9,8))
        gs = GridSpec(nrows = 3, ncols = 2)
        ax1 = fig.add_subplot(gs[:,0], projection= '3d')
        ax1.axis('off')
        ax1.plot3D(pos_3d[:, 0], pos_3d[:, 1], 
                          pos_3d[:, 2], 'slategray', linewidth=.4, zorder=0)
       
        ax1.scatter(pos_3d[tmpspkpos, 0], 
                               pos_3d[tmpspkpos, 1],
                               pos_3d[tmpspkpos, 2], s=5, color='red', zorder=10)
        ax1.plot(closed_coord[:, 0], closed_coord[:, 1],
                   color='lightslategrey', linewidth=2)
        ax1.plot(open_coord[:, 0], open_coord[:, 1],
                   color='lightgrey', linewidth=2)
        ax1.set_zlim([0, 100])
        ax1.set_ylim([-160, 160])
        ax1.set_xlim([-160, 160])
        ax1.view_init(45, 130)
        
        
        fontsize = 15
        tmpspkpos_2d = pos_2d[tmpspkpos]
        _, spk_close, _ = np.intersect1d(closedpos,tmpspkpos_2d[:, closedaxis], return_indices= True)
        _, spk_open, _ = np.intersect1d(openpos,tmpspkpos_2d[:, openaxis], return_indices= True)
        _, spk_center, _ = np.intersect1d(centerpos,tmpspkpos_2d[:, closedaxis], return_indices= True)
           
        ax2 = fig.add_subplot(gs[0,1])
        ax2.scatter(newclosedpos[spk_close], closedtime[spk_close], s = 15, 
                    marker = 's',color = 'black', alpha = .4)
        ax2.scatter(newopenpos[spk_open], opentime[spk_open], s = 15,
                     marker = 's', color = 'black', alpha = .4)
        ax2.scatter(newcenterpos[spk_center], centertime[spk_center], s = 15,
                     marker = 's', color = 'black', alpha = .4)
        ax2.set_title('Raster plot', fontsize = fontsize)
        ax2.set_ylim(0, 100)
        ax2.set_xlim(0,75)
        ax2.axvline(x=18, linestyle='--', color='gray', alpha = .4)
        ax2.axvline(x=36, linestyle='-', color='k', alpha = .4)
        ax2.axvline(x=57, linestyle='--', color='gray', alpha = .4)
        ax2.axvline(x=39, linestyle='-', color='k', alpha = .4)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        
        ax3 = fig.add_subplot(gs[1,1])
        smooth_occupancy = gaussian_filter1d(occupancy,2)
        ax3.plot(smooth_occupancy)
        ax3.set_title('Occupancy', fontsize = fontsize)
        ax3.set_ylim(0, np.max(smooth_occupancy)*1.2)
        scale =np.max(smooth_occupancy)*1.2/100
        ax3.set_xlim(0,75)
        ax3.set_ylabel('Time (s)')
        ax3.axvline(x=18, linestyle='--', color='gray', alpha = .4)
        ax3.axvline(x=36, linestyle='-', color='k', alpha = .4)
        ax3.axvline(x=57, linestyle='--', color='gray', alpha = .4)
        ax3.axvline(x=39, linestyle='-', color='k', alpha = .4)
        ax3.axes.xaxis.set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        
        # firingrate = [i/j for i, j in zip(spike[-1],occupancy)]
        ax4 = fig.add_subplot(gs[2,1])
        firingrate = gaussian_filter1d(fr[icell],2)
        ax4.plot(firingrate)
        scale = np.max(firingrate)*1.2/100
        ax4.set_ylim(0, np.max(firingrate)*1.2)
        ax4.set_xlim(0,75)
        ax4.add_patch(plt.Rectangle((36,-15*scale),3,15*scale,clip_on=False,
                                       facecolor = 'lightslategrey', edgecolor = 'k',
                                       linewidth = 1))
        ax4.add_patch(plt.Rectangle((0,-15*scale),36,15*scale,clip_on=False,
                                       facecolor = 'lightslategrey', edgecolor = 'k',
                                       linewidth = 4))
        ax4.add_patch(plt.Rectangle((39,-15*scale),36,15*scale,clip_on=False,
                                       facecolor = 'lightgrey', edgecolor = 'k',
                                       linewidth = 1))
        ax4.text(2,-11*scale, 'Closed', fontsize = 10)
        ax4.text(24,-11*scale, 'Closed', fontsize = 10)
        ax4.text(41,-11*scale, 'Open', fontsize = 10)
        ax4.text(65,-11*scale, 'Open', fontsize = 10)
        ax4.axvline(x=18, linestyle='--', color='gray', alpha = .4)
        ax4.axvline(x=36, linestyle='-', color='k', alpha = .4)
        ax4.axvline(x=57, linestyle='--', color='gray', alpha = .4)
        ax4.axvline(x=39, linestyle='-', color='k', alpha = .4)
        ax4.set_ylabel('Firing rate (Hz)')
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.axes.xaxis.set_visible(False)
        ax4.set_title('Firing rate', fontsize = fontsize)
        plt.suptitle('%s' % (iname))
        plt.savefig('%s' % (iname))
    
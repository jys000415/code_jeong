# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:04:21 2022

@author: yeong
"""
#%% Plotting indiviudal neuron with trajectory 
# %% 2d & 3d plotting (trajectory, spikes across time)
warnings.filterwarnings('ignore')
for ijson in range(len(batchjson)):
    tmpname = batchjson[ijson]['ID']
    tmppathinfo = datapath+targetsite+optopath+batchname+contextname + "/" + tmpname
    os.chdir(tmppathinfo)
    tmpdata = batchjson[ijson]
    key_var = tmpdata.keys()
    tmppos = batchjson[ijson]['pos_2d_EPM']
    tmptime = batchjson[ijson]['time_2d_EPM']
    tmpepmtime = batchjson[ijson]['Time_EPM']
    tmpspk = batchjson[ijson]['spk']['EPM']
    tmpactive = batchjson[ijson]['Active_neuron']
    tmpinactive = batchjson[ijson]['Inactive_neuron']
    tmpplace = batchjson[ijson]['Place_cell_ind_EPM']
    tmpaxis = batchjson[ijson]['Axis_EPM']
    
    if 'Linearized_EPM_2' in key_var:
        remap = 1
        tmppos_2 = batchjson[ijson]['pos_2d_EPM_2']
        tmptime_2 = batchjson[ijson]['time_2d_EPM_2']
        tmpepmtime_2 = batchjson[ijson]['Time_EPM_2']
        tmpspk_2 = batchjson[ijson]['spk']['EPM_2']
        tmpaxis_2 = batchjson[ijson]['Axis_EPM_2']
        tmppos = [tmppos, tmppos_2]
        tmptime = [tmptime, tmptime_2]
        tmpepmtime = [tmpepmtime, tmpepmtime_2]
        tmpspk = [tmpspk, tmpspk_2]
        tmpaxis = [tmpaxis, tmpaxis_2]
    else:
        remap = 0
        tmppos = [tmppos]
        tmptime = [tmptime]
        tmpepmtime = [tmpepmtime]
        tmpspk = [tmpspk]
        tmpaxis = [tmpaxis]
    
    if 'Time_Openfield' in key_var:
        of_plot = 1
        ofpos = batchjson[ijson]['Dist_Openfield']
        oftime = batchjson[ijson]['Time_Openfield'][5]
        openspk = batchjson[ijson]['spk']['Openfield']
        openpos_2d = batchjson[ijson]['pos_2d_Openfield']
        opentime_2d = batchjson[ijson]['time_2d_Openfield']
    else:
        of_plot = 0

    closedtime = []
    opentime = []
    centertime = []
    closedind = []; openind = []
    closed2d = []; open2d = []
    gradient_time = []; gradient_time_open = []
    pos_3d = []; gradient_time_3d = []
    for imaze in range(remap+1):
        ttmpclosedtime = np.concatenate((tmpepmtime[imaze][0], tmpepmtime[imaze][1]), axis=0)
        closedtime.append(ttmpclosedtime)
        ttmpopentime = np.concatenate((tmpepmtime[imaze][3], tmpepmtime[imaze][4]), axis=0)
        opentime.append(ttmpopentime)
        centertime.append(tmpepmtime[imaze][2])
        _, ind_closed, _ = np.intersect1d(tmptime[imaze], ttmpclosedtime,
                                                  return_indices=True)
        closedind.append(ind_closed)
        ttmpclosed_2d = np.array([tmppos[imaze][i] for i in ind_closed])
        closed2d.append(ttmpclosed_2d)
        gradient_time.append(np.linspace(0, 100, num=len(ttmpclosed_2d)))

        _, ind_open, _ = np.intersect1d(tmptime[imaze], ttmpopentime,
                                                    return_indices=True)
        openind.append(ind_open)
        ttmpopen_2d = np.array([tmppos[imaze][i] for i in ind_open])
        open2d.append(ttmpopen_2d)
        gradient_time_open.append(np.linspace(0, 100, num=len(ttmpopen_2d)))
        tmppos3d = np.array(tmppos[imaze])
        tmptime3d = np.linspace(0, 100, num=len(tmppos3d))
        gradient_time_3d.append(tmptime3d)
        pos_3d.append(np.concatenate(
            (tmppos3d, tmptime3d.reshape(len(tmptime[imaze]), 1)), axis=1))

    openscale = tmpdata['Scale']
    opendist = calculateDistance(tmpaxis[0][3][0][0], tmpaxis[0][3][0][1],
                                 tmpaxis[0][4][0][0], tmpaxis[0][4][0][1])
    width = 45
    mazearm = 40/openscale
    closed_coord = [[-width/2, width/2], [-width/2, mazearm], [width/2, mazearm],
                    [width/2, width/2], [width/2, -
                                         width / 2], [width/2, -mazearm],
                    [-width/2, -mazearm], [-width/2, -width/2]]
    open_coord = [[width/2, width/2], [opendist, width/2],
                  [opendist, -width/2], [width/2, -width / 2],
                  [-width/2, -width/2], [-opendist, -width/2],
                  [-opendist, width/2], [-width/2, width/2]]
    
    open_coord2 = [[-width/2, width/2], [-width/2, mazearm], [width/2, mazearm],
                    [width/2, width/2], [width/2, -
                                         width / 2], [width/2, -mazearm],
                    [-width/2, -mazearm], [-width/2, -width/2]]
    closed_coord2 = [[width/2, width/2], [opendist, width/2],
                  [opendist, -width/2], [width/2, -width / 2],
                  [-width/2, -width/2], [-opendist, -width/2],
                  [-opendist, width/2], [-width/2, width/2]]
    
    closed_coord1 = np.vstack(closed_coord)
    open_coord1 = np.vstack(open_coord)
    closed_coord2 = np.vstack(closed_coord2)
    open_coord2 = np.vstack(open_coord2)
    closed_whole = [closed_coord1, closed_coord2]
    open_whole = [open_coord1, open_coord2]
    cellname = list(tmpspk[0].keys())
    
    fontsize = 25
    for idx, icell in enumerate(cellname):
        if (icell in tmpactive) & (idx in tmpplace):
            fig_name = 'Trajectory_Spike_%s_Active_Place' % (icell)
        elif (icell in tmpactive) & (idx not in tmpplace):
            fig_name = 'Trajectory_Spike_%s_Active' % (icell)
        elif (icell in tmpinactive) & (idx in tmpplace):
            fig_name = 'Trajectory_Spike_%s_Inactive_Place' % (icell)
        elif (icell in tmpinactive) & (idx not in tmpplace):
            fig_name = 'Trajectory_Spike_%s_Inactive' % (icell)
        elif (idx in tmpplace) & (idx in tmpplace):
            fig_name = 'Trajectory_Spike_%s_non_Place' % (icell)
        else:
            fig_name = 'Trajectory_Spike_%s_non' % (icell)
        remap_figure = remap+1
        row_num_figure = 2*remap_figure+of_plot
        fig, axs = plt.subplots(2, row_num_figure, figsize=(20, 8), constrained_layout = False)
        # fig.tight_layout()
      
        if 'Time_Openfield' in key_var:
            tmpspkpos = []
            for ispk in openspk[icell]:
                tmpspkpos.append(np.argmin(abs(np.array(oftime)-ispk)))
            gradient_time_of = np.linspace(0, 100, num=len(oftime))    
            
            axs[0, 2*remap_figure].plot(ofpos, gradient_time_of,
                     linewidth=.2, color='slategray', zorder=0)
            
            wholedist = np.array(ofpos)
            axs[0, 2*remap_figure].scatter(wholedist[tmpspkpos], gradient_time_of[tmpspkpos],
                        color='red', s=5, zorder=5)
            bottom, top = axs[0, 2*remap_figure].get_ylim()
            axs[0, 2*remap_figure].plot([10, 10], [bottom, top], linestyle="--",
                     linewidth=2, color='black')
            axs[0, 2*remap_figure].set_xticks([5, 20])
            axs[0, 2*remap_figure].set_yticks([])
            axs[0, 2*remap_figure].set_xticklabels(['Center', 'Corner'], fontsize=fontsize)
            axs[0, 2*remap_figure].tick_params(axis=u'both', which=u'both', length=0)
            arrow = mpatches.FancyArrowPatch((0, 0), (0, 100),
                                             mutation_scale=20, color='dimgrey')
            axs[0, 2*remap_figure].add_patch(arrow)
            axs[0, 2*remap_figure].set_ylabel('Time in Openfield', fontsize=fontsize)
            # Heatmap plotting from here
            num_bin = 30
            projectpos = np.array(openpos_2d)
            occupancy, xedge, yedge = np.histogram2d(projectpos[:, 0], projectpos[:, 1],
                                                     bins=num_bin)
            spikenum, xedge, yedge = np.histogram2d(projectpos[tmpspkpos, 0], projectpos[tmpspkpos, 1],
                                                    bins=[xedge, yedge])
            tmpfr = spikenum/(occupancy*0.025)
            tmpfr[np.isinf(tmpfr)] = float('NaN')
            mask = np.isnan(tmpfr)
            tmpfr[np.isnan(tmpfr)] = 0
            tmpfr_gauss = sp.ndimage.gaussian_filter(tmpfr, sigma=3, truncate=3)
            tmpfr_gauss[mask] = float('NaN')
            axs[1,2*remap_figure].set_aspect('equal')
            img = axs[1, 2*remap_figure].imshow(tmpfr_gauss, cmap = 'coolwarm')
            cbar = colorbar(img, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize = fontsize*.7)
            axs[1, 2*remap_figure].axis('off')        
            axs[1, 2*remap_figure].set_title('Firing rate (Hz)', fontsize = fontsize)

        for imaze in range(remap+1): 
            closed_coord = closed_whole[imaze]
            open_coord = open_whole[imaze]
            tmpspkpos = []
            for ispk in tmpspk[imaze][icell]:
                tmpspkpos.append(np.argmin(abs(np.array(tmptime[imaze])-ispk)))
            if np.max(closed2d[imaze][:, 1])-np.min(closed2d[imaze][:, 1]) > np.max(closed2d[imaze][:, 0])-np.min(closed2d[imaze][:, 0]):
                closedaxis = 1
                openaxis = 0
                project_closed = [1,0]
                project_open = [0,1]
            else:
                closedaxis = 0
                openaxis = 1
                project_closed = [0,1]
                project_open = [1,0]
            pos_2d = tmppos[imaze]
            pos_2d = np.array(pos_2d)
            close_pos_2d = pos_2d.copy()
            close_pos_2d[openind[imaze],:] = 0
            open_pos_2d = pos_2d.copy()
            open_pos_2d[closedind[imaze],:] = 0
            spk_close = np.array([closedind[imaze][i] for i in range(len(closedind[imaze]))
                                         if closedind[imaze][i] in tmpspkpos])
            spk_open = np.array([openind[imaze][i] for i in range(len(openind[imaze]))
                                         if openind[imaze][i] in tmpspkpos])
            axs[0, imaze*2].plot(close_pos_2d[:, closedaxis], gradient_time_3d[imaze],
                           linewidth=1, color='slategray', zorder=0)
            if not (len(closedind[imaze])*len(spk_close) == 0):
                axs[0, imaze*2].scatter(close_pos_2d[spk_close, closedaxis],
                                  gradient_time_3d[imaze][spk_close], color='red',
                                  s=5, zorder=5)
            axs[0, imaze*2].set_xticks([])
            axs[0, imaze*2].set_yticks([])
            tmpminmax = [np.max(close_pos_2d[:, closedaxis]),
                         np.abs(np.min(close_pos_2d[:, closedaxis]))]
            xmax = np.max(tmpminmax)+50
            xarrow = xmax-25
            axs[0, imaze*2].set_xlim([-xmax, xmax])
            arrow = mpatches.FancyArrowPatch((-xarrow, 0), (-xarrow, 100),
                                             mutation_scale=20, color='dimgrey')
            axs[0, imaze*2].add_patch(arrow)
            axs[0, imaze*2].set_ylabel('Time in EPM', fontsize=fontsize)
            axs[0, imaze*2].set_xlabel('Closed Arm', fontsize=fontsize)
            bottom, top = axs[0, imaze*2].get_ylim()
            axs[0, imaze*2].plot([0, 0], [bottom, top], linestyle="--",
                           linewidth=2, color='black')
            axs[0, imaze*2].set_xticks([])
            axs[0, imaze*2+1].plot(open_pos_2d[:, openaxis], gradient_time_3d[imaze],
                           linewidth=1, color='slategray')
            if not (len(openind[imaze])*len(spk_open) == 0):
                axs[0, imaze*2+1].scatter(open_pos_2d[spk_open, openaxis],
                                  gradient_time_3d[imaze][spk_open],
                                  color='red', s=5, zorder=5)
            axs[0, imaze*2+1].plot([0, 0], [bottom, top], linestyle="--",
                           linewidth=2, color='black')
            axs[0, imaze*2+1].set_xticks([])
            tmpminmax = [np.max(open_pos_2d[:, openaxis]),
                         np.abs(np.min(open_pos_2d[:, openaxis]))]
            xmax = np.max(tmpminmax)+50
            xarrow = xmax-25
            axs[0, imaze*2+1].set_xlim([-xmax, xmax])
            axs[0, imaze*2+1].set_yticks([])
            arrow = mpatches.FancyArrowPatch((-xarrow, 0), (-xarrow, 100),
                                             mutation_scale=20, color='dimgrey', zorder=0)
            axs[0, imaze*2+1].add_patch(arrow)
            axs[0, imaze*2+1].set_ylabel('Time in EPM', fontsize=fontsize)
            axs[0, imaze*2+1].set_xlabel('Open Arm', fontsize=fontsize)
            
            fig_3dpos = row_num_figure+1+2*imaze
            axs[1, imaze*2].remove()
            axs[1, imaze*2] = fig.add_subplot(2,row_num_figure,fig_3dpos, projection= '3d')
            axs[1, imaze*2].axis('off')
            axs[1, imaze*2].plot3D(pos_3d[imaze][:, 0], pos_3d[imaze][:, 1], 
                              pos_3d[imaze][:, 2], 'slategray', linewidth=.4, zorder=0)
            if len(tmpspkpos):
                axs[1, imaze*2].scatter(pos_3d[imaze][tmpspkpos, 0], 
                                   pos_3d[imaze][tmpspkpos, 1],
                                   pos_3d[imaze][tmpspkpos, 2], color='red', s=5, zorder=10)
            axs[1, imaze*2].arrow3D(width/2*1.2, mazearm*1.2, 0, 0, -10, 50, mutation_scale=20,
                          ec='dimgrey', fc='dimgrey')
            axs[1, imaze*2].plot(closed_coord[:, 0], closed_coord[:, 1],
                       color='saddlebrown', linewidth=2)
            axs[1, imaze*2].plot(open_coord[:, 0], open_coord[:, 1],
                       color='darkorange', linewidth=2)
            axs[1, imaze*2].set_zlim([0, 150])
            axs[1, imaze*2].view_init(40, -50)
            
            # Spatial firing rate (heat map)
            axs[1, imaze*2+1].remove()
            axs[1, imaze*2+1] = fig.add_subplot(2,row_num_figure,
                                             fig_3dpos+1, projection='3d')
            # Calculate firing rate
            num_bin = 30
            projectpos = np.array(tmppos[imaze])
            tmpproject = projectpos[openind[imaze], closedaxis]
            tmpproject[tmpproject < -width/2] = -width/2
            tmpproject[tmpproject > width/2] = width/2
            projectpos[openind[imaze], closedaxis] = tmpproject
            
            xbin = np.linspace(-opendist*.9,opendist*.9,num_bin+1)
            ybin = np.linspace(-opendist*.9,opendist*.9,num_bin+1)
            occupancy, xedge, yedge = np.histogram2d(projectpos[:, project_closed[imaze]], 
                                                     projectpos[:, project_open[imaze]],
                                                     bins=[xbin, ybin])
            spikenum, xedge, yedge = np.histogram2d(projectpos[tmpspkpos, project_closed[imaze]],
                                                    projectpos[tmpspkpos, project_open[imaze]],
                                                    bins=[xedge, yedge])
            a = spikenum/(occupancy*0.025)
            a[np.isinf(a)] = float('NaN')
            mask = np.isnan(a)
            a[np.isnan(a)] = 0
            aa = sp.ndimage.gaussian_filter(a, sigma=2, truncate=4)
            aa[mask] = float('NaN')
            X, Y = np.meshgrid(np.linspace(np.min(closed_coord[:, imaze-1]), 
                                           np.max(closed_coord[:, imaze-1]), num_bin),
                               np.linspace(np.min(open_coord[:, imaze]),
                                           np.max(open_coord[:, imaze]), num_bin))
            img =  axs[1, imaze*2+1].plot_surface(X, Y, aa, cmap='coolwarm', linewidth=0,
                                     rstride=1, cstride=1, antialiased=False,
                                     vmin=np.nanmin(aa), vmax=np.nanmax(aa))
            axs[1, imaze*2+1].set_zlim(0, 1000)
            axs[1, imaze*2+1].axis('off')
            axs[1, imaze*2+1].plot(closed_coord[:, 0], closed_coord[:, 1],
                       color='saddlebrown', linewidth=2)
            axs[1, imaze*2+1].plot(open_coord[:, 0], open_coord[:, 1],
                       color='darkorange', linewidth=2)
            axs[1, imaze*2+1].view_init(40, -50)
            
            dia_size = np.diag([1.8, 1.8, 1.8, 1])
            axs[1, 0].get_proj = lambda: np.dot(
                Axes3D.get_proj(axs[1, 0]), dia_size)
            axs[1, 2].get_proj = lambda: np.dot(
                Axes3D.get_proj(axs[1, 2]), dia_size)
            
            axs[1, 1].get_proj = lambda: np.dot(
                Axes3D.get_proj(axs[1, 1]), dia_size)
            axs[1, 3].get_proj = lambda: np.dot(
                Axes3D.get_proj(axs[1, 3]), dia_size)
            
            axs[1, imaze*2+1].set_title('Firing rate (Hz)', fontsize=fontsize)
            cbar = colorbar(img, fraction=0.04, pad=0.04)
            cbar.ax.tick_params(labelsize = fontsize*.7)

        fig.suptitle(fig_name, fontsize = fontsize*1.5)
        fig.savefig(tmpname+fig_name)
        plt.show(block=False)
        plt.close()
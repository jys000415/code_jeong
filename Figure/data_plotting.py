# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:45:27 2021

@author: yeong
"""

def plot_dataframe(dataset,color_table,xy_label, yrange, figure_name, title, print_val):
    all_dataframe = dataset.melt(var_name = xy_label[0], value_name = xy_label[1])
    import seaborn as sns, matplotlib.pyplot as plt
    import numpy as np
    clrs = color_table
    fig, ax = plt.subplots()
    sns.set(style = 'white',rc = {'figure.figsize':(10,10)}, font_scale = 2)
    sns.barplot(x = xy_label[0], y = xy_label[1], data = all_dataframe,
                     palette = clrs, capsize = .1).set_title(title)
    sns.swarmplot(x = xy_label[0], y = xy_label[1], data = all_dataframe, 
                  palette = ['black','black','black'],alpha = .6,size = 8)
    columnname = dataset.columns.values.tolist()
    # idx0 = 0
    # idx1 = 1
    # locs1 = ax.get_children()[idx0].get_offsets()
    # locs2 = ax.get_children()[idx1].get_offsets()
    # sort_idxs1 = np.argsort(dataset[columnname[idx0]])
    # sort_idxs2 = np.argsort(dataset[columnname[idx1]])
    # locs2_sorted = locs2[sort_idxs2.argsort()][sort_idxs1]
    # if len(columnname) > 2:
    #     idx0 = 2
    #     idx1 = 3
    #     locs3 = ax.get_children()[idx0].get_offsets()
    #     locs4 = ax.get_children()[idx1].get_offsets()
    #     sort_idxs3 = np.argsort(dataset[columnname[idx0]])
    #     sort_idxs4 = np.argsort(dataset[columnname[idx1]])
    #     locs4_sorted = locs4[sort_idxs4.argsort()][sort_idxs3]
    #     for i in range(locs3.shape[0]):
    #         x = [locs3[i,0], locs4_sorted[i,0]]
    #         y = [locs3[i,1], locs4_sorted[i,1]]
    #         ax.plot(x, y, color = 'black', alpha = 0.3)
    
    # for i in range(locs1.shape[0]):
    #     x = [locs1[i,0], locs2_sorted[i,0]]
    #     y = [locs1[i,1], locs2_sorted[i,1]]
    #     ax.plot(x, y, color = 'black', alpha = 0.3)
    
    
    sns.despine()
    plt.yticks(fontsize = 20)
    plt.title(title, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.axis('tight')
    plt.ylim(yrange)    
    plt.ylabel('')
    if print_val:
        plt.savefig(figure_name)
        



def plot_dataframe_box(data, color_table, figure_name, yname, yrange, print_val):
    import seaborn as sns, matplotlib.pyplot as plt
    clrs = color_table
    plt.figure(figsize=(8,8))
    sns.set(style = "white", font_scale = 1.6)
    sns.boxplot(data = data, palette = clrs, linewidth=2.5, width = .7,
                showfliers=False, showmeans = True).set_title(figure_name)
    sns.stripplot(data = data, color = "0",alpha = .4, size = 4.5)
    sns.despine()
    plt.ylabel(yname)
    plt.ylim(yrange)
    plt.show()
    if print_val:
        plt.savefig(figure_name)


def plot_line(data, color_table, figure_name, xname, yname, yrange, print_val):
    import seaborn as sns, matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.plot(data, color = color_table[0],
             marker = 'o', linewidth = 1.5, markerfacecolor = color_table[1])
    plt.xticks(xname[0],xname[1])
    sns.despine()
    plt.ylabel(yname)
    plt.ylim(yrange)
    plt.title(figure_name)
    if print_val:
        plt.savefig(figure_name)
        

def plot_heatmap(smoothed_data,norm_method,sort_val,print_val,
                 figure_title,xtick_label,xy_label,fig_size,cmap,idorder, 
                 fontsize, box_plt, contextname, armlength, centerlength, remap):
    import numpy as np   
    import matplotlib.patches as mpatches
    if norm_method == 'mean':
        mean_data = np.mean(smoothed_data,axis = 1)[:,np.newaxis]
        max_data = np.max(smoothed_data,axis = 1)[:,np.newaxis]
        min_data = np.min(smoothed_data,axis = 1)[:,np.newaxis]
        norm_data = (smoothed_data-mean_data)/(max_data-min_data)
    elif norm_method == 'peak':
        mean_data = np.mean(smoothed_data,axis = 1)[:,np.newaxis]
        max_data = np.max(smoothed_data,axis = 1)[:,np.newaxis]
        min_data = np.min(smoothed_data,axis = 1)[:,np.newaxis]
        maxfr = smoothed_data.max(axis =1)[:,np.newaxis]
        norm_data = smoothed_data/maxfr
    elif norm_method =='none':
        norm_data = smoothed_data
    maxplaceind =[]
    # Align to max position
    if sort_val:
        if len(idorder)==0:    
            maxplaceind = norm_data.argmax(axis = 1)
        else:
            maxplaceind = idorder
        z = np.insert(norm_data,0,maxplaceind,axis = 1)
        sortedZ = z[z[:,0].argsort()]
        sortedZ = np.delete(sortedZ,0,1)
    else:
        sortedZ = norm_data
    font = {'size':fontsize}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=fig_size)
    im = axes.imshow(sortedZ, cmap = cmap)
    plt.xlabel(xy_label[0],fontdict = font)
    plt.ylabel(xy_label[1],fontdict = font)
    plt.xticks(xtick_label[0],xtick_label[1],fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    
    if box_plt:
        ylimit = axes.get_ylim()
        xlimit = axes.get_xlim()
        ymax = np.max(ylimit)
        scale = ymax/250
        axes.set_xlim(0,np.shape(sortedZ)[1])
        axes.set_ylim(0,np.shape(sortedZ)[0])
        if contextname == 'EPM':
            axes.add_patch(plt.Rectangle((armlength-.5,-15*scale),centerlength,15*scale,clip_on=False,
                                           facecolor = 'lightgrey', edgecolor = 'k',
                                           linewidth = 1))
        axes.add_patch(plt.Rectangle((0,-15*scale),armlength-.5,15*scale,clip_on=False,
                                       facecolor = 'lightslategrey', edgecolor = 'k',
                                       linewidth = 4))
        axes.add_patch(plt.Rectangle((armlength+centerlength-.5,-15*scale),armlength,15*scale,clip_on=False,
                                       facecolor = 'lightgrey', edgecolor = 'k',
                                       linewidth = 1))
    cbar = fig.colorbar(im)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    axes.grid(False)
    axes.axis('tight')
    axes.set_xlim(-.5,np.shape(sortedZ)[1]-0.5)
    axes.set_ylim(0,np.shape(sortedZ)[0]-0.5)
    axes.invert_yaxis()
    if remap:
        axes.invert_xaxis()
        axes.text(16,-3*scale, 'Closed', fontsize = 20)
        axes.text(armlength-2,-3*scale, 'Closed', fontsize = 20)
        axes.text(armlength+centerlength+12,-3*scale, 'Open', fontsize = 20)
        axes.text(np.max(xlimit)-2,-3*scale, 'Open', fontsize = 20)
    else:
        axes.text(2,-3*scale, 'Closed', fontsize = 20)
        axes.text(armlength-17,-3*scale, 'Closed', fontsize = 20)
        axes.text(armlength+centerlength+2,-3*scale, 'Open', fontsize = 20)
        axes.text(np.max(xlimit)-13,-3*scale, 'Open', fontsize = 20)
    if print_val:
        plt.savefig(figure_title)
        
    return maxplaceind, sortedZ


def plot_heatmap_overlap(smoothed_data,norm_method,sort_val,print_val,
                 figure_title,xtick_label,xy_label,fig_size,cmap,idorder, 
                 fontsize, box_plt, contextname, armlength, centerlength, remap):
    import numpy as np   
    import matplotlib.patches as mpatches
    if norm_method == 'mean':
        mean_data = np.mean(smoothed_data,axis = 1)[:,np.newaxis]
        max_data = np.max(smoothed_data,axis = 1)[:,np.newaxis]
        min_data = np.min(smoothed_data,axis = 1)[:,np.newaxis]
        norm_data = (smoothed_data-mean_data)/(max_data-min_data)
    elif norm_method == 'peak':
        mean_data = np.mean(smoothed_data,axis = 1)[:,np.newaxis]
        max_data = np.max(smoothed_data,axis = 1)[:,np.newaxis]
        min_data = np.min(smoothed_data,axis = 1)[:,np.newaxis]
        maxfr = smoothed_data.max(axis =1)[:,np.newaxis]
        norm_data = smoothed_data/maxfr
    elif norm_method =='none':
        norm_data = smoothed_data
    maxplaceind =[]
    # Align to max position
    if sort_val:
        if len(idorder)==0:    
            maxplaceind = norm_data.argmax(axis = 1)
        else:
            maxplaceind = idorder
        z = np.insert(norm_data,0,maxplaceind,axis = 1)
        sortedZ = z[z[:,0].argsort()]
        sortedZ = np.delete(sortedZ,0,1)
    else:
        sortedZ = norm_data
    font = {'size':fontsize}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=fig_size)
    im = axes.imshow(sortedZ, cmap = cmap)
    plt.xlabel(xy_label[0],fontdict = font)
    plt.ylabel(xy_label[1],fontdict = font)
    plt.xticks(xtick_label[0],xtick_label[1],fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    
    if box_plt:
        ylimit = axes.get_ylim()
        xlimit = axes.get_xlim()
        ymax = np.max(ylimit)
        scale = ymax/250
        axes.set_xlim(0,np.shape(sortedZ)[1])
        axes.set_ylim(0,np.shape(sortedZ)[0])
        if contextname == 'EPM':
            axes.add_patch(plt.Rectangle((armlength-.5,-15*scale),centerlength,15*scale,clip_on=False,
                                           facecolor = 'lightgrey', edgecolor = 'k',
                                           linewidth = 1))
        axes.add_patch(plt.Rectangle((0,-15*scale),armlength-.5,15*scale,clip_on=False,
                                       facecolor = 'lightslategrey', edgecolor = 'k',
                                       linewidth = 4))
        axes.add_patch(plt.Rectangle((armlength+centerlength-.5,-15*scale),armlength,15*scale,clip_on=False,
                                       facecolor = 'lightgrey', edgecolor = 'k',
                                       linewidth = 1))
    cbar = fig.colorbar(im)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    axes.grid(False)
    axes.axis('tight')
    axes.set_xlim(-.5,np.shape(sortedZ)[1]-0.5)
    axes.set_ylim(0,np.shape(sortedZ)[0]-0.5)
    axes.invert_yaxis()
    if remap:
        axes.invert_xaxis()
        axes.text(armlength-10,-3*scale, 'Closed', fontsize = 20)
        axes.text(np.max(xlimit)-1,-3*scale, 'Open', fontsize = 20)
    else:
        axes.text(1,-3*scale, 'Closed', fontsize = 20)
        axes.text(np.max(xlimit)-7,-3*scale, 'Open', fontsize = 20)
    if print_val:
        plt.savefig(figure_title)
        
    return maxplaceind, sortedZ



def plot_combined_data(data, armlength, centerlength, contextname, animalname,
                       plotname, printval, printval_whole, remap):
    import numpy as np
    import matplotlib.pyplot as plt
    combined_data_del = []
    for ijson in range(len(data)):  
        tmpdata = data[ijson]
        if len(tmpdata) > 0:
            data_del = np.delete(tmpdata, 
                                 [np.where((np.sum(tmpdata, 1)/len(tmpdata[0, :]) < 0.01) |
                                           (np.sum(tmpdata, 1)/len(tmpdata[0, :]) > 15))], 0)
            combined_data_del.append(data_del)
            xy_label = ['Spatial bin (2cm)', 'Neuron ID']
            fig_size = [8, 8]
            xtick_label = [[-.5, armlength-.5, 2*armlength-.5, 2*armlength+centerlength-.5,
                            len(data_del[0, :])-armlength-.5, len(data_del[0, :])-.5],
                           ['-%d' % (armlength*2), '-%d' % (armlength), '0', '0', '%d' % (armlength), '%d' % (armlength*2)]]
        
            plot_heatmap(data_del, 'mean', 1, 0, 'Mean Normalized Spatial heatmap',
                         xtick_label, xy_label, fig_size, 'jet', [], 20, 1, contextname,
                         2*armlength, centerlength, remap)
            # plt.title('Mean Normalized Spatial heatmap_%s' % (contextname), fontsize=15)
            plt.yticks(fontsize=20)
            plt.axvline(x=len(data_del[0, :])/2-.5, color='black')
            plt.axvline(x=len(data_del[0, :])/2-centerlength /
                        2-.5, linestyle='--', color='gray')
            plt.axvline(x=len(data_del[0, :])/2+centerlength /
                        2-.5, linestyle='--', color='gray')
            plt.axvline(x=armlength-.5, color='gray')
            plt.axvline(x=len(data_del[0, :])-armlength-.5, color='gray')
            if printval:
                plt.savefig('Mean Normalized Spatial heatmap_%s_%s' % (animalname, plotname))
    combined_data_del = np.vstack(combined_data_del)
    plot_heatmap(combined_data_del, 'mean', 1, 0, 'Mean Normalized Spatial heatmap',
                     xtick_label, xy_label, fig_size, 'jet', [], 20, 1, contextname,
                         2*armlength, centerlength, remap)
    # plt.title('Mean Normalized Spatial heatmap_%s' % (contextname), fontsize=15)
    plt.yticks(fontsize=20)
    plt.axvline(x=len(data_del[0, :])/2-.5, color='black')
    plt.axvline(x=len(data_del[0, :])/2-centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=len(data_del[0, :])/2+centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=armlength-.5, color='gray')
    plt.axvline(x=len(data_del[0, :])-armlength-.5, color='gray')
    if printval_whole:
        plt.savefig('Mean Normalized Spatial heatmap_%s' % (plotname))    
    return combined_data_del


def compare_sort_firingmap(randomfiringmap, testfiringmap, contextname, armlength, 
                           centerlength, comparetitle, draw, remap):
    import numpy as np
    import matplotlib.pyplot as plt
    if len(randomfiringmap) > 1 :
        vstackrandom = np.vstack(randomfiringmap)
        vstacktest = np.vstack(testfiringmap)
        exclude_ind = [np.where((np.sum(vstackrandom, 1)/len(vstackrandom[0, :]) < 0.01) |
                                            (np.sum(vstackrandom, 1)/len(vstackrandom[0, :]) > 15))]
        vstackrandom_del = np.delete(vstackrandom, exclude_ind, 0)
        vstacktest_del = np.delete(vstacktest, exclude_ind, 0)
    else:
        vstackrandom = randomfiringmap[0]
        vstackrandom_del = randomfiringmap[0]
        vstacktest_del = testfiringmap[0]
    xy_label = ['Spatial bin (2cm)', 'Neuron ID']
    fig_size = [8, 8]
    xtick_label = [[-.5, armlength-.5, 2*armlength-.5, 2*armlength+centerlength-.5,
                    len(vstackrandom[0, :])-armlength-.5, len(vstackrandom[0, :])-.5], ['-36', '-18', '0', '0', '18', '36']]
    maxind, sortZ = plot_heatmap(vstackrandom_del, 'mean', 1, draw,
                                 'Mean Normalized Spatial heatmap_%s' % (comparetitle[0]),
                                 xtick_label, xy_label, fig_size, 'jet', [], 20, 1, contextname,
                                 2*armlength, centerlength, 0)
    # plt.title('Mean Normalized Spatial heatmap_%s_EPM' % (comparetitle[0]), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.axvline(x=len(vstackrandom[0, :])/2-.5, color='black')
    plt.axvline(x=len(vstackrandom[0, :])/2-centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=len(vstackrandom[0, :])/2+centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=armlength-.5, color='gray')
    plt.axvline(x=len(vstackrandom[0, :])-armlength-.5, color='gray')
    
    maxind2, sortZ2 = plot_heatmap(vstacktest_del, 'mean', 1, draw, 
                                   'Mean Normalized Spatial heatmap_%s' % (comparetitle[1]),
                                   xtick_label, xy_label, fig_size, 'jet', maxind, 20, 1, contextname,
                                   2*armlength, centerlength, remap)
    # plt.title('Mean Normalized Spatial heatmap_%s_EPM' % (comparetitle[1]), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.axvline(x=len(vstackrandom[0, :])/2-.5, color='black')
    plt.axvline(x=len(vstackrandom[0, :])/2-centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=len(vstackrandom[0, :])/2+centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=armlength-.5, color='gray')
    plt.axvline(x=len(vstackrandom[0, :])-armlength-.5, color='gray')
    return sortZ, sortZ2
    

def compare_sort_firingmap_overlap(randomfiringmap, testfiringmap, contextname, armlength, 
                           centerlength, comparetitle, draw, remap):
    import numpy as np
    import matplotlib.pyplot as plt
    if len(randomfiringmap) > 1 :
        vstackrandom = np.vstack(randomfiringmap)
        vstacktest = np.vstack(testfiringmap)
        exclude_ind = [np.where((np.sum(vstackrandom, 1)/len(vstackrandom[0, :]) < 0.01) |
                                            (np.sum(vstackrandom, 1)/len(vstackrandom[0, :]) > 15))]
        vstackrandom_del = np.delete(vstackrandom, exclude_ind, 0)
        vstacktest_del = np.delete(vstacktest, exclude_ind, 0)
    else:
        vstackrandom = randomfiringmap[0]
        vstackrandom_del = randomfiringmap[0]
        vstacktest_del = testfiringmap[0]
    xy_label = ['Spatial bin (2cm)', 'Neuron ID']
    fig_size = [8, 8]
    xtick_label = [[-.5, armlength-.5, armlength+centerlength-.5,
                    len(vstackrandom[0, :])-.5], ['-18', '0', '0', '18']]
    maxind, sortZ = plot_heatmap_overlap(vstackrandom_del, 'mean', 1, draw,
                                 'Mean Normalized Spatial heatmap_%s' % (comparetitle[0]),
                                 xtick_label, xy_label, fig_size, 'jet', [], 20, 1, contextname,
                                 armlength, centerlength, 0)
    # plt.title('Mean Normalized Spatial heatmap_%s_EPM' % (comparetitle[0]), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.axvline(x=len(vstackrandom[0, :])/2-.5, color='black')
    plt.axvline(x=len(vstackrandom[0, :])/2-centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=len(vstackrandom[0, :])/2+centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=armlength-.5, color='gray')
    plt.axvline(x=len(vstackrandom[0, :])-armlength-.5, color='gray')
    
    maxind2, sortZ2 = plot_heatmap_overlap(vstacktest_del, 'mean', 1, draw, 
                                   'Mean Normalized Spatial heatmap_%s' % (comparetitle[1]),
                                   xtick_label, xy_label, fig_size, 'jet', maxind, 20, 1, contextname,
                                   armlength, centerlength, remap)
    # plt.title('Mean Normalized Spatial heatmap_%s_EPM' % (comparetitle[1]), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.axvline(x=len(vstackrandom[0, :])/2-.5, color='black')
    plt.axvline(x=len(vstackrandom[0, :])/2-centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=len(vstackrandom[0, :])/2+centerlength /
                2-.5, linestyle='--', color='gray')
    plt.axvline(x=armlength-.5, color='gray')
    plt.axvline(x=len(vstackrandom[0, :])-armlength-.5, color='gray')
    return sortZ, sortZ2


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)


def kmean_time_figure(data, clustertime, n_clusters, max_iter, cluster_order,
                      timewindow, timebin_size, fig_name, pre_order):
    from sklearn.cluster import KMeans
    import numpy as np
    import matplotlib.pyplot as plt
    kmeans = KMeans(n_clusters, max_iter=max_iter,
                    random_state=0).fit(data[:, clustertime[0]:clustertime[1]])
    label = kmeans.labels_
    newlabel = []
    for ilabel in range(n_clusters):
        newlabel.append(np.where(label == ilabel))
    for ilabel in range(len(newlabel)):
        label[newlabel[ilabel]] = cluster_order[ilabel]
    if len(pre_order) > 1:
        maxind = pre_order
    else:
        maxind = label.argmax(axis = 1)
    z = np.insert(data, 0, maxind, axis=1)
    sortedZ = z[z[:, 0].argsort()]
    sortedZ = np.delete(sortedZ, 0, 1)
    sorted_data = sortedZ
    time_zero = np.abs(timewindow[0])/timebin_size
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(sorted_data, interpolation='nearest', aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.clim(-2.5, 2.5)
    plt.plot([time_zero, time_zero], ax.get_ylim(), linestyle="--",
             linewidth=2, color='black')
    plt.xticks(ticks=[0, time_zero, ax.get_xlim()[1]],
               labels=['-%d s' % (timewindow[0]), '0s', '%d s' % (timewindow[1])])
    plt.ylabel('Cell ID')
    plt.title('Cell response in %s' % (fig_name))
    plt.rc('font', size = 15)
    plt.savefig('Cell response in %s' % (fig_name))
    return pre_order

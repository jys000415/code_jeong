# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:58:06 2022

@author: yeong
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind

# Load dataset
path_epm = 'J:/Jeong Yeongseok/Project_spatial_coding/Behavior/Sub-pIC/Cohort2/EPM'
os.chdir(path_epm)
epm_pre = pd.read_csv("EPM_pre_2.csv")
epm_pre_2min = pd.read_csv("EPM_pre_2_2min.csv")
epm_pre_5min = pd.read_csv("EPM_pre_2_5min.csv")
epm_post = pd.read_csv("EPM_post_2.csv")
epm_post_2min = pd.read_csv("EPM_post_2_2min.csv")
epm_post_5min = pd.read_csv("EPM_post_2_5min.csv")

path_ezm = 'J:/Jeong Yeongseok/Project_spatial_coding/Behavior/Sub-pIC/Cohort2/EZM'
os.chdir(path_ezm)
ezm = pd.read_csv("EZM_2.csv")
ezm_2min = pd.read_csv("EZM_2_segment_2min.csv")
ezm_5min = pd.read_csv("EZM_2_segment_5min.csv")

path_of = 'J:/Jeong Yeongseok/Project_spatial_coding/Behavior/Sub-pIC/Cohort2/Openfield'
os.chdir(path_of)
openfield = pd.read_csv("Openfield_2.csv")
openfield_2min = pd.read_csv("Openfield_2_2min.csv")
openfield_5min = pd.read_csv("Openfield_2_5min.csv")

path_shelter = 'J:/Jeong Yeongseok/Project_spatial_coding/Behavior/Sub-pIC/Cohort2/Openfield_shelter'
os.chdir(path_shelter)
shelter = pd.read_csv("Openfield_shelter_2.csv")
shelter_2min = pd.read_csv("Openfield_shelter_2_seg_2min.csv")
shelter_5min = pd.read_csv("Openfield_shelter_2_seg_5min.csv")

path_epm2 = 'J:/Jeong Yeongseok/Project_spatial_coding/Behavior/Sub-pIC/Cohort2/EPM2'
os.chdir(path_epm2)
epm2 = pd.read_csv("EPM2_2.csv")
epm2_2min = pd.read_csv("EPM2_2_seg_2min.csv")
epm2_5min = pd.read_csv("EPM2_2_seg_5min.csv")

#%%
# Plot dataframe
colors = {'Control':'gray','ChR2':'steelblue','eOPN3':'darkorange'}
user_order = ['Control','ChR2','eOPN3']
sns.set(rc = {'figure.figsize' : (6, 6),               
               'axes.labelsize' : 16,
               'axes.titlesize' : 18,
               'xtick.labelsize' : 15,
               'ytick.labelsize' : 15})
sns.set_style('white')

#%%

# EPM_pre Closed arm time
def dataframe_ttest(data, param_name):    
    cat1 = data[data['Treatment']=='Control'][param_name]
    cat2 = data[data['Treatment']=='eOPN3'][param_name]
    cat3 = data[data['Treatment']=='ChR2'][param_name]
    df_pval = [ttest_ind(cat1,cat3)[1], ttest_ind(cat2,cat3)[1],
               ttest_ind(cat1,cat2)[1]]
    return df_pval


def ttest_plot(data, df_pval, yvalue, ylabel, title_name, fig_name):
    plt.figure()
    if yvalue == 'Percent':
        ydata = data[param_name]/600*100

    elif yvalue == 'Distance':
        ydata = data[param_name]
    elif yvalue == 'Speed':
        ydata = data[param_name]*100

    ax = sns.boxplot(x = data['Treatment'], y = ydata,
                 boxprops = {'alpha':0.4}, order = user_order,palette = colors,
                 showfliers=False)
    sns.stripplot(x = data['Treatment'], y = ydata, 
                  size = 7, order = user_order, ax = ax, palette = colors)
    ymax = np.max(ydata)
    if yvalue == 'Percent':
        ymax = 100
        
    for ipval in range(len(df_pval)):
        if ipval == 2:
            x0, x1, ymax_t, t_loc = 0,2, ymax*1.12, 1
        else:
            x0, x1, ymax_t, t_loc = ipval, ipval+1, ymax*(1.02+ipval*0.05), ipval+0.5
        if df_pval[ipval] < 0.05:
            ax.plot([x0, x1], [ymax_t, ymax_t],
                    lw = 1.5, c = 'k')
            plt.text(t_loc, ymax_t*1.01, '*',
                     horizontalalignment = 'center', fontsize = 15)
        elif (df_pval[ipval] > 0.05) & (df_pval[ipval] < 0.2):
            ax.plot([x0, x1], [ymax_t, ymax_t], 
                    lw = 1.5, c = 'k')
            plt.text(t_loc, ymax_t*1.01, 'p_value = %.2f' %(df_pval[ipval]),
                     horizontalalignment = 'center')
    ax.set(xlabel = '', ylabel = ylabel, ylim = (0, ymax*1.15), 
       title = title_name)
    sns.despine()
    plt.savefig(fig_name)
    
    
data = ezm
yvalue, ylabel = 'Percent', 'Time (%)'
os.chdir(path_ezm)
param_name = 'Closed arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EZM Closed arm','EZM_Closed arm')

param_name = 'Open arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EZM Open arm','EZM_Open arm')

# EPM_pre 
data = epm_pre
os.chdir(path_epm)
param_name = 'Closed arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Closed arm','EPM_pre_Closed arm')

param_name = 'Open arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Open arm','EPM_pre_Open arm')

param_name = 'Center : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Center','EPM_pre_Center')


# EPM_post
data = epm_post
os.chdir(path_epm)
param_name = 'Closed arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Closed arm','EPM_post_Closed arm')

param_name = 'Open arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Open arm','EPM_post_Open arm')

param_name = 'Center : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Center','EPM_post_Center')


# EPM_2nd
data = epm2
os.chdir(path_epm2)
param_name = 'Closed arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Closed arm','EPM_2nd_Closed arm')

param_name = 'Open arm : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Open arm','EPM_2nd_Open arm')

param_name = 'Center : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'EPM Center','EPM_2nd_Center')


# Open field
data = openfield
os.chdir(path_of)
param_name = 'Corner : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Corner','Openfield_corner')

param_name = 'Center : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Center','Openfield_center')

param_name = 'Border : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Border','Openfield_border')


# Shelter
data = shelter
os.chdir(path_shelter)
param_name = 'Open zone : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Open','Shelter_open')

param_name = 'Shelter zone : time'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Shelter','Shelter_shelter')

#%% Segment analysis
def ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, title_name,
                   fig_name):
    plt.figure()
    segment_value = data['Segment of test'].unique()
    if yvalue == 'Percent':
        ydata = data[param_name]/(600/len(segment_value))*100
    elif yvalue == 'Distance':
        ydata = data[param_name]
    elif yvalue == 'Speed':
        ydata = data[param_name]*100
    ax = sns.lineplot(x = data['Segment of test'], y = ydata,
                 hue = data['Treatment'],palette = colors,
                 err_style ='bars', errorbar = ("se",1))
    ymax = np.max(ydata)
    if yvalue == 'Percent':
        ymax = 100
    color = ['steelblue','red','darkorange']
    for iseg in range(len(segment_value)):
        df_pval = dataframe_ttest(
            data[data['Segment of test']==segment_value[iseg]],
            param_name)
        for ipval in range(len(df_pval)):            
            if df_pval[ipval] < 0.05:
                plt.text(iseg, ymax*(1+0.02*ipval), '*', color = color[ipval],
                         horizontalalignment = 'center', fontsize = 20)
    ax.set(xlabel = '', ylabel = ylabel, ylim = (0, ymax*1.1), 
    title = title_name)
    ax.set_xticklabels(xlabel)
    sns.despine()
    plt.legend(bbox_to_anchor=(1.02,1), loc = 'upper left', borderaxespad = 0)
    plt.tight_layout()
    plt.savefig(fig_name)


#EPM time segment
os.chdir(path_epm)
xlabel = ['2min','4min','6min','8min','10min']
yvalue, ylabel = 'Percent','Time (%)'
data = epm_pre_2min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_pre_2min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_pre_2min_open arm')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_pre_2min_center arm')

xlabel = ['5min','10min']
data = epm_pre_5min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_pre_5min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_pre_5min_open arm')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_pre_5min_center arm')

xlabel = ['2min','4min','6min','8min','10min']
data = epm_post_2min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_post_2min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_post_2min_open arm')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_post_2min_center arm')

xlabel = ['5min','10min']
data = epm_post_5min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_post_5min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_post_5min_open arm')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_post_5min_center arm')

#EZM time segment
os.chdir(path_ezm)
xlabel = ['2min','4min','6min','8min','10min']
data = ezm_2min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EZM_2min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EZM_2min_open arm')

xlabel = ['5min','10min']
data = ezm_5min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EZM_5min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EZM_5min_open arm')

os.chdir(path_of)
xlabel = ['2min','4min','6min','8min','10min']
data = openfield_2min
param_name = 'Corner : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Corner', 
               'Openfield_2min_corner')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center',
               'Openfield_2min_center')

param_name = 'Border : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Border',
               'Openfield_2min_border')

xlabel = ['5min','10min']
data = openfield_5min
param_name = 'Corner : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Corner', 
               'Openfield_5min_corner')

param_name = 'Border : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Border',
               'Openfield_5min_border')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center',
               'Openfield_5min_center')

# Shelter
os.chdir(path_shelter)
xlabel = ['2min','4min','6min','8min','10min']
data = shelter_2min
param_name = 'Shelter zone : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Shelter', 
               'Shelter_2min_shelter')

param_name = 'Open zone : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open',
               'Shelter_2min_open')

param_name = 'Shelter zone : time'
xlabel = ['5min','10min']
data = shelter_5min
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Shelter', 
               'Shelter_5min_shelter')

param_name = 'Open zone : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open',
               'Shelter_5min_open')

# EPM_2nd
os.chdir(path_epm2)
xlabel = ['2min','4min','6min','8min','10min']
data = epm2_2min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_2nd_2min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_2nd_2min_open arm')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_2nd_2min_center arm')

xlabel = ['5min','10min']
data = epm2_5min
param_name = 'Closed arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_2nd_5min_closed arm')

param_name = 'Open arm : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_2nd_5min_open arm')

param_name = 'Center : time'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_2nd_5min_center arm')

#%% Distance, Speed analysis
data = ezm
yvalue, ylabel = 'Distance', 'Distance (m)'
os.chdir(path_ezm)
param_name = 'Closed arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Closed','EZM_Closed arm_distance')

param_name = 'Open arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Open','EZM_Open arm_distance')

param_name = 'Distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance','EZM_distance')

yvalue, ylabel = 'Speed', 'Speed (cm/s)'
param_name = 'Closed arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Closed','EZM_Closed arm_speed')

param_name = 'Open arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Open','EZM_Open arm_speed')

param_name = 'Mean speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed','EZM_speed')

# EPM_pre
data = epm_pre
yvalue, ylabel = 'Distance', 'Distance (m)'
os.chdir(path_epm)
param_name = 'Closed arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Closed','EPM_pre_Closed arm_distance')

param_name = 'Open arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Open','EPM_pre_Open arm_distance')

param_name = 'Center : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Center','EPM_pre_Center_distance')

param_name = 'Distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance','EPM_pre_distance')

yvalue, ylabel = 'Speed', 'Speed (cm/s)'
param_name = 'Closed arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Closed','EPM_pre_Closed arm_speed')

param_name = 'Open arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Open','EPM_pre_Open arm_speed')

param_name = 'Center : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Center','EPM_pre_Center_speed')

param_name = 'Mean speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed','EPM_pre_speed')

# EPM_post
data = epm_post
yvalue, ylabel = 'Distance', 'Distance (m)'
os.chdir(path_epm)
param_name = 'Closed arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Closed','EPM_post_Closed arm_distance')

param_name = 'Open arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Open','EPM_post_Open arm_distance')

param_name = 'Center : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Center','EPM_post_Center_distance')

param_name = 'Distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance','EPM_post_distance')

yvalue, ylabel = 'Speed', 'Speed (cm/s)'
param_name = 'Closed arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Closed','EPM_post_Closed arm_speed')

param_name = 'Open arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Open','EPM_post_Open arm_speed')

param_name = 'Center : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Center','EPM_post_Center_speed')

param_name = 'Mean speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed','EPM_post_speed')

# EPM_2nd
data = epm2
yvalue, ylabel = 'Distance', 'Distance (m)'
os.chdir(path_epm2)
param_name = 'Closed arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Closed','EPM_2nd_Closed arm_distance')

param_name = 'Open arm : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Open','EPM_2nd_Open arm_distance')

param_name = 'Center : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Center','EPM_2nd_Center_distance')

param_name = 'Distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance','EPM_2nd_distance')

yvalue, ylabel = 'Speed', 'Speed (cm/s)'
param_name = 'Closed arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Closed','EPM_2nd_Closed arm_speed')

param_name = 'Open arm : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Open','EPM_2nd_Open arm_speed')

param_name = 'Center : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Center','EPM_2nd_Center_speed')

param_name = 'Mean speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed','EPM_2nd_speed')

# Openfield
data = openfield
yvalue, ylabel = 'Distance', 'Distance (m)'
os.chdir(path_of)
param_name = 'Corner : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Corner','Openfield_Corner_distance')

param_name = 'Center : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Center','Openfield_Center_distance')

param_name = 'Border : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Border','Openfield_Border_distance')

param_name = 'Distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance','Openfield_distance')

param_name = 'Corner : average speed'  
yvalue, ylabel = 'Speed', 'Speed (cm/s)'
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Corner','Openfield_Corner_speed')

param_name = 'Center : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Center','Openfield_Center_speed')

param_name = 'Border : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Border','Openfield_Border_speed')

yvalue, ylabel = 'Speed', 'Speed (cm/s)'
param_name = 'Mean speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed','Openfield_speed')

# Openfield Shelter
data = shelter
yvalue, ylabel = 'Distance', 'Distance (m)'

os.chdir(path_shelter)
param_name = 'Open zone : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Open','Shelter_Open_distance')

param_name = 'Shelter zone : distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance in Shelter','Shelter_Shelter_distance')

param_name = 'Distance'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Distance','Shelter_distance')

yvalue, ylabel = 'Speed', 'Speed (cm/s)'
param_name = 'Open zone : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Open','Shelter_Open_speed')

param_name = 'Shelter zone : average speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed in Shelter','Shelter_Shelter_speed')

param_name = 'Mean speed'  
df_pval = dataframe_ttest(data, param_name)
ttest_plot(data, df_pval, yvalue, ylabel,'Speed','Shelter_speed')


#%% EPM time segment
os.chdir(path_epm)
xlabel = ['2min','4min','6min','8min','10min']
yvalue, ylabel = 'Distance','Distance (m)'
data = epm_pre_2min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_pre_2min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_pre_2min_open arm_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_pre_2min_center arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EPM_pre_2min_Distance')

xlabel = ['5min','10min']
data = epm_pre_5min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_pre_5min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_pre_5min_open arm_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_pre_5min_center arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EPM_pre_5min_Distance')

xlabel = ['2min','4min','6min','8min','10min']
data = epm_post_2min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_post_2min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_post_2min_open arm_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_post_2min_center arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EPM_post_2min_Distance')

xlabel = ['5min','10min']
data = epm_post_5min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_post_5min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_post_5min_open arm_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_post_5min_center arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EPM_post_5min_Distance')

#EZM time segment
os.chdir(path_ezm)
xlabel = ['2min','4min','6min','8min','10min']
data = ezm_2min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EZM_2min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EZM_2min_open arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EZM_post_2min_Distance')

xlabel = ['5min','10min']
data = ezm_5min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EZM_5min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EZM_5min_open arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EZM_post_5min_Distance')

os.chdir(path_of)
xlabel = ['2min','4min','6min','8min','10min']
data = openfield_2min
param_name = 'Corner : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Corner', 
               'Openfield_2min_corner_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center',
               'Openfield_2min_center_Distance')

param_name = 'Border : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Border',
               'Openfield_2min_border_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'Openfield_2min_Distance')

xlabel = ['5min','10min']
data = openfield_5min
param_name = 'Corner : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Corner', 
               'Openfield_5min_corner_Distance')

param_name = 'Border : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Border',
               'Openfield_5min_border_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center',
               'Openfield_5min_center_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'Openfield_5min_Distance')

# Shelter
os.chdir(path_shelter)
xlabel = ['2min','4min','6min','8min','10min']
data = shelter_2min
param_name = 'Shelter zone : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Shelter', 
               'Shelter_2min_shelter_Distance')

param_name = 'Open zone : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open',
               'Shelter_2min_open_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'Shelter_2min_Distance')

param_name = 'Shelter zone : distance'
xlabel = ['5min','10min']
data = shelter_5min
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Shelter', 
               'Shelter_5min_shelter_Distance')

param_name = 'Open zone : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open',
               'Shelter_5min_open_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'Shelter_5min_Distance')

# EPM_2nd
os.chdir(path_epm2)
xlabel = ['2min','4min','6min','8min','10min']
data = epm2_2min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_2nd_2min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_2nd_2min_open arm_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_2nd_2min_center arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EPM_2nd_2min_Distance')

xlabel = ['5min','10min']
data = epm2_5min
param_name = 'Closed arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_2nd_5min_closed arm_Distance')

param_name = 'Open arm : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_2nd_5min_open arm_Distance')

param_name = 'Center : distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_2nd_5min_center arm_Distance')

param_name = 'Distance'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Distance', 
               'EPM_2nd_5min_Distance')

#%% Speed analysis time segment
os.chdir(path_epm)
xlabel = ['2min','4min','6min','8min','10min']
yvalue, ylabel = 'Speed','Speed (cm/s)'
data = epm_pre_2min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_pre_2min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_pre_2min_open arm_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_pre_2min_center arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EPM_pre_2min_speed')

xlabel = ['5min','10min']
data = epm_pre_5min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_pre_5min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_pre_5min_open arm_Distance')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_pre_5min_center arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EPM_pre_5min_speed')

xlabel = ['2min','4min','6min','8min','10min']
data = epm_post_2min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_post_2min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_post_2min_open arm_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_post_2min_center arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EPM_post_2min_speed')

xlabel = ['5min','10min']
data = epm_post_5min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_post_5min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_post_5min_open arm_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_post_5min_center arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EPM_post_5min_speed')

#EZM time segment
os.chdir(path_ezm)
xlabel = ['2min','4min','6min','8min','10min']
data = ezm_2min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EZM_2min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EZM_2min_open arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EZM_post_2min_speed')

xlabel = ['5min','10min']
data = ezm_5min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EZM_5min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EZM_5min_open arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EZM_post_5min_speed')

os.chdir(path_of)
xlabel = ['2min','4min','6min','8min','10min']
data = openfield_2min
param_name = 'Corner : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Corner', 
               'Openfield_2min_corner_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center',
               'Openfield_2min_center_speed')

param_name = 'Border : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Border',
               'Openfield_2min_border_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'Openfield_2min_speed')

xlabel = ['5min','10min']
data = openfield_5min
param_name = 'Corner : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Corner', 
               'Openfield_5min_corner_speed')

param_name = 'Border : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Border',
               'Openfield_5min_border_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center',
               'Openfield_5min_center_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'Openfield_5min_speed')

# Shelter
os.chdir(path_shelter)
xlabel = ['2min','4min','6min','8min','10min']
data = shelter_2min
param_name = 'Shelter zone : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Shelter', 
               'Shelter_2min_shelter_speed')

param_name = 'Open zone : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open',
               'Shelter_2min_open_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'Shelter_2min_speed')

param_name = 'Shelter zone : average speed'
xlabel = ['5min','10min']
data = shelter_5min
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Shelter', 
               'Shelter_5min_shelter_speed')

param_name = 'Open zone : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open',
               'Shelter_5min_open_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'Shelter_5min_speed')

# EPM_2nd
os.chdir(path_epm2)
xlabel = ['2min','4min','6min','8min','10min']
data = epm2_2min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_2nd_2min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_2nd_2min_open arm_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_2nd_2min_center arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EPM_2nd_2min_speed')

xlabel = ['5min','10min']
data = epm2_5min
param_name = 'Closed arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Closed arm', 
               'EPM_2nd_5min_closed arm_speed')

param_name = 'Open arm : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Open arm',
               'EPM_2nd_5min_open arm_speed')

param_name = 'Center : average speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Center', 
               'EPM_2nd_5min_center arm_speed')

param_name = 'Mean speed'
ttest_lineplot(data, param_name, yvalue, xlabel, ylabel, 'Speed', 
               'EPM_2nd_5min_speed')

#%% Latency
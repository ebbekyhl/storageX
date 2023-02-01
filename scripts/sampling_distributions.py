# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:30:36 2022

@author: au485969
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from statistics import mode
from collections import Counter
plt.close('all')

# Plotting layout
fs = 17
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

# fig6, ax6 = plt.subplots(figsize=[10,6])
fig7, ax7 = plt.subplots(figsize=[10,6])
fig8, ax8 = plt.subplots(figsize=[10,6])
fig9, ax9 = plt.subplots(figsize=[10,6])
fig10, ax10 = plt.subplots(figsize=[10,6])
fig11, ax11 = plt.subplots(figsize=[10,6])
fig12, ax12 = plt.subplots(figsize=[10,6])
#%%
perfection = {'eta1 [-]':0.95,
              'eta2 [-]':0.95,
              'c1':35,
              'c2':35,
              'c_hat [EUR/kWh]':1}

N_perfections = 6
#%%
sspace = pd.read_csv('../results/sspace_w_sectorcoupling_merged.csv',index_col=0)
sector = sspace.loc['sector'].fillna(0)
    
sspace_i = pd.read_csv('../results/sspace_w_sectorcoupling.csv',index_col=0)
sector_i = sspace_i.loc['sector'].fillna(0)

color_dic = {'0':'green',
            'T-H':'blue',
            'T-H-I-B':'orange'}

threshold = 2

# yax = 'percentage'
yax = 'number'

count = 0
for sec in [0,'T-H','T-H-I-B']:
    sspace_sector_initial = sspace[sector[sector == sec].index].drop(index='sector').astype(float)
    sspace_sector_initial = sspace_sector_initial[sspace_sector_initial.loc['eta1 [-]'][sspace_sector_initial.loc['eta1 [-]'] <= 1].index]
    
    sspace_sector_initial_i = sspace_i[sector_i[sector_i == sec].index].drop(index='sector').astype(float)
    sspace_sector_initial_i = sspace_sector_initial_i[sspace_sector_initial_i.loc['eta1 [-]'][sspace_sector_initial_i.loc['eta1 [-]'] <= 1].index]
    
    # sspace_tau_10 = sspace_sector_initial[sspace_sector_initial.loc['tau [n_days]'][sspace_sector_initial.loc['tau [n_days]'] == 10].index]
    # sspace_tau_30 = sspace_sector_initial[sspace_sector_initial.loc['tau [n_days]'][sspace_sector_initial.loc['tau [n_days]'] == 30].index]
    # df_10 = sspace_tau_10.loc[['c1','c2','eta1 [-]','eta2 [-]','c_hat [EUR/kWh]']].T
    # df_30 = sspace_tau_30.loc[['c1','c2','eta1 [-]','eta2 [-]','c_hat [EUR/kWh]']].T
    # df_diff = pd.concat([df_10,df_30]).drop_duplicates(keep=False)
    # sspace_sector_initial.drop(columns = df_diff.index,inplace=True)
    
    data_initial_7 = np.array(sspace_sector_initial.loc['c1'])
    data_initial_7.sort()
    data_initial_8 = np.array(sspace_sector_initial.loc['c2'])
    data_initial_8.sort()
    data_initial_9 = np.array(sspace_sector_initial.loc['eta1 [-]'])
    data_initial_9.sort()
    data_initial_10 = np.array(sspace_sector_initial.loc['eta2 [-]'])
    data_initial_10.sort()
    data_initial_11 = np.array(sspace_sector_initial.loc['c_hat [EUR/kWh]'])
    data_initial_11.sort()
    data_initial_12 = np.array(sspace_sector_initial.loc['tau [n_days]'])
    data_initial_12.sort()
    
    data_initial_7_i = np.array(sspace_sector_initial_i.loc['c1'])
    data_initial_7_i.sort()
    data_initial_8_i = np.array(sspace_sector_initial_i.loc['c2'])
    data_initial_8_i.sort()
    data_initial_9_i = np.array(sspace_sector_initial_i.loc['eta1 [-]'])
    data_initial_9_i.sort()
    data_initial_10_i = np.array(sspace_sector_initial_i.loc['eta2 [-]'])
    data_initial_10_i.sort()
    data_initial_11_i = np.array(sspace_sector_initial_i.loc['c_hat [EUR/kWh]'])
    data_initial_11_i.sort()
    data_initial_12_i = np.array(sspace_sector_initial_i.loc['tau [n_days]'])
    data_initial_12_i.sort()
    
    # print(len(sspace_sector_initial.columns))
    sspace_sector = sspace_sector_initial.copy()
    for i in sspace_sector.columns:
        conf_i = sspace_sector[i]
        perf_i = []
        for j in perfection.keys():
            conf_ij = conf_i.loc[j]
            if conf_ij == perfection[j]:
                perf_i.append(conf_ij)
            
        if len(perf_i) > N_perfections:
            print('dropping',perf_i)
            sspace_sector = sspace_sector.drop(columns=i)
    
    E = sspace_sector.loc['E [GWh]']*sspace_sector.loc['eta2 [-]']/1000
    
    
    E = E[E >= threshold]
    
    
    sspace_sector = sspace_sector[E.index]
    
    # ax6.plot(sspace_sector.loc['c_hat [EUR/kWh]'],sspace_sector.loc['E [GWh]'],'.',color=color_dic[str(sec)])

    sspace_40 = sspace_sector.loc["c_hat [EUR/kWh]"][sspace_sector.loc["c_hat [EUR/kWh]"] == 40.0].index
        
    wi = 0.15
    data_7 = np.array(sspace_sector.loc['c1'])
    data_7.sort()
    dif = list(set(list(Counter(data_initial_7).keys())) - set(list(Counter(data_7).keys())))
    keys = list(Counter(data_7).keys()) + list(np.sort(dif))
    values = list(Counter(data_7).values()) + [0]*len(dif)
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('keys',inplace=True)
    keys = df['keys'].values
    values = df['values'].values
    
    if yax == 'number':
        ax7.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_7_i).values())),color='none',hatch='//',edgecolor='grey',lw=0,zorder=3,width=3*wi)
        ax7.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_7).values())),color='darkgrey',zorder=2,width=3*wi)
        ax7.bar(np.arange(len(keys)),[2016/len(keys)],color='lightgrey',zorder=1,width=3*wi)
        denominator = 100 # cancels division
    else:
        # denominator = np.array(list(Counter(data_initial_7).values())) # Number of configurations with c1 = 35, c1 = 350, ...
        denominator = 2016/len(keys)
        
    ax7.bar(np.arange(len(keys))-wi+count*wi,
            np.array(values)/denominator*100,
            width=wi,color=color_dic[str(sec)],zorder=4)
    
    data_8 = np.array(sspace_sector.loc['c2'])
    data_8.sort()
    dif = list(set(list(Counter(data_initial_8).keys())) - set(list(Counter(data_8).keys())))
    keys = list(Counter(data_8).keys()) + list(np.sort(dif))
    values = list(Counter(data_8).values()) + [0]*len(dif)
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('keys',inplace=True)
    keys = df['keys'].values
    values = df['values'].values
    
    if yax == 'number':
        ax8.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_8_i).values())),color='none',hatch='//',edgecolor='grey',lw=0,zorder=3,width=3*wi)
        ax8.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_8).values())),color='darkgrey',zorder=2,width=3*wi)
        ax8.bar(np.arange(len(keys)),[2016/len(keys)],color='lightgrey',zorder=1,width=3*wi)
        denominator = 100 # cancels division
    else:
        denominator = 2016/len(keys) #np.array(list(Counter(data_initial_8).values())) # Number of configurations with c1 = 35, c1 = 350, ...
    ax8.bar(np.arange(len(keys))-wi+count*wi,
            np.array(values)/denominator*100,
            width=wi,color=color_dic[str(sec)],zorder=4)
    
    wi = 0.1
    data_9 = np.array(sspace_sector.loc['eta1 [-]'])
    data_9.sort()
    dif = list(set(list(Counter(data_initial_9).keys())) - set(list(Counter(data_9).keys())))
    keys = list(Counter(data_9).keys()) + list(np.sort(dif))
    values = list(Counter(data_9).values()) + [0]*len(dif)
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('keys',inplace=True)
    keys = df['keys'].values
    values = df['values'].values
    
    if yax == 'number':
        ax9.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_9_i).values())),color='none',hatch='//',edgecolor='grey',lw=0,zorder=3,width=3*wi)
        ax9.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_9).values())),color='darkgrey',zorder=2,width=3*wi)
        ax9.bar(np.arange(len(keys)),[2016/len(keys)],color='lightgrey',zorder=1,width=3*wi)
        denominator = 100 # cancels division
    else:
        denominator = 2016/len(keys) #np.array(list(Counter(data_initial_9).values())) # Number of configurations with c1 = 35, c1 = 350, ...
    ax9.bar(np.arange(len(keys))-wi+count*wi,
            np.array(values)/denominator*100,
            width=wi,color=color_dic[str(sec)],zorder=4)
    
    data_10 = np.array(sspace_sector.loc['eta2 [-]'])
    data_10.sort()
    dif = list(set(list(Counter(data_initial_10).keys())) - set(list(Counter(data_10).keys())))
    keys = list(Counter(data_10).keys()) + list(np.sort(dif))
    values = list(Counter(data_10).values()) + [0]*len(dif)
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('keys',inplace=True)
    keys = df['keys'].values
    values = df['values'].values
    
    if yax == 'number':
        ax10.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_10_i).values())),color='none',hatch='//',edgecolor='grey',lw=0,zorder=3,width=3*wi)
        ax10.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_10).values())),color='darkgrey',zorder=2,width=3*wi)
        ax10.bar(np.arange(len(keys)),[2016/len(keys)],color='lightgrey',zorder=1,width=3*wi)
        denominator = 100 # cancels division
    else:
        denominator = 2016/len(keys) #np.array(list(Counter(data_initial_10).values())) # Number of configurations with c1 = 35, c1 = 350, ...
    ax10.bar(np.arange(len(keys))-wi+count*wi,
            np.array(values)/denominator*100,
            width=wi,color=color_dic[str(sec)],zorder=4)
    
    wi = 0.2
    data_11 = np.array(sspace_sector.loc['c_hat [EUR/kWh]'])
    data_11.sort()
    dif = list(set(list(Counter(data_initial_11).keys())) - set(list(Counter(data_11).keys())))
    keys = list(Counter(data_11).keys()) + list(np.sort(dif))
    values = list(Counter(data_11).values()) + [0]*len(dif)
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('keys',inplace=True)
    keys = df['keys'].values
    values = df['values'].values
    
    if yax == 'number':
        ax11.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_11_i).values())),color='none',hatch='//',edgecolor='grey',lw=0,zorder=3,width=3*wi)
        ax11.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_11).values())),color='darkgrey',zorder=2,width=3*wi)
        ax11.bar(np.arange(len(keys)),[2016/len(keys)],color='lightgrey',zorder=1,width=3*wi)
        denominator = 100 # cancels division
    else:
        denominator = 2016/len(keys) # np.array(list(Counter(data_initial_11).values())) # Number of configurations with c1 = 35, c1 = 350, ...
    ax11.bar(np.arange(len(keys))-wi+count*wi,
            np.array(values)/denominator*100,
            width=wi,color=color_dic[str(sec)],zorder=4)
    
    data_12 = np.array(sspace_sector.loc['tau [n_days]'])
    data_12.sort()
    dif = list(set(list(Counter(data_initial_12).keys())) - set(list(Counter(data_12).keys())))
    keys = list(Counter(data_12).keys()) + list(np.sort(dif))
    values = list(Counter(data_12).values()) + [0]*len(dif)
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('keys',inplace=True)
    keys = df['keys'].values
    values = df['values'].values
    
    if yax == 'number':
        ax12.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_12_i).values())),color='none',hatch='//',edgecolor='grey',lw=0,zorder=3,width=3*wi)
        ax12.bar(np.arange(len(keys)),np.array(list(Counter(data_initial_12).values())),color='darkgrey',zorder=2,width=3*wi)
        ax12.bar(np.arange(len(keys)),[2016/len(keys)],color='lightgrey',zorder=1,width=3*wi)
        denominator = 100 # cancels division
    else:
        denominator = 2016/len(keys) # np.array(list(Counter(data_initial_12).values())) # Number of configurations with c1 = 35, c1 = 350, ...'

    ax12.bar(np.arange(len(keys))-wi+count*wi,
            np.array(values)/denominator*100,
            width=wi,color=color_dic[str(sec)],zorder=4)
    
    count += 1

ax7.set_xticks([0,1,2,3])
ax7.set_xticklabels([35,350,490,700])
# ax7.set_ylabel('Number of configurations')
ax7.set_xlabel('Charge capacity cost ' + r'$c_c$' + ' [EUR/kW]',fontsize=fs)
# ax7.set_ylim([0,ymax])
ax7.grid()

ax8.set_xticks([0,1,2,3])
ax8.set_xticklabels([35,350,490,700])
# ax8.set_ylabel('Number of configurations')
ax8.set_xlabel('Discharge capacity cost ' + r'$c_d$' + ' [EUR/kW]',fontsize=fs)
# ax8.set_ylim([0,ymax])
ax8.grid()

ax9.set_xticks([0,1,2])
ax9.set_xticklabels([25,50,95])
# ax9.set_ylabel('Number of configurations')
ax9.set_xlabel('Charge efficiency ' + r'$\eta_c$' + ' [%]',fontsize=fs)
# ax9.set_ylim([0,ymax])
ax9.grid()

ax10.set_xticks([0,1,2])
ax10.set_xticklabels([25,50,95])
# ax10.set_xlim([-0.2,2.2])
# ax10.set_ylabel('Number of configurations')
ax10.set_xlabel('Discharge efficiency ' + r'$\eta_d$' + ' [%]',fontsize=fs)
# ax10.set_ylim([0,ymax])
ax10.grid()

ax11.set_xticks([0,1,2,3,4,5,6])
ax11.set_xticklabels([1,2,5,10,20,30,40])
# ax11.set_ylabel('Number of configurations')
ax11.set_xlabel('Energy capacity cost ' + r'$\hat{c}$' + ' [EUR/kWh]',fontsize=fs)
# ax11.set_ylim([0,ymax])
ax11.grid()

ax12.set_xticks([0,1])
ax12.set_xticklabels([10,30])
# ax12.set_ylabel('Number of configurations')
ax12.set_xlabel('Self-discharge time ' + r'$\tau_{SD}$' + ' [days]',fontsize=fs)
ax12.grid()


for ax in [ax7,ax8,ax9,ax10,ax11,ax12]:
    if yax == 'percentage':
        ymax = 40
        ax.set_ylim([0,ymax])
        ax.set_ylabel('Frequency [%]')
    else:
        ax.set_ylabel('Number of configurations')

ii = 0
params = ['c_c','c_d','eta_c','eta_d','chat','tau']
for fig in [fig7,fig8,fig9,fig10,fig11,fig12]:
    fig.savefig('../figures/Sampling_space_' + params[ii] + '_' + str(threshold) +'.png',dpi=300,bbox_inches="tight")
    ii += 1
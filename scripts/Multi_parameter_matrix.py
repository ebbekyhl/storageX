# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:41:10 2021
@author: au485969
"""

import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter
plt.close('all')

# Plotting layout
fs = 25
ls = 20
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

#%% Setup
visual = '_threshold' # ''
colorcode = 'round-trip'
path = '../results/'
filename = path + 'sspace.csv'
df1 = pd.read_csv(path + 'sspace.csv',index_col=0).fillna(0)
df1_T = df1.T
df1_T = df1_T.query('sector == @sector').drop(columns='sector').astype(float)
df1_T = df1_T.sort_values(by='c_hat [EUR/kWh]')
df1_T['E [GWh]'] = df1_T['E [GWh]']*df1_T['eta2 [-]']*1e-3 # convert GWh to TWh
df1_T.rename(columns={'E [GWh]':'E [TWh]'}, inplace=True)
df1 = df1_T.reset_index(drop=True).T

def get_unique_numbers(numbers):
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers
#%% 
threshold_E = 2 # Storage energy capacity lower threshold (TWh)
threshold_L = 0 # Load coverage lower threshold (%)

df = df1.copy()
co2s = df.loc['co2_cap [%]'].unique()
taus = df.loc['tau [n_days]'].unique()
color_list = ['royalblue','darkred','orange','yellow','lawngreen','deeppink',
              'darkgreen','midnightblue','darkviolet','steelblue','peru','orangered']
N = 256
yellow = np.ones((N, 4))
yellow[:, 0] = np.linspace(255/256, 1, N) # R = 255
yellow[:, 1] = np.linspace(232/256, 1, N) # G = 232
yellow[:, 2] = np.linspace(11/256, 1, N)  # B = 11
yellow_cmp = ListedColormap(yellow)
yellow_cmp_r = ListedColormap(yellow_cmp.colors[::-1])

eta2_1pc_group = df[df.loc['co2_cap [%]'].index[df.loc['co2_cap [%]'] == co2s[0]]].T.groupby('eta2 [-]')
eta2s = np.sort(df[df.loc['co2_cap [%]'].index[df.loc['co2_cap [%]'] == co2s[0]]].loc['eta2 [-]'].unique())
df_all = {} # multiindex dataframe instead
df_min = {}
df_max = {}
df_mid = {}
df_max_1 = 0
for co2 in co2s:
    df_1p = df[df.loc['co2_cap [%]'][df.loc['co2_cap [%]'] == co2].index].T
    df_1p.set_index(df_1p['c_hat [EUR/kWh]'],inplace=True)
    for eta2 in eta2s:
        df_eta2 = df_1p[df_1p['eta2 [-]'] == eta2]#df_1p[df_1p['eta1 [-]'] == eta2]
        eta1s = np.sort(df_eta2['eta1 [-]'].unique())
        for eta1 in eta1s:
            df_eta1 = df_eta2[df_eta2['eta1 [-]'] == eta1]
            c1s = df_eta1['c1'].unique()
            for c1 in c1s:
                df_c1 = df_eta1[df_eta1['c1'] == c1]
                c2s = df_c1['c2'].unique()
                for c2 in c2s:
                    df_c2 = df_c1[df_c1['c2'] == c2]
                    df_max_2 = df_c2['E [TWh]'].max()
                    max_2_index = [eta1,eta2,c1,c2]
                    if df_max_2 > df_max_1:
                        df_max_1 = df_max_2
                        max_1_index = max_2_index 
                        
                    df_tau_min = df_c2[df_c2['tau [n_days]'] == 10]
                    df_tau_max = df_c2[df_c2['tau [n_days]'] == 30]

                    df_min[co2,eta1,eta2,c1,c2] = df_tau_min
                    df_max[co2,eta1,eta2,c1,c2] = df_tau_max

for co2 in co2s:
    df_1p = df[df.loc['co2_cap [%]'][df.loc['co2_cap [%]'] == co2].index].T
    df_1p.set_index(df_1p['c_hat [EUR/kWh]'],inplace=True)
    for eta2 in eta2s:
        df_eta2 = df_1p[df_1p['eta2 [-]'] == eta2]#df_1p[df_1p['eta1 [-]'] == eta2]
        eta1s = np.sort(df_eta2['eta1 [-]'].unique())
        for eta1 in eta1s:
            df_eta1 = df_eta2[df_eta2['eta1 [-]'] == eta1]
            c1s = df_eta1['c1'].unique()
            for c1 in c1s:
                df_c1 = df_eta1[df_eta1['c1'] == c1]
                c2s = df_c1['c2'].unique()
                for c2 in c2s:
                    df_c2 = df_c1[df_c1['c2'] == c2]
                    for tau in taus:
                        df_all[co2,eta1,eta2,c1,c2,tau] = df_c2[df_c2['tau [n_days]'] == tau]
#%% Plotting
plot_var = 'E [TWh]'
if plot_var == 'load_shift [%]':
    threshold = threshold_L
else:
    threshold = threshold_E
c1s = np.sort(df1.loc['c1'].unique())
c2s = np.sort(df1.loc['c2'].unique())

fig = plt.figure(figsize=(19, 10))
nrows = len(c2s)
ncols = len(c1s)
gs = gridspec.GridSpec(nrows, ncols)
gs.update(wspace=0.05)
gs.update(hspace=0.05) 
keys = list(df_min.keys())
ax = {}
ylim1 = {}
ylim2 = {}
alpha = 1
max_df = pd.DataFrame(index=df1.loc['c_hat [EUR/kWh]'].unique())
mid_df = pd.DataFrame()
eff1_df = pd.DataFrame()
eff2_df = pd.DataFrame()
cost1_df = pd.DataFrame()
cost2_df = pd.DataFrame()
config_i = pd.DataFrame()
c_count = 0

counter = np.zeros([200,nrows,ncols])
counter1 = np.zeros([200,nrows,ncols])
d = 0.1
dy = 0.05
for i in range(nrows):
    for j in range(ncols):
        if i == 0 and j == 0:
            ax[i,j] = plt.subplot(gs[i,j])
        else:
            ax[i,j] = plt.subplot(gs[i,j])
            
        if j == 0:
            ax[i,j].set_ylabel(r'$\eta_d E$' + ' [TWh' + r'$_e$' + ']',fontsize=20,labelpad=-3)
            ax[i,j].text(-0.3-d,0.6-dy,r'$c_d$', transform=ax[i,j].transAxes,fontsize = fs) 
            if i == 0:
                ax[i,j].text(-0.33-d,0.7,'Low', transform=ax[i,j].transAxes,fontsize = fs)
                ax[i,j].text(-0.3-d,0.45-2*dy,'35', transform=ax[i,j].transAxes,fontsize = fs) 
                ax[i,j].text(-0.36-d,0.3-3*dy,'€/kW', transform=ax[i,j].transAxes,fontsize = fs) 
            if i == 1:
                ax[i,j].text(-0.33-d,0.7,'', transform=ax[i,j].transAxes,fontsize = fs)
                ax[i,j].text(-0.33-d,0.45-2*dy,'350', transform=ax[i,j].transAxes,fontsize = fs) 
                ax[i,j].text(-0.36-d,0.3-3*dy,'€/kW', transform=ax[i,j].transAxes,fontsize = fs) 
            if i == 2:
                ax[i,j].text(-0.33-d,0.7,'', transform=ax[i,j].transAxes,fontsize = fs)
                ax[i,j].text(-0.33-d,0.45-2*dy,'490', transform=ax[i,j].transAxes,fontsize = fs) 
                ax[i,j].text(-0.36-d,0.3-3*dy,'€/kW', transform=ax[i,j].transAxes,fontsize = fs) 
            if i == 3:
                ax[i,j].text(-0.33-d,0.7,'High', transform=ax[i,j].transAxes,fontsize = fs)
                ax[i,j].text(-0.33-d,0.45-2*dy,'750', transform=ax[i,j].transAxes,fontsize = fs) 
                ax[i,j].text(-0.36-d,0.3-3*dy,'€/kW', transform=ax[i,j].transAxes,fontsize = fs) 
            
        if i == 0:
            if j == 0:
                ax[i,j].set_title('Low ' + r'$c_c=35$'+ ' €/kW' ,fontsize = fs)
            if j == 1:
                ax[i,j].set_title('     ' + r'$c_c=350$'+ ' €/kW',fontsize = fs)
            if j == 2:
                ax[i,j].set_title('     ' + r'$c_c=490$'+ ' €/kW',fontsize = fs)
            if j == 3:
                ax[i,j].set_title('High ' + r'$c_c=750$'+ ' €/kW',fontsize = fs)
        if i == nrows - 1:
            ax[i,j].set_xlabel('Energy capapacity \n cost [€/kWh]',fontsize=20)
        
        effs = []
        kc = 0
        
        for key in keys:
            if key[0] == co2s[0]:
                tau_upper = df_max[(co2s[0],key[1],key[2],key[3],key[4])]['tau [n_days]'].iloc[0]
                tau_lower = df_min[(co2s[0],key[1],key[2],key[3],key[4])]['tau [n_days]'].iloc[0]
                tau_mean = (tau_upper + tau_lower)/2
                if (key[3] == c1s[j]) and (key[4] == c2s[i]):
                    
                    effs.append(key[1]*key[2])
                    
                    if colorcode == 'round-trip':
                        data_max = max(eta1s)*max(eta2s) 
                        data_normalized1 = ((key[1]*key[2]))/ data_max
                    
                    if colorcode == 'eta1':
                        data_max = max(eta1s) 
                        data_normalized1 = ((key[1])/ data_max)
                        
                    if colorcode == 'eta2':
                        data_max = max(eta2s)
                        data_normalized1 = ((key[2])/ data_max)
                        
                    cmap1 = plt.cm.get_cmap('summer_r')
                    color1 = cmap1(data_normalized1)
                    
                    if colorcode != 'all':
                        color = color1
                    else:
                        color = 'grey'
                    
                    ax[i,j].fill_between(df_min[(co2s[0],key[1],key[2],key[3],key[4])].index,df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var],
                                          df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var], alpha=1, fc=color,zorder=2-data_normalized1)
                    
                    if visual == '_threshold':
                        sm4 = ax[i,j].plot(df_min[(co2s[0],key[1],key[2],key[3],key[4])].index,df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var],'.',color='grey',markersize=0.5)
                        ax[i,j].plot(df_max[(co2s[0],key[1],key[2],key[3],key[4])].index,df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var],'.',color='grey',markersize=0.5)
                    
                        counter[kc][i][j] = df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var][df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var] > threshold].shape[0] + df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var][df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var] > threshold].shape[0]
                        counter1[kc][i][j] = df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var][df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var] > 0].shape[0] + df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var][df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var] > 0].shape[0]
                    
                    for tau in taus:
                        val_eta_d = df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['eta2 [-]']
                        val_E = df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['E [TWh]']
                        val_L = df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['load_shift [%]']
                        for ii in range(len(val_E)):
                            val_x = val_E.index[ii]
                            val_y_E = val_E.iloc[ii]
                            val_y_L = val_E.iloc[ii]
                            val_y_eta_d = val_eta_d.iloc[ii]
                            
                            if (val_y_E > threshold_E) and (val_y_L > threshold_L):                                
                                eta_c = key[1]
                                eta_d = key[2]
                                c_c = key[3]
                                c_d = key[4]
                                e = df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['E [TWh]'].loc[val_x]
                                lc = df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['load_shift [%]'].loc[val_x]
                                dt = df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['E [TWh]'].loc[val_x]*1000/df_all[(co2s[0],key[1],key[2],key[3],key[4],tau)]['G_discharge [GW]'].loc[val_x]
                                config_i[c_count] = [eta_d,val_x,c_c,eta_c,tau,c_d,e,lc,i,j,dt]
                                c_count += 1
                                
                max_df[kc] = df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var]
                mult1 = 0.95
                mult2 = 1.05
                ylim1_i = mult1*df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var].min()
                ylim2_i = mult2*df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var].max()
                if kc == 0:
                    ylim1[i] = mult1*df_min[(co2s[0],key[1],key[2],key[3],key[4])][plot_var].min()
                    ylim2[i] = mult2*df_max[(co2s[0],key[1],key[2],key[3],key[4])][plot_var].max()
                else:
                    if ylim1_i < ylim1[i]:
                        ylim1[i] = ylim1_i
                    if ylim2_i > ylim2[i]:
                        ylim2[i] = ylim2_i
                ax[i,j].set_xlim([df1.loc['c_hat [EUR/kWh]'].min(),df1.loc['c_hat [EUR/kWh]'].max()])
            kc += 1
        if j != 0:
            ax[i,j].axes.get_yaxis().set_major_formatter(NullFormatter())
        if i != (nrows - 1):
            ax[i,j].axes.get_xaxis().set_visible(False)
                
        if (i == 1) and (j == 1):
            ax[i,j].axvline(3,color='grey',lw=0.2)

            ax[i,j].axhline(0.260*0.5,color='grey',lw=0.2)
            sm3=ax[i,j].plot(3,0.260*0.5,color='k',marker='X',lw=0.5,markersize=10,label='Ref.')

        ax[i,j].set_yscale('log')
        
        if j != 0:
            ax[i,j].set_yticks([])
        else:
            ax[i,j].set_yticks([1,2,6,18])
            ax[i,j].set_yticklabels(['1','2','6','18'])
        
eff1_stack = eff1_df.stack().reset_index(level=[0,1])
eff1_stack.columns = ['c_hat', 'no','eta1']
eff2_stack = eff2_df.stack().reset_index(level=[0,1])
eff2_stack.columns = ['c_hat', 'no','eta2']

cost1_stack = cost1_df.stack().reset_index(level=[0,1])
cost1_stack.columns = ['c_hat', 'no','c1']

max_df = max_df.fillna(0)

config = config_i.T
config.rename(columns={0: 'eta_d',1:'c_hat',2:'c_c',3:'eta_c',4:'tau_SD',5:'c_d',6:'E',7:'lc',10:'dt'},inplace=True)
config.reset_index(inplace=True,drop=True)

for i in range(nrows):
    config_is = config.loc[config[8][config[8] == i].index]
    for j in range(ncols): 
        ax[i,j].tick_params(axis='both', which='major', labelsize=ls)
        ax[i,j].set_axisbelow(True)
        sm1 = ax[i,j].plot(max_df.max(axis=1).index, max_df.max(axis=1), ls='--',color='grey', lw=2,zorder=10,label='Max') # + ', ' + r'$\tau = $' + str(tau_mean) + r'$\pm$' + str(tau_upper-tau_mean) + ' days')
        
        ax[i,j].set_ylim([1e-1,3.5e1]) 
        ax[i,j].text(0.3,0.8,'n = ' + str(int(counter.sum(axis=0)[i][j])) + ' (' + str(int(counter1.sum(axis=0)[i][j])) + ')', transform=ax[i,j].transAxes,fontsize = fs,color='grey') 
        
        if visual == '_threshold':
            sm2 = ax[i,j].fill_between(max_df.max(axis=1).index, np.ones(len(max_df.max(axis=1)))*threshold, max_df.max(axis=1),
                                  where=[(max_df.max(axis=1).iloc[i]>=(np.ones(len(max_df.max(axis=1)))*threshold)[i]*0.9) for i in range(len(max_df.max(axis=1).index))],
                                  color='red', alpha=0.3,lw=0,label=r'$E \geq $' + str(threshold) + ' TWh',zorder=-1)
            
            ax[i,j].axhline(threshold,xmin=0,xmax=20,ls='--',color='#ffb2b2',zorder=100,lw=1.5)
        
        if plot_var == 'E [TWh]':
            ax_r = ax[i,j].twinx()
            ax_r.tick_params(axis='both', which='major', labelsize=ls)
            ax_r.set_yscale('log')
            ax_r.set_ylim([ax[i,j].get_ylim()[0]/0.320,ax[i,j].get_ylim()[1]/0.320]) # Europe load in 2018 was 2806 TWh    
            
            if (i!=3) and (j==(ncols-1)):
                ax_r.set_yticks([1,2,6,20,60])
                ax_r.set_yticklabels([1,2,6,20,60])
                ax_r.set_ylabel('av.h.l' + r'$_{el}$',fontsize=20)
            else:
                ax_r.set_yticklabels([])
                
            ax_r.spines['top'].set_visible(False)

if visual == '_threshold':
    fig.legend(sm1 + [sm2] + sm3,['max',r'$E \geq $' + str(threshold) + ' TWh','Reference storage-X parameters'],bbox_to_anchor=(0.75, -0.05), loc='center right',frameon=True,ncol=3,borderaxespad=0,prop={'size': 20})
else:
    fig.legend(sm1,['max'],loc='center right',frameon=True,ncol=1,borderaxespad=0,prop={'size': fs}) # loc='lower center',frameon=False,ncol=3

bounds = get_unique_numbers(effs)
divider = make_axes_locatable(ax[i,j])

if colorcode == 'round-trip':
    fig.text(0.91, 0.28, r'RTE [%]',fontsize=20)

if colorcode == 'eta1':
    fig.text(0.915, 0.28, r'$\eta_1$',fontsize=fs)
    
if colorcode == 'eta2':
    fig.text(0.915, 0.28, r'$\eta_2$',fontsize=fs)

cb_ax = fig.add_axes([0.91,0.11,0.02,0.16])
norm = mpl.colors.Normalize(vmin=0, vmax=data_max)    
cb1 = mpl.colorbar.ColorbarBase(cb_ax,orientation='vertical', cmap=mpl.cm.summer_r,norm=norm) #,ticks=bounds, boundaries=bounds) #ticks=[0.15,0.25,0.48,0.90])

cb1.set_ticks([0,0.2,0.4,0.6,0.8])
cb1.set_ticklabels(['','20','40','60','80'])
cb1.ax.tick_params(labelsize=ls)

fig.savefig('../figures/Storage_matrix_' + plot_var + '_' + str(threshold) + '.pdf', transparent=True,
            bbox_inches="tight")
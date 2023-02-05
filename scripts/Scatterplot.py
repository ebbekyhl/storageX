# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:27:51 2022

@author: au485969
"""
# import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import warnings
warnings.filterwarnings('ignore')
plt.close('all')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define here the system %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# sector = 0 # no sector-coupling
# sector = 'T-H' # Transportation and heating
sector = 'T-H-I-B' # Transportation, heating, industry (elec + feedstock), and biomass

# filename = '../results/sspace_3888.csv' (does not include sector-coupling)
filename = '../results/sspace_w_sectorcoupling_merged.csv'

color = 'c_c'
# color = 'RTE'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plotting layout
fs = 17
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

def scatter_hist(x, y, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # now determine nice limits by hand:
    # bw1_factor = 0.01  # <---------------- This one takes quite a while to compute
    bw1_factor = 0.5
    binwidth = bw1_factor*(x.max() - x.min())
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, color='grey', bins=bins)
    
    # bw2_factor = 0.01  # <---------------- This one takes quite a while to compute
    bw2_factor = 0.5
    binwidth = bw2_factor*(y.max() - y.min())
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histy.hist(y, orientation='horizontal',color='grey', bins=bins)

# Setup
df = pd.read_csv(filename,index_col=0).fillna(0)
df_T = df.T
df_T = df_T.query('sector == @sector').drop(columns='sector').astype(float)
df_T = df_T.sort_values(by='c_hat [EUR/kWh]')
df_T['E [TWh]'] = df_T['E [GWh]']*1e-3 # convert GWh to TWh

df_T = df_T.loc[df_T['eta1 [-]'][df_T['eta1 [-]'] <= 1].index] # <-------- Omit charge efficiencies above 1

df = df_T.reset_index(drop=True).T
threshold_E = 0 # TWh

# For "E" on the x-axis, uncomment this:
var = 'E [GWh]'
var_str = 'E'
yval = 'E'

# For "LC" on the x-axis, uncomment this: 
# yval = 'lc'
# if filename == '../results/sspace.csv' or filename == '../results/sspace_3888.csv':
#     var = 'load_shift [%]'
#     var_str = 'load_shift'
# else:
#     var = 'load_coverage [%]'
#     var_str = 'load_coverage'

x_low = df[df.loc['E [GWh]'].idxmin()]
df.loc['E [GWh]'] = df.loc['E [GWh]']*df.loc['eta2 [-]']/1000
x = df[df.loc['E [GWh]'][df.loc['E [GWh]'] > threshold_E].index]

# Reference store
r_1 = df[df.loc['eta1 [-]'][df.loc['eta1 [-]'] == 0.5].index]
r_11 = r_1[r_1.loc['eta2 [-]'][r_1.loc['eta2 [-]'] == 0.5].index]
if filename == '../results/sspace_3888.csv':
    r_111 = r_11[r_11.loc['c_hat [EUR/kWh]'][r_11.loc['c_hat [EUR/kWh]'] == 1.997].index]
else:
    r_111 = r_11[r_11.loc['c_hat [EUR/kWh]'][r_11.loc['c_hat [EUR/kWh]'] == 2].index]
r_1111 = r_111[r_111.loc['c1'][round(r_111.loc['c1']) == 350].index]
r_11111 = r_1111[r_1111.loc['c2'][round(r_1111.loc['c2']) == 350].index]
r_111111 = r_11111[r_11111.loc['tau [n_days]'][r_11111.loc['tau [n_days]'] == 30].index]

Xs = {'eta1 [-]':x.loc['eta1 [-]'].unique(), 'eta2 [-]':x.loc['eta2 [-]'].unique(), 
      'c1':x.loc['c1'].unique(),'c2':x.loc['c2'].unique(),'c_hat [EUR/kWh]':x.loc['c_hat [EUR/kWh]'].unique(),
      'tau [n_days]':x.loc['tau [n_days]'].unique()}

X_name_dict = {'eta1 [-]':r'$\eta_c$','eta2 [-]':r'$\eta_d$','c1':r'$c_c$','c2':r'$c_d$','c_hat [EUR/kWh]':r'$\hat{c}$','tau [n_days]':r'$\tau_{SD}$'}
X_name_short_dict = {'RTE':'RTE','eta1 [-]':'eta1','eta2 [-]':'eta2','c1':'c1','c2':'c2','c_hat [EUR/kWh]':'chat','tau [n_days]':'tau'}
ylab = {'dt':r'$\frac{E}{G_d}$' + ' [h]','E':'Energy capacity [GWh]','lc':'Load coverage ' + r'$LC^X$' + ' [%]','c_sys':'System cost [bEUR]','c1':'Charge capacity cost [EUR/kW]'}
marker_dic = {35:'*',
              350:'s',
              490:'^',
              700:'X'}

marker_dic1 = {0.0625:'.',
               0.125:'*',
               0.2375:'<',
               0.25:'s',
               0.475:'^',
               0.9025:'X'}
#%% Plot system cost and curtailment
if color == 'RTE':
    cmap1 = plt.cm.get_cmap('summer_r')
else:
    cmap1 = plt.cm.get_cmap('summer')

xlab = {'c_sys [bEUR]': 'Normalized system cost','battery_' + var_str +' [%]':'Battery load coverage ' + r'$LC^B$' + ' [%]','curtailment':'Average curtailed energy [%]'}
ylims = {('c_sys [bEUR]',0):[0.86,1.005],
         ('c_sys [bEUR]','T-H'):[0.86,1.005],
         ('c_sys [bEUR]','T-H-I-B'):[0.91,1.005],
         ('curtailment',0):[0,7.6],
         ('curtailment','T-H'):[0,7.6],
         ('curtailment','T-H-I-B'):[0,3],
         }
for t in ['curtailment','c_sys [bEUR]']:
    print(t)
    fig = plt.figure(figsize=(8, 7))
    nrows = 5
    ncols = 5
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0)
    gs.update(hspace=0)
    t_it = 0
    
    cp = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if t == 'curtailment':
        avail_energy = x.loc['avail_solar [MWh]'] + x.loc['avail_onwind [MWh]'] + x.loc['avail_offwind [MWh]']
        used_energy = x.loc['used_solar [MWh]'] + x.loc['used_onwind [MWh]'] + x.loc['used_offwind [MWh]']
        gen_tech_rel_thres = (avail_energy - used_energy)/avail_energy*100
    else:
        if t == 'c_sys [bEUR]':
            # gen_tech_rel_all = df.loc[t]/df.loc[t].loc[r_111111.T.index].item()
            gen_tech_rel_all = df.loc[t]/df.loc[t].loc[r_111111.T.index].unique()[0]
            
            # gen_tech_rel_thres = x.loc[t]/df.loc[t].loc[r_111111.T.index].item()
            gen_tech_rel_thres = x.loc[t]/df.loc[t].loc[r_111111.T.index].unique()[0]
            
        else:
            gen_tech_rel_all = df.loc[t]
            gen_tech_rel_thres = x.loc[t]

    lc_thres = x.loc[var]
    c_c = x.loc['c1']
    c_d = x.loc['c2']
    c_hat = x.loc['c_hat [EUR/kWh]']
    
    ax = plt.subplot(gs[1:,0:-1])
    ax.grid()
    x_plot = lc_thres
    y_plot = gen_tech_rel_thres
    
    RTEs = (df.loc['eta1 [-]']*df.loc['eta2 [-]']).unique()
    RTEs.sort()
    it = 0
    for X1 in Xs['eta2 [-]']:
        x_eta2 = x.T.groupby(x.T['eta2 [-]']).get_group(X1)
        for X2 in x_eta2['eta1 [-]'].unique():
            x_eta1 = x_eta2.groupby(x.T['eta1 [-]']).get_group(X2)
            RTE = X1*X2
            if color == 'RTE':
                color_var = RTE/(0.95*0.95) # Color according to round-trip efficiency
            
            # color_var = X1/0.95 # Color according to discharge efficiency
            # color_var = X2/0.95 # Color according to discharge efficiency
            # color_var = c_hat.loc[x_eta1.index]/40  # Color according to energy capacity cost
            # color_var = c_d.loc[x_eta1.index]/700  # Color according to discharge capacity cost
            
            if color == 'c_c':
                color_var = c_c.loc[x_eta1.index]/700  # Color according to charge capacity cost
            
            gen_tech_rel_y = y_plot.loc[x_eta1.index] 
            lc_x = x_eta1[var] 
            # c_c = x_eta1['c1']
            color1 = cmap1(color_var)
            
            # ax.scatter(lc_x,gen_tech_rel_y,marker='s',s=10,color=color1,zorder=1000-RTE)
            
            # ax.scatter(lc_x,gen_tech_rel_y,marker=marker_dic1[RTE],s=20,color=color1,zorder=1000-RTE)
            
            if color == 'c_c':
                for j in range(len(lc_x)):
                    ax.scatter(lc_x.iloc[j],gen_tech_rel_y.iloc[j],marker='s',s=10, zorder=color_var.iloc[j],color=color1[j]) 
            else:
                ax.scatter(lc_x,gen_tech_rel_y,marker='s',s=10,color=color1,zorder=1000-RTE)
            

            it += 1
            
    cb_ax = fig.add_axes([0.90,0.11,0.02,0.3])
    # cb_ax.set_zorder(100)
    
    if color == 'RTE':
        cmap = mpl.cm.summer_r
    if color == 'c_c':
        cmap = mpl.cm.summer
    
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    if color == 'c_c':
        RTEs = x.loc['c1'].unique()
        RTEs.sort()
        fig.text(0.91, 0.45, r'$c_c$',fontsize=15)
    if color == 'RTE':
        fig.text(0.91, 0.45, r'RTE [%]',fontsize=15)
        RTEs = RTEs*100
    bounds = [0] + list(RTEs.astype(int))
    
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = mpl.colorbar.ColorbarBase(cb_ax, orientation='vertical',cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    cb1.set_ticks(RTEs - np.array([RTEs[0]] + list(np.diff(RTEs)))*0.5)
    cb1.set_ticklabels((RTEs).astype(int))
    cb1.ax.tick_params(labelsize=15)
    
    ax_histx = plt.subplot(gs[0:1,0:-1],sharex=ax)
    ax_histy = plt.subplot(gs[1:,-1],sharey=ax)
    
    scatter_hist(x_plot, y_plot, ax_histx, ax_histy) # <---------------- This one takes quite a while to compute
    
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    ax_histx.axis('off')
    ax_histy.axis('off')
    
    ax.set_ylabel(xlab[t])
    ax.set_xlabel(ylab[yval])
    
    if yval == 'lc':
        ax.set_xlim([-0.1,20])
    else:
        ax.set_xlim([-3,70])
        
    ax.set_ylim(ylims[(t,sector)])
    if color == 'c_c':
        fig.savefig('../figures/scatter_plot_' + t + '_' + yval + str(sector) + '_c_c.png', transparent=False,
                    bbox_inches="tight",dpi=300)
    else:
        fig.savefig('../figures/scatter_plot_' + t + '_' + yval + '_' + str(sector) + '.png', transparent=False,
                    bbox_inches="tight",dpi=300)
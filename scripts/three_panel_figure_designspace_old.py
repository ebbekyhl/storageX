# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:31:23 2023

@author: au485969
"""

import pandas as pd
import numpy as np

# df = df_chat_eta2
# var1='eta2'
# var2='c_hat'
# var_name1=r'$\hat{c}$' + ' [€/kWh]'
# var_name2=r'$\eta_d$'+ ' [-]'
# ax=ax[2]
# normfactor=normfactor
# shading=shading
# colormap=cmap

def check_boundary(variable_values, variable_names):
    uni_max_dic = {'c_hat': 40, 
                   'c1': 700,
                   'c2': 700,
                   'eta1': 0.25, 
                   'eta2': 0.25}
    output = []
    for i in range(len(variable_names)):
        if variable_values[i] == uni_max_dic[variable_names[i]]:
            output.append('')
        else:
            output.append(variable_values[i])
    
    return output

def annotate(df, df1_update, color_variable, nrows, ncols, var1, var2, var_name1, var_name2, ax, normfactor, write_extra_indices, shading='auto', colormap="cool_r"):
    fs=15
    Z = df[color_variable].values.reshape(nrows, ncols)
    x = np.arange(ncols) 
    # x_l = np.arange(-0.5,ncols,0.5)
    y = np.arange(nrows)
    # y_l = np.arange(-0.5,nrows,0.5)
    vmin = 0
    im = ax.pcolormesh(x, y, Z, vmin=vmin, vmax=normfactor,shading=shading, cmap=colormap,zorder=0)
    
    for x_i in x:
        for y_i in y:
            if Z[y_i,x_i] == 0:
                ax.fill_between([x_i-0.5,x_i+0.5], y_i+0.5, y_i-0.5,facecolor="none",hatch="/",edgecolor="grey", linewidth=0.0)
    
    ax.scatter(np.meshgrid(x, y)[0],np.meshgrid(x, y)[1],color='grey',alpha=0.5)
    
    if write_extra_indices:
        strcolor1 = 'grey' # Text font color of "remaining parameters"
        strcolor2 = 'k'
        # The indices are ordered in the following way: 'eta1','eta2','c1','c2','c_hat'
        if var1 == 'eta2' and var2 == 'eta1':
            A_name = '$c_c$'
            A_round = 0
            B_name = '$c_d$'
            B_round = 0
            C_name = '$\hat{c}$'
            xdis = 0.75
            C_round = 0
            
        elif var1 == 'eta2' and var2 == 'c_hat':
            A_name = '$\eta_c$'
            A_round = 2
            B_name = '$c_c$'
            B_round = 0
            C_name = '$c_d$'
            C_round = 0
            xdis = 1.1
            
        elif var1 == 'c2' and var2 == 'c1':
            A_name = '$\eta_c$'
            A_round = 2
            B_name = '$\eta_d$'
            B_round = 2
            C_name = '$\hat{c}$'
            C_round = 0
            xdis = 0.8
        
        for count in range(np.meshgrid(x, y)[0].size):
            ii = np.meshgrid(x, y)[0].flatten()[count]
            jj = np.meshgrid(x, y)[1].flatten()[count]
            text_var = df['extra_coordinates'].values[count]
            
            if A_round == 0:
                A = str(int(text_var[0])) if type(text_var[0]) == np.float64 else str(text_var[0])
            else:
                A = str(text_var[0].round(A_round)) if type(text_var[0]) == np.float64 else str(text_var[0])
                
            if B_round == 0:
                B = str(int(text_var[1])) if type(text_var[1]) == np.float64 else str(text_var[1])
            else:
                B = str(text_var[1].round(B_round)) if type(text_var[1]) == np.float64 else str(text_var[1])
                
            if C_round == 0:
                C = str(int(text_var[2])) if type(text_var[2]) == np.float64 else str(text_var[2])
            else:
                C = str(text_var[2].round(C_round)) if type(text_var[2]) == np.float64 else str(text_var[2])
            
            if count == ncols*(nrows-1):
                ax.text(ii-xdis,jj-0.1,A_name,zorder=11, horizontalalignment='left', verticalalignment = 'top',color=strcolor1,fontsize=fs)
                ax.text(ii-xdis,jj,B_name,zorder=11, horizontalalignment='left', verticalalignment = 'center',color=strcolor1,fontsize=fs)
                ax.text(ii-xdis,jj+0.1,C_name,zorder=11, horizontalalignment='left', verticalalignment = 'bottom',color=strcolor1,fontsize=fs)
            
            ax.text(ii,jj-0.1,str(A),zorder=11, horizontalalignment='center', verticalalignment = 'top',color=strcolor2,fontsize=fs)
            ax.text(ii,jj,str(B),zorder=11, horizontalalignment='center', verticalalignment = 'center',color=strcolor2,fontsize=fs)
            ax.text(ii,jj+0.1,str(C),zorder=11, horizontalalignment='center', verticalalignment = 'bottom',color=strcolor2,fontsize=fs)
    
    ax.set_yticks(np.arange(nrows))
    ax.set_xticks(np.arange(ncols))
    
    if var1 == 'eta1' or var1 == 'eta2':
        ax.set_yticklabels(np.sort(df1_update[var1].unique()))
    else:
        ax.set_yticklabels(np.sort(df1_update[var1].unique()).astype(int))
    
    if var2 == 'eta1' or var2 == 'eta2':
        ax.set_xticklabels(np.sort(df1_update[var2].unique()))
    else:
        ax.set_xticklabels(np.sort(df1_update[var2].unique()).astype(int))
    ax.set_xlabel(var_name1)
    ax.set_ylabel(var_name2,labelpad=0)
    
    ax.grid(True, color="grey", lw=1, zorder = 10,alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', which='major', pad=17)
    return im

def read_sspace(sspace_og,sector,output,lock_tau,omit_charge_efficiency):
    sspace = sspace_og.fillna('0').T
    sspace = sspace.query('sector == @sector')
    sspace = sspace.drop(columns = 'sector').T.astype(float)
    # Input
    df1 = pd.DataFrame(columns=['c_hat'])
    df1['c_hat'] = sspace.loc['c_hat [EUR/kWh]']
    df1['c1'] = sspace.loc['c1']
    df1['eta1'] = sspace.loc['eta1 [-]']
    df1['c2'] = sspace.loc['c2']
    df1['eta2'] = sspace.loc['eta2 [-]']
    df1['tau_SD'] = sspace.loc['tau [n_days]']
    
    if lock_tau:
        df1 = df1.loc[df1['tau_SD'][df1['tau_SD'] == 30].index] 
    
    # Output
    # if output == 'E_cor':
    
    df1['E_cor'] = sspace.loc['E [GWh]']*df1['eta2']
    df1['lc'] = sspace.loc['load_coverage [%]']
    df1['SCR'] = (1 - sspace.loc['c_sys [bEUR]']/sspace.loc['c_sys [bEUR]'].max())*100
    
    # elif output == 'lc':
        # df1['lc'] = sspace.loc['load_coverage [%]'].astype(float)
        # df1['E_cor'] = sspace.loc['E [GWh]']*df1['eta2']
    # else:
        # df1[output] = sspace.loc[output].astype(float)
        # df1['E_cor'] = sspace.loc['E [GWh]']*df1['eta2']
        
    if omit_charge_efficiency:
        df1_update = df1.loc[df1['eta1'][df1['eta1'] < 1].index] # Remove all charge efficiencies above or equal to 1
    else:
        df1_update = df1
        
    MI_df = df1_update[['eta1','eta2','c1','c2','c_hat',output]].copy()
    MI_df = MI_df.set_index(['eta1','eta2','c1','c2','c_hat']) 
    MI_df.sort_values(['eta1','eta2','c1','c2','c_hat'],inplace=True)
    
    MI_df.rename(columns={output:'output'},inplace=True)
    
    df1_update = df1_update.query("E_cor > 1") # Omit "zero" storage
    
    if output != 'E_cor':
        df1_update.drop(columns=['E_cor'],inplace=True)
    
    return df1_update, MI_df

# sector='T-H-I-B'
# combination = False
# write_extra_indices = True
# slack=100
# threshold=2000
# normfactor = 100
# color_variable = 'count_norm'
# omit_charge_efficiency = True
# lock_tau = False

def plot_2D_panels(sector, 
                   output,
                   slack, 
                   threshold=1, 
                   normfactor=100, 
                   color_variable='count_norm', 
                   combination=True, 
                   write_extra_indices = False,
                   omit_charge_efficiency = True, 
                   lock_tau = False):
   
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="ticks")
    import matplotlib as mpl
    fs = 18
    plt.style.use('seaborn-ticks')
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['axes.axisbelow'] = True 
    
    sspace_og = pd.read_csv('results/sspace_w_sectorcoupling_wo_duplicates.csv',index_col=0)
    # sspace_og = pd.read_csv('results/sspace_3888.csv',index_col=0)
    
    shading = 'nearest' # No interpolation or averaging
    # shading = 'flat' # The color represent the average of the corner values
    # shading='gouraud' # Gouraud: the color in the quadrilaterals is linearly interpolated
    
    cmap = "cool"
    # cmap = "spring_r"
    
    # output = 'E_cor'
    # output = 'lc'
    # output = 'c_sys [bEUR]'
    
    threshold = threshold - slack
    
    # normfactor = 100 #threshold_E # 2000 # what storage-X needs to provide in terms of cumulative storage energy capacity
    # normfactor = 2 # what storage-X needs to provide in terms of cumulative load coverage over a year
    # normfactor = 1
    
    figsiz = [18,4]
    df1_update,MI_df = read_sspace(sspace_og,sector,output,lock_tau,omit_charge_efficiency)
    df1_update['output'] = df1_update[output]
    #%% Loop over quantiles
    fig, ax = plt.subplots(1,3,figsize=figsiz)
    plt.subplots_adjust(wspace=0.3,
                        hspace=0.3)
    
    eta1s = np.array([0.25,0.50,0.95], dtype=object)
    eta2s = np.array([0.25,0.50,0.95], dtype=object)
    index_names = ['eta2','eta1']
    multiindex = pd.MultiIndex.from_product([eta2s, eta1s], names=index_names)
    a = np.empty(len(multiindex))
    a[:] = np.nan
    df_etas = pd.DataFrame(a,index=multiindex)
    df_etas.columns = ['count']
    
    c1s = np.array([35,350,490,700], dtype=object)
    c2s = np.array([35,350,490,700], dtype=object)
    index_names = ['c2','c1']
    multiindex = pd.MultiIndex.from_product([c2s, c1s], names=index_names)
    a = np.empty(len(multiindex))
    a[:] = np.nan
    df_cs = pd.DataFrame(a,index=multiindex)
    df_cs.columns = ['count']
    
    c_hats = np.array([1,2,5,10,20,30,40], dtype=object)
    index_names = ['eta2','c_hat']
    multiindex = pd.MultiIndex.from_product([eta2s, c_hats], names=index_names)
    a = np.empty(len(multiindex))
    a[:] = np.nan
    df_chat_eta2 = pd.DataFrame(a,index=multiindex)
    df_chat_eta2.columns = ['count']
    
    # Efficiency
    df_etas_count = df1_update.query("output >= @threshold")[['eta1','eta2',output]].groupby(['eta2','eta1']).count() #quantile(quantile,interpolation='nearest').copy()
    df_etas_E = df1_update.query("output >= @threshold")[['eta1','eta2',output]].groupby(['eta2','eta1']).min() #quantile(quantile,interpolation='nearest').copy()
    df_etas.loc[df_etas_count.index] = df_etas_count
    
    a = np.empty(len(df_etas))
    a[:] = np.nan
    df_etas['count_all'] = a
    df_etas['count_all'].loc[df1_update.groupby(['eta2','eta1']).count().index] = df1_update.groupby(['eta2','eta1']).count()[output]
    df_etas['count'].loc[df_etas['count_all'].dropna().index] = df_etas['count'].loc[df_etas['count_all'].dropna().index].fillna(0)
    df_etas['count_norm'] = np.zeros(len(df_etas))
    df_etas['count_norm'].loc[df1_update.groupby(['eta2','eta1']).count().index] = (df_etas.loc[df1_update.groupby(['eta2','eta1']).count().index].values.T/(df1_update.groupby(['eta2','eta1']).count()[output].values)*100)[0]
    
    # df_etas['count_norm'] = np.zeros(len(df_etas))
    # df_etas['count_norm'].loc[df1_update.groupby(['eta2','eta1']).count().index] = (df_etas.loc[df1_update.groupby(['eta2','eta1']).count().index].values.T/(df1_update.groupby(['eta2','eta1']).count()[output].values)*100)[0]
    df_etas[output] = np.zeros(len(df_etas))
    df_etas[output].loc[df_etas_count.index] = df_etas_E[output].values
    df_etas[output].loc[list(df_etas['count'][df_etas['count'] == 0].index.values)] = df1_update.groupby(['eta2','eta1']).max().loc[list(df_etas['count'][df_etas['count'] == 0].index.values)][output]
    
    # -------------------------- #
    extra_indeces = [] # We are reducing the space from 5D to 2D. Here, we collect descriptors from the omitted 3D space.
    # The indices are ordered in the following way: 'eta1','eta2','c1','c2','c_hat'
    for i in range(len(df_etas)):
        out = df_etas.iloc[i][output].item()
        out1 = df_etas.iloc[i]['count'].item()
        if np.isnan(out1) or out1 == 0:
            list_add = ('','','')
            extra_indeces.append(list_add)
        else:    
            if combination == True:
                list_add = MI_df.query("output == @out").index[0][2:]
            else:
                max_ind = df1_update.query('output >= @threshold').groupby(['eta2','eta1']).max().loc[df_etas.iloc[i].name]
                
                max_c_hat = max_ind.loc['c_hat']
                max_c1 = max_ind.loc['c1']
                max_c2 = max_ind.loc['c2']
                
                [max_c_hat, max_c1, max_c2] = check_boundary([max_c_hat, max_c1, max_c2],['c_hat','c1','c2'])
    
                list_add = (max_c1, max_c2, max_c_hat)
            
            extra_indeces.append(list_add)
    df_etas['extra_coordinates'] = extra_indeces
    # ---------------------------#
    
    # Power capacity cost
    df_cs_count = df1_update.query("output >= @threshold")[['c1','c2',output]].groupby(['c2','c1']).count() #quantile(quantile,interpolation='nearest').copy()
    df_cs_E = df1_update.query("output >= @threshold")[['c1','c2',output]].groupby(['c2','c1']).min() #quantile(quantile,interpolation='nearest').copy()
    df_cs.loc[df_cs_count.index] = df_cs_count
    
    a = np.empty(len(df_cs))
    a[:] = np.nan
    df_cs['count_all'] = a
    df_cs['count_all'].loc[df1_update.groupby(['c2','c1']).count().index] = df1_update.groupby(['c2','c1']).count()[output]
    df_cs['count'].loc[df_cs['count_all'].dropna().index] = df_cs['count'].loc[df_cs['count_all'].dropna().index].fillna(0)
    df_cs['count_norm'] = np.zeros(len(df_cs))
    df_cs['count_norm'].loc[df1_update.groupby(['c2','c1']).count().index] = (df_cs.loc[df1_update.groupby(['c2','c1']).count().index].values.T/(df1_update.groupby(['c2','c1']).count()[output].values)*100)[0]
    
    df_cs[output] = np.zeros(len(df_cs))
    df_cs[output].loc[df_cs_count.index] = df_cs_E[output].values
    df_cs[output].loc[list(df_cs['count'][df_cs['count'] == 0].index.values)] = df1_update.groupby(['c2','c1']).max().loc[list(df_cs['count'][df_cs['count'] == 0].index.values)]['output']
    
    # ---------------------------#
    # The indices are ordered in the following way: 'eta1','eta2','c1','c2','c_hat'
    extra_indeces = []
    for i in range(len(df_cs)):
        out = df_cs.iloc[i][output].item()
        out1 = df_cs.iloc[i]['count'].item()
        if np.isnan(out1) or out1 == 0:
            list_add = ('','','')
            extra_indeces.append(list_add)
        else:    
            if combination == True:
                list_adds = MI_df.query("output == @out").index[0]
                list_add = list_adds[0:2] + (list_adds[-1],) 
            else:
                max_ind = df1_update.query('output >= @threshold').groupby(['c2','c1']).max().loc[df_cs.iloc[i].name]
                min_ind = df1_update.query('output >= @threshold').groupby(['c2','c1']).min().loc[df_cs.iloc[i].name]
                
                max_c_hat = max_ind.loc['c_hat']
                min_eta1 = min_ind.loc['eta1']
                min_eta2 = min_ind.loc['eta2']
                
                [min_eta1, min_eta2, max_c_hat] = check_boundary([min_eta1, min_eta2, max_c_hat],['eta1','eta2','c_hat'])
                
                list_add = (min_eta1,min_eta2,max_c_hat)
            
            extra_indeces.append(list_add)
    df_cs['extra_coordinates'] = extra_indeces
    # ---------------------------#
    
    # Energy capacity cost
    df_chat_eta2_count = df1_update.query("output >= @threshold")[['c_hat','eta2',output]].groupby(['eta2','c_hat']).count() #quantile(quantile,interpolation='nearest').copy()
    df_chat_eta2_E = df1_update.query("output >= @threshold")[['c_hat','eta2',output]].groupby(['eta2','c_hat']).min() #quantile(quantile,interpolation='nearest').copy()
    df_chat_eta2.loc[df_chat_eta2_count.index] = df_chat_eta2_count
    
    a = np.empty(len(df_chat_eta2))
    a[:] = np.nan
    df_chat_eta2['count_all'] = a
    df_chat_eta2['count_all'].loc[df1_update.groupby(['eta2','c_hat']).count().index] = df1_update.groupby(['eta2','c_hat']).count()[output]
    df_chat_eta2['count'].loc[df_chat_eta2['count_all'].dropna().index] = df_chat_eta2['count'].loc[df_chat_eta2['count_all'].dropna().index].fillna(0)
    df_chat_eta2['count_norm'] = df_chat_eta2['count']/df_chat_eta2['count_all']*100
    df_chat_eta2[output] = np.zeros(len(df_chat_eta2))
    df_chat_eta2[output].loc[df_chat_eta2_count.index] = df_chat_eta2_E[output].values
    df_chat_eta2[output].loc[list(df_chat_eta2['count'][df_chat_eta2['count'] == 0].index.values)] = df1_update.groupby(['eta2','c_hat']).max().loc[list(df_chat_eta2['count'][df_chat_eta2['count'] == 0].index.values)]['output']
    
    # ---------------------------#
    # The indices are ordered in the following way: 'eta1','eta2','c1','c2','c_hat'
    extra_indeces = []
    for i in range(len(df_chat_eta2)):
        out = df_chat_eta2.iloc[i][output].item()
        out1 = df_chat_eta2.iloc[i]['count'].item()
        if np.isnan(out1) or out1 == 0:
            list_add = ('','','')
            extra_indeces.append(list_add)
        else:
            if combination == True:
                list_adds = MI_df.query("output == @out").index[0]
                list_add = (list_adds[0],) + list_adds[2:4] 
            else:
                max_ind = df1_update.query('output >= @threshold').groupby(['eta2','c_hat']).max().loc[df_chat_eta2.iloc[i].name]
                min_ind = df1_update.query('output >= @threshold').groupby(['eta2','c_hat']).min().loc[df_chat_eta2.iloc[i].name]
                max_c1 = max_ind.loc['c1']
                max_c2 = max_ind.loc['c2']
                min_eta1 = min_ind.loc['eta1']
                
                [min_eta1, max_c1, max_c2] = check_boundary([min_eta1, max_c1, max_c2],['eta1','c1','c2'])
                
                list_add = (min_eta1,max_c1,max_c2)
            
            extra_indeces.append(list_add)
    df_chat_eta2['extra_coordinates'] = extra_indeces
    # ---------------------------#
    
    # Plotting
    # Capacity cost
    nrows = 4
    ncols = 4
    # color_variable = 'count'
    annotate(df_cs, df1_update, color_variable, nrows, ncols, var1='c2', var2='c1', var_name1=r'$c_c$' + ' [€/kW]', var_name2 = r'$c_d$' + ' [€/kW]', ax=ax[0], normfactor=normfactor,  write_extra_indices=write_extra_indices, shading=shading, colormap=cmap)
    
    # Efficiency
    nrows = 3
    ncols = 3 if omit_charge_efficiency else 4
    
    # color_variable = 'count_norm'
    annotate(df_etas, df1_update, color_variable, nrows, ncols, var1='eta2',var2='eta1',var_name1=r'$\eta_c$' + ' [-]',var_name2=r'$\eta_d$' + ' [-]', ax=ax[1], normfactor=normfactor, write_extra_indices=write_extra_indices,shading=shading, colormap=cmap)
    
    # Energy capacity cost vs discharge efficiency
    nrows = 3
    ncols = 7
    # color_variable = 'count_norm'
    annotate(df_chat_eta2, df1_update, color_variable, nrows, ncols, var1='eta2',var2='c_hat',var_name1=r'$\hat{c}$' + ' [€/kWh]',var_name2=r'$\eta_d$'+ ' [-]', ax=ax[2], normfactor=normfactor,write_extra_indices=write_extra_indices,shading=shading, colormap=cmap)
       
    cb_ax = fig.add_axes([0.92,0.1,0.02,0.6])
    cb_ax.tick_params(direction='out', length=6, width=2, colors='k',
                      grid_color='k', grid_alpha=1)   
    
    norm = mpl.colors.Normalize(vmin=0, vmax=normfactor)
    cb = mpl.colorbar.ColorbarBase(cb_ax,orientation='vertical', cmap= plt.cm.get_cmap(cmap),norm=norm) #,ticks=bounds, boundaries=bounds) #ticks=[0.15,0.25,0.48,0.90])
    cb.ax.tick_params(labelsize=18)
    
    cb.set_label('Configurations \n ' + 'E' + r'$\geq$' + '2TWh (%)', rotation=90,fontsize=18,labelpad=16)
    
    fig.savefig('figures/Panel_requirements_' + sector + '_' + output + '_' + shading + '_old.png', dpi=300, bbox_inches='tight')
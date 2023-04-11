# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:51:26 2022

@author: au485969
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="ticks")
import numpy as np
import matplotlib as mpl

plt.close('all')

# Plotting function
def annotate(df, df1_update, output, nrows, ncols, var1, var2, var_name1, var_name2, ax, quantiles, q, normfactor, shading='auto', colormap="cool_r"):
    Z = df[output].values.reshape(nrows, ncols)
    x = np.arange(ncols) 
    y = np.arange(nrows)
        
    im = ax.pcolormesh(x, y, Z, vmin=0, vmax=normfactor,shading=shading, cmap=colormap,zorder=0)
    
    ax.scatter(np.meshgrid(x, y)[0],np.meshgrid(x, y)[1],color='grey',alpha=0.5)

    strcolor = 'k' # Text font color of "remaining parameters"
    
    # The indices are ordered in the following way: 'eta1','eta2','c1','c2','c_hat'
    
    if var1 == 'eta2' and var2 == 'eta1':
        A_name = r'$\langle c_c \rangle=$'
        A_round = 0
        B_name = r'$\langle c_d \rangle=$'
        B_round = 0
        C_name = r'$\langle \hat{c} \rangle=$'
        xdis = 0.6
        C_round = 0
        
    elif var1 == 'eta2' and var2 == 'c_hat':
        A_name = r'$\langle \eta_c\rangle=$'
        A_round = 2
        B_name = r'$\langle c_c \rangle=$'
        B_round = 0
        C_name = r'$\langle c_d \rangle=$'
        C_round = 0
        xdis = 1.3
        
    elif var1 == 'c2' and var2 == 'c1':
        A_name = r'$\langle \eta_c\rangle=$'
        A_round = 2
        B_name = r'$\langle \eta_d\rangle=$'
        B_round = 2
        C_name = r'$\langle \hat{c}\rangle=$'
        C_round = 0
        xdis = 0.8
    
    for count in range(np.meshgrid(x, y)[0].size):
        ii = np.meshgrid(x, y)[0].flatten()[count]
        jj = np.meshgrid(x, y)[1].flatten()[count]
        text_var = df['extra_coordinates'].values[count]
        
        if type(text_var) == tuple:
            if np.isnan(text_var[0]):
                A = ''
                B = ''
                C = ''
            else:
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
                ax.text(ii-xdis,jj-0.1,A_name,zorder=11, horizontalalignment='left', verticalalignment = 'top',color=strcolor)
                ax.text(ii-xdis,jj,B_name,zorder=11, horizontalalignment='left', verticalalignment = 'center',color=strcolor)
                ax.text(ii-xdis,jj+0.1,C_name,zorder=11, horizontalalignment='left', verticalalignment = 'bottom',color=strcolor)
            
            ax.text(ii,jj-0.1,str(A),zorder=11, horizontalalignment='center', verticalalignment = 'top',color=strcolor)
            ax.text(ii,jj,str(B),zorder=11, horizontalalignment='center', verticalalignment = 'center',color=strcolor)
            ax.text(ii,jj+0.1,str(C),zorder=11, horizontalalignment='center', verticalalignment = 'bottom',color=strcolor)
    else:
        A = ''
        B = ''
        C = ''
    
    ax.set_yticks(np.arange(nrows))
    ax.set_xticks(np.arange(ncols))
    
    if var1 == 'eta1' or var1 == 'eta2':
        ax.set_yticklabels(np.sort(df1_update[var1].unique()))
    else:
        ax.set_yticklabels(np.sort(df1_update[var1].unique()).astype(int))
    
    if q == len(quantiles)-1:
        if var2 == 'eta1' or var2 == 'eta2':
            ax.set_xticklabels(np.sort(df1_update[var2].unique()))
        else:
            ax.set_xticklabels(np.sort(df1_update[var2].unique()).astype(int))
    else:
        ax.set_xticklabels([])
    
    ax.set_xlabel(var_name1)
    ax.set_ylabel(var_name2,labelpad=-4)
    
    ax.grid(True, color="grey", lw=1, zorder = 10,alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', which='major', pad=20)
    return im

def read_sspace(sspace_og,sector,output,lock_tau,omit_charge_efficiency):
    sspace = sspace_og.fillna('0').T
    sspace = sspace.query('sector == @sector')
    sspace = sspace.drop(columns = 'sector').T.astype(float)
    df1 = pd.DataFrame(columns=['c_hat'])
    df1['c_hat'] = sspace.loc['c_hat [EUR/kWh]']
    df1['c1'] = sspace.loc['c1']
    df1['eta1'] = sspace.loc['eta1 [-]']
    df1['c2'] = sspace.loc['c2']
    df1['eta2'] = sspace.loc['eta2 [-]']
    df1['tau_SD'] = sspace.loc['tau [n_days]']
    if lock_tau:
        df1 = df1.loc[df1['tau_SD'][df1['tau_SD'] == 30].index] 
    if output == 'E_cor':
        df1['E_cor'] = sspace.loc['E [GWh]']*df1['eta2']
    elif output == 'lc':
        df1['lc'] = sspace.loc['load_coverage [%]'].astype(float)
    if omit_charge_efficiency:
        df1_update = df1.loc[df1['eta1'][df1['eta1'] < 1].index] # Remove all charge efficiencies above or equal to 1
    else:
        df1_update = df1
    MI_df_g = df1_update[['eta1','eta2','c1','c2','c_hat',output]].copy()
    MI_df_g = MI_df_g.set_index(['eta1','eta2','c1','c2','c_hat']) 
    MI_df_g.sort_values(['eta1','eta2','c1','c2','c_hat'],inplace=True)
    MI_df_g.rename(columns={output:'output'},inplace=True)
    return df1_update, MI_df_g

fs = 18
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

sspace_og1 = pd.read_csv('../Results/sspace_w_sectorcoupling.csv',index_col=0)
sspace_og2 = pd.read_csv('../Results/sspace_w_sectorcoupling_merged.csv',index_col=0)
# sspace_og = pd.read_csv('../Results/sspace_3888.csv',index_col=0)

shading = 'nearest' # No interpolation or averaging
# shading = 'flat' # The color represent the average of the corner values
# shading='gouraud' # Gouraud: the color in the quadrilaterals is linearly interpolated

omit_charge_efficiency = True # Omits charge efficiencies above 1 (Set to False to keep them in the figure)
lock_tau = False #True

cmap = "cool"
# cmap = "spring_r"

output = 'E_cor'
# output = 'lc'

normfactor = 2000 # what storage-X needs to provide in terms of cumulative storage energy capacity
# normfactor = 2 # what storage-X needs to provide in terms of cumulative load coverage over a year

#%% Loop over sectors
sectors = ['T-H-I-B'] #['0','T-H']
quantiles = [0.25,0.50,0.75,1.00]
# quantiles = [0.20,0.40,0.60,0.80]

ranges = {0: list(np.arange(0,0.25+0.25/10,0.25/10).round(3)),
          1: list(np.arange(0.25+0.25/10,0.50+0.25/10,0.25/10).round(3)),
          2: list(np.arange(0.50+0.25/10,0.75+0.25/10,0.25/10).round(3)),
          3: list(np.arange(0.75+0.25/10,1.0+0.25/10,0.25/10).round(3))}

figsiz = [16,16]
for sector in sectors:
    
    df1_update1,MI_df_g1 = read_sspace(sspace_og1,sector,output,lock_tau,omit_charge_efficiency)
    df1_update2,MI_df_g2 = read_sspace(sspace_og2,sector,output,lock_tau,omit_charge_efficiency)
    
    #%% Loop over quantiles
    fig, ax = plt.subplots(len(quantiles),3,figsize=figsiz)
    plt.subplots_adjust(wspace=0.3,
                        hspace=0.3)
    q = 0
    for quantile in quantiles:
        #%% ---------------- EFFICIENCY ----------------------------
        df_etas = df1_update1.copy()
        extra_indeces = [] # We are reducing the space from 5D to 2D. Here, we collect descriptors from the omitted 3D space.
        for i1 in range(len(df_etas)):
            E_out = df_etas[output].iloc[i1].item()   
            list_add = MI_df_g1.query("output == @E_out").index[0][2:]
            extra_indeces.append(list_add)
        df_etas['extra_coordinates'] = extra_indeces
        
        MI_df = df1_update1[['eta2','eta1','c1','c2','c_hat',output]].copy()
        MI_df = MI_df.set_index(['eta2','eta1','c1','c2','c_hat']) 
        MI_df.sort_values(['eta2','eta1','c1','c2','c_hat'],inplace=True)
        
        if q > 0:
            threshold_0 = df_etas[['eta1','eta2',output,'extra_coordinates']].groupby(['eta2','eta1']).quantile(quantiles[q-1],interpolation='nearest')
            
        threshold = df_etas[['eta1','eta2',output,'extra_coordinates']].groupby(['eta2','eta1']).quantile(quantile,interpolation='nearest') 
        
        # Get the remaining indices (averages of the configurations within the groups)
        extra_indeces = []
        for i2 in range(len(threshold)):

            if q == 0:
                list_add_avg = tuple(MI_df.loc[threshold.index[i2]][MI_df.loc[threshold.index[i2]] <= threshold[output].iloc[i2]]
                                     .dropna().reset_index(level=['c1','c2','c_hat'])
                                     .drop(columns=[output])
                                     .mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! Look at the max. Does it work as it should?
                                     .round(1).values) 
            else:
                temp = MI_df.loc[threshold.index[i2]][(MI_df.loc[threshold.index[i2]] > threshold_0[output].iloc[i2]) & (MI_df.loc[threshold.index[i2]] <= threshold[output].iloc[i2])].dropna()
                temp = temp.reset_index()
                
                if temp.shape[0] > 0:   
                    temp = temp[['c1','c2','c_hat']]
                    list_add_avg = tuple(np.abs(temp.mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! Look at the max. 
                                            .round(3).values))  
                else:
                    temp = MI_df.loc[threshold.index[i2]]
                    temp = temp.reset_index()
                    temp = temp[['c1','c2','c_hat']]
                    if temp.shape[0] > 1:
                        list_add_avg = tuple(temp
                                             .mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! Look at the max. 
                                             .values.round(3))
                    else:
                        list_add_avg = tuple(temp.values[0].round(3))
                
                
            extra_indeces.append(list_add_avg)
        threshold['extra_coordinates'] = extra_indeces
        
        df_etas = threshold.copy()
        if not omit_charge_efficiency:
            df_etas.loc[(0.95,2.0),:] = np.nan
            df_etas.sort_index(inplace=True)
        
        # Get the grouped values: 
        df_etas[output] = (pd.concat([df1_update1[['eta1','eta2',output]]
                          .groupby(['eta2','eta1'])
                          .quantile(qi,interpolation='nearest').copy() 
                          for qi in ranges[q]]).groupby(['eta2','eta1'])
                            .mean()) # <-----------------------------------------WE CAN TAKE MAX/MEAN/MEDIAN
        
        #%% ---------------- POWER CAPACITY COST ----------------------------
        df_cs = df1_update1.copy()
        
        if not omit_charge_efficiency:
            df_cs = df_cs[df_cs['eta1'] < 1]
        
        extra_indeces = [] # We are reducing the space from 5D to 2D. Here, we collect descriptors from the omitted 3D space.
        for i1 in range(len(df_cs)):
            E_out = df_cs[output].iloc[i1].item()   
            list_adds = MI_df_g1.query("output == @E_out").index[0]
            list_add = list_adds[0:2] + (list_adds[-1],) 
            extra_indeces.append(list_add)
        df_cs['extra_coordinates'] = extra_indeces
        
        MI_df = df1_update1[['c2','c1','eta1','eta2','c_hat',output]].copy()
        if not omit_charge_efficiency:
            MI_df = MI_df[MI_df['eta1'] < 1]
        MI_df = MI_df.set_index(['c2','c1','eta1','eta2','c_hat']) 
        MI_df.sort_values(['c2','c1','eta1','eta2','c_hat'],inplace=True)
        
        if q > 0:
            threshold_0 = df_cs[['c1','c2',output,'extra_coordinates']].groupby(['c2','c1']).quantile(quantiles[q-1],interpolation='nearest')
            
        threshold = df_cs[['c1','c2',output,'extra_coordinates']].groupby(['c2','c1']).quantile(quantile,interpolation='nearest') 
        
        # Get the remaining indices (averages of the configurations within the groups)
        extra_indeces = []
        for i2 in range(len(threshold)):
            if q == 0:
                temp = MI_df.loc[threshold.index[i2]][MI_df.loc[threshold.index[i2]] <= threshold[output].iloc[i2]].dropna()
                temp = temp.reset_index()
                
                if temp.shape[0] > 0:
                    temp[['eta1','eta2']]= temp[['eta1','eta2']]*-1   
                    temp = temp[['eta1','eta2','c_hat']]
                    list_add_avg = tuple(np.abs(temp.mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                            .round(3).values))
                else:
                    temp = MI_df.loc[threshold.index[i2]]
                    temp = temp.reset_index()
                    temp = temp[['eta1','eta2','c_hat']]
                    if temp.shape[0] > 1:
                        list_add_avg = tuple(temp
                                             .mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                             .values.round(3))
                    else:
                        list_add_avg = tuple(temp.values[0].round(3))
                
            else:   
                temp = MI_df.loc[threshold.index[i2]][(MI_df.loc[threshold.index[i2]] > threshold_0[output].iloc[i2]) & (MI_df.loc[threshold.index[i2]] <= threshold[output].iloc[i2])].dropna()
                temp = temp.reset_index()
                
                if temp.shape[0] > 0:
                    temp[['eta1','eta2']]= temp[['eta1','eta2']]*-1   
                    temp = temp[['eta1','eta2','c_hat']]
                    list_add_avg = tuple(np.abs(temp.mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                            .round(3).values))  
                else:
                    temp = MI_df.loc[threshold.index[i2]]
                    temp = temp.reset_index()
                    temp = temp[['eta1','eta2','c_hat']]
                    if temp.shape[0] > 1:
                        list_add_avg = tuple(temp
                                             .mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                             .values.round(3))
                    else:
                        list_add_avg = tuple(temp.values[0].round(3))
                
            extra_indeces.append(list_add_avg)
        threshold['extra_coordinates'] = extra_indeces
        
        df_cs = threshold.copy()
        
        # Get the grouped values: 
        df_cs[output] = (pd.concat([df1_update1[['c1','c2',output]].groupby(['c2','c1']).quantile(qi,interpolation='nearest').copy() for qi in ranges[q]]).groupby(['c2','c1'])
                          .mean()) # <-----------------------------------------WE CAN TAKE MAX/MEAN/MEDIAN
        
        #%% ---------------- ENERGY CAPACITY COST ----------------------------
        df_chat_eta2 = df1_update2.copy()
        
        if not omit_charge_efficiency:
            df_chat_eta2 = df_chat_eta2[df_chat_eta2['eta1'] < 1]
        
        extra_indeces = [] # We are reducing the space from 5D to 2D. Here, we collect descriptors from the omitted 3D space.
        for i1 in range(len(df_chat_eta2)):
            E_out = df_chat_eta2[output].iloc[i1].item()   
            list_adds = MI_df_g2.query("output == @E_out").index[0]
            list_add = (list_adds[0],) + list_adds[2:4] 
            extra_indeces.append(list_add)
        df_chat_eta2['extra_coordinates'] = extra_indeces
        
        MI_df = df1_update2[['eta2','c_hat','c2','c1','eta1',output]].copy()
        if not omit_charge_efficiency:
            MI_df = MI_df[MI_df['eta1'] < 1]
        MI_df = MI_df.set_index(['eta2','c_hat','c2','c1','eta1']) 
        MI_df.sort_values(['eta2','c_hat','c2','c1','eta1'],inplace=True)
        
        if q > 0:
            threshold_0 = df_chat_eta2[['c_hat','eta2',output,'extra_coordinates']].groupby(['eta2','c_hat']).quantile(quantiles[q-1],interpolation='nearest')
            
        threshold = df_chat_eta2[['c_hat','eta2',output,'extra_coordinates']].groupby(['eta2','c_hat']).quantile(quantile,interpolation='nearest') 
        
        # Get the remaining indices (averages of the configurations within the groups)
        extra_indeces = []
        for i2 in range(len(threshold)):
            if q == 0:
                temp = MI_df.loc[threshold.index[i2]][MI_df.loc[threshold.index[i2]] <= threshold[output].iloc[i2]].dropna()
                temp = temp.reset_index()
                if temp.shape[0] > 0:
                    temp['eta1']= temp['eta1']*-1 # if .max() is used below, we need to have efficiency being negative
                    temp = temp[['eta1','c1','c2']]
                    list_add_avg = tuple(np.abs(temp.mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN!
                                                .round(3).values))
                else:
                    temp = MI_df.loc[threshold.index[i2]]
                    temp = temp.reset_index()
                    temp = temp[['eta1','c1','c2']]
                    if temp.shape[0] > 1:
                        list_add_avg = tuple(temp
                                             .mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                             .values.round(3))
                    else:
                        list_add_avg = tuple(temp.values[0].round(3))
        
            else:   
                temp = MI_df.loc[threshold.index[i2]][(MI_df.loc[threshold.index[i2]] > threshold_0[output].iloc[i2]) & (MI_df.loc[threshold.index[i2]] <= threshold[output].iloc[i2])].dropna()
                temp = temp.reset_index()
                if temp.shape[0] > 0:
                    temp['eta1']= temp['eta1']*-1 # if .max() is used below, we need to have efficiency being negative
                    temp = temp[['eta1','c1','c2']]
                    list_add_avg = tuple(np.abs(temp.mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                                .round(3).values))
                else:
                    temp = MI_df.loc[threshold.index[i2]]
                    temp = temp.reset_index()
                    temp = temp[['eta1','c1','c2']]
                    if temp.shape[0] > 1:
                        list_add_avg = tuple(temp
                                             .mean() # <-----------------------------------------WE CAN TAKE MAX OR MEAN! 
                                             .values.round(3))
                    else:
                        list_add_avg = tuple(temp.values[0].round(3))
                        
            extra_indeces.append(list_add_avg)
        threshold['extra_coordinates'] = extra_indeces
        
        df_chat_eta2 = threshold.copy()
        
        # Get the grouped values: 
        df_chat_eta2[output] = (pd.concat([df1_update2[['c_hat','eta2',output]].groupby(['eta2','c_hat']).quantile(qi,interpolation='nearest').copy() for qi in ranges[q]]).groupby(['eta2','c_hat'])
                                 .mean()) # <-----------------------------------------WE CAN TAKE MAX/MEAN/MEDIAN

        if quantile == 1.0:
            df_chat_eta2.loc[(0.25,40.0,),:] = np.nan
            df_chat_eta2.loc[(0.25,30.0),:] = np.nan
            df_chat_eta2.loc[(0.25,20.0),:] = np.nan
            df_chat_eta2.loc[(0.50,40.0,),:] = np.nan
            df_chat_eta2.loc[(0.50,30.0),:] = np.nan
            df_chat_eta2.sort_index(inplace=True)
        else:
            df_chat_eta2.loc[(0.25,10.0),:] = np.nan
            df_chat_eta2.loc[(0.25,20.0),:] = np.nan
            df_chat_eta2.loc[(0.25,30.0),:] = np.nan
            df_chat_eta2.loc[(0.25,40.0,),:] = np.nan
            df_chat_eta2.loc[(0.50,20.0),:] = np.nan
            df_chat_eta2.loc[(0.50,30.0),:] = np.nan
            df_chat_eta2.loc[(0.50,40.0,),:] = np.nan
            df_chat_eta2.loc[(0.95,40.0,),:] = np.nan
            df_chat_eta2.sort_index(inplace=True)

        #%% ------------------ PLOTTING --------------------------------------
        # Capacity cost
        nrows = 4
        ncols = 4
        im = annotate(df_cs, df1_update1, output, nrows, ncols, var1='c2', var2='c1', var_name1=r'$c_c$' + ' [€/kW]', var_name2 = r'$c_d$' + ' [€/kW]', ax=ax[q,0], quantiles=quantiles, q=q, normfactor=normfactor, shading=shading, colormap=cmap)
        
        # Efficiency
        nrows = 3
        ncols = 3 if omit_charge_efficiency else 4 
        im = annotate(df_etas, df1_update1, output, nrows, ncols, var1='eta2',var2='eta1',var_name1=r'$\eta_c$' + ' [-]',var_name2=r'$\eta_d$' + ' [-]', ax=ax[q,1], quantiles=quantiles, q=q, normfactor=normfactor,shading=shading, colormap=cmap)
        
        # Energy capacity cost vs discharge efficiency
        nrows = 3
        ncols = 7
        im = annotate(df_chat_eta2, df1_update2, output, nrows, ncols, var1='eta2',var2='c_hat',var_name1=r'$\hat{c}$' + ' [€/kWh]',var_name2=r'$\eta_d$'+ ' [-]', ax=ax[q,2], quantiles = quantiles, q=q, normfactor=normfactor,shading=shading, colormap=cmap)
        
        q += 1
   #%%    
    cb_ax = fig.add_axes([0.95,0.12,0.02,0.12])
    cb_ax.tick_params(direction='out', length=6, width=2, colors='k',
                      grid_color='k', grid_alpha=1)   
    
    if shading != 'gouraud':
        bounds = np.linspace(0, normfactor, 5)
        cmap4norm = plt.cm.get_cmap(cmap)
        norm = mpl.colors.BoundaryNorm(bounds, cmap4norm.N)
    else:
        norm = mpl.colors.Normalize(vmin=0, vmax=normfactor) 
    
    cb = mpl.colorbar.ColorbarBase(cb_ax,orientation='vertical', cmap= plt.cm.get_cmap(cmap),norm=norm) #,ticks=bounds, boundaries=bounds) #ticks=[0.15,0.25,0.48,0.90])
    cb.ax.tick_params(labelsize=fs)
    
    if output == 'E_cor':
        cb.set_ticks([0,1000,1750])
        cb.ax.set_yticklabels(['$0$', '$1000$', r'$\geq 2000$'])
        cb.set_label(r'$E$' + ' [GWh]', rotation=90,fontsize=fs,labelpad=16)
    fig.text(0.9, 0.79, 
             'Q1', 
             style = 'italic',
             fontsize = 30,
             color = "grey")
    
    fig.text(0.9, 0.59, 
             'Q2', 
             style = 'italic',
             fontsize = 30,
             color = "grey")
    
    fig.text(0.9, 0.38, 
             'Q3', 
             style = 'italic',
             fontsize = 30,
             color = "grey")
    
    fig.text(0.9, 0.18, 
             'Q4', 
             style = 'italic',
             fontsize = 30,
             color = "grey")
    
    fig.savefig('../figures/Matrix_requirements_' + sector + '_' + output + '_' + shading + '_mean.png', dpi=600, bbox_inches='tight')






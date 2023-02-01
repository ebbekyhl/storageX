# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 08:31:10 2022

@author: au485969
"""
import matplotlib.pyplot as plt
from tech_colors import tech_colors
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
plt.close('all')
fs = 18
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True
variable = 'E'
if variable == 'E':
    output = 'E [GWh]'
    output_text = ' Installed energy capacity ' + r'$E$' + ' [TWh]' # ' Installed energy capacity ' + r'$E$' + ' [TWh]'
    factor = 1e3 #1e3 # convert into TWh
elif variable == 'G_charge':
    output = 'G_charge [GW]'
    output_text = ' Installed (charge) power capacity ' + r'$G_1$' + ' [GW]'
    factor = 1
elif variable == 'G_discharge':
    output = 'G_discharge [GW]'
    output_text = ' Installed (discharge) power capacity ' + r'$G_2$' +' [GW]'
    factor = 1
    
path = '../results/simple_sweep/'

#%% Choose which system is considered (power or sector-coupled energy system)
# sector = '-'            # Power system
# sector = 'T-H'        # Land transport (T) + heating (H) 
sector = 'T-H-I-B'    # Land transport (T) + heating (H) + industry, shipping, and aviation (I) + biomass (B)
# 
#%% Read files
wy = 2013
filename = path + 'sspace_eta_c.csv'
df2 = pd.read_csv(filename,index_col=0).T
df2['sector'] = df2['sector'].fillna('-')
df2 = df2.query('sector == @sector')
df2 = df2.drop(columns='sector').astype(float).sort_values(by='eta1 [-]')
df2['E [GWh]'] = df2['E [GWh]']*df2['eta2 [-]']
df2 = df2.T
df2 = df2[df2.loc['weatheryear'][df2.loc['weatheryear'] == wy].index]
df_ref = df2[df2.loc['eta1 [-]'][df2.loc['eta1 [-]'] == 0.5].index]

filename = path + 'sspace_eta_d.csv'
df3 = pd.read_csv(filename,index_col=0).T
df3['sector'] = df3['sector'].fillna('-')
df3 = df3.query('sector == @sector')
df3 = df3.drop(columns='sector').astype(float).sort_values(by='eta2 [-]')
df3['E [GWh]'] = df3['E [GWh]']*df3['eta2 [-]']
df3 = df3.T
df3 = df3[df3.loc['weatheryear'][df3.loc['weatheryear'] == wy].index]

filename = path + 'sspace_c_c.csv' 
df4 = pd.read_csv(filename,index_col=0).T
df4['sector'] = df4['sector'].fillna('-')
df4 = df4.query('sector == @sector')
df4 = df4.drop(columns='sector').astype(float).sort_values(by='c1')
df4['E [GWh]'] = df4['E [GWh]']*df4['eta2 [-]']
df4 = df4.T
df4 = df4[df4.loc['weatheryear'][df4.loc['weatheryear'] == wy].index]

filename = path + 'sspace_c_d.csv'
df5 = pd.read_csv(filename,index_col=0).T
df5['sector'] = df5['sector'].fillna('-')
df5 = df5.query('sector == @sector')
df5 = df5.drop(columns='sector').astype(float).sort_values(by='c2')
df5['E [GWh]'] = df5['E [GWh]']*df5['eta2 [-]']
df5 = df5.T
df5 = df5[df5.loc['weatheryear'][df5.loc['weatheryear'] == wy].index]

filename = path + 'sspace_chat.csv' 
df6 = pd.read_csv(filename,index_col=0).T
df6['sector'] = df6['sector'].fillna('-')
df6 = df6.query('sector == @sector')
df6 = df6.drop(columns='sector').astype(float).sort_values(by='c_hat [EUR/kWh]')
df6['E [GWh]'] = df6['E [GWh]']*df6['eta2 [-]']
df6 = df6.T
df6 = df6[df6.loc['weatheryear'][df6.loc['weatheryear'] == wy].index]

filename = path + 'sspace_tau.csv'
df7 = pd.read_csv(filename,index_col=0).T
df7['sector'] = df7['sector'].fillna('-')
df7 = df7.query('sector == @sector')
df7 = df7.drop(columns='sector').astype(float).sort_values(by='tau [n_days]')
df7['E [GWh]'] = df7['E [GWh]']*df7['eta2 [-]']
df7 = df7.T
df7 = df7[df7.loc['weatheryear'][df7.loc['weatheryear'] == wy].index]


#%% Plotting
cp = ['darkblue','crimson','darkblue','crimson','saddlebrown','saddlebrown']
axes_t = {}
fig1,axes = plt.subplots(2,3,figsize=[14,8],sharey=True)
output1 = output
output2 = 'G_discharge [GW]'
sm1 = axes[0,0].plot(-10*df2.loc['eta1 [-]'],df2.loc[output1]/factor,color='k',marker='.',label='Energy capacity ' + r'$E$')
axes_t[0] = axes[0,0].twinx()
sm2 = axes_t[0].plot(-10*df2.loc['eta1 [-]'],df2.loc[output2],color='k',marker='.',ls='--',label='Power capacity ' + r'$G$')
plt.close('all')
#%%
indexes = [df2.loc['eta1 [-]']*100,
           df7.loc['tau [n_days]'],
           df3.loc['eta2 [-]']*100,
           490 - df4.loc['c1'],
           20 - df6.loc['c_hat [EUR/kWh]'],
           490 - df5.loc['c2']]

fig3,axes = plt.subplots(2,3,figsize=[14,8],sharey=True)
axes = axes.flatten()

xs = ['eta1 [-]','tau [n_days]','eta2 [-]','c1','c_hat [EUR/kWh]','c2']

xlims = [[30,95], # eta1
         [2,57], # tau
         [30,95], # eta2
         [0,455], # c1
         [0,20], # chat
         [0,455]] # c2] 

xticks = [[30,50,70,95], # eta1
          [2,10,18,30,45,57], # tau
          [30,50,70,95], # eta2
          [0,140,300,390,455], # c1
          [0,5,10,15,20], # chat
          [0,140,300,390,455]] # c2

xticklabels = [[30,50,70,95], # eta1
               [2,10,18,30,45,57], # tau
               [30,50,70,95], # eta 2
               [490,350,490-300,490-390,35], #c1
               [20,15,10,5,0], # chat
               [490,350,490-300,490-390,35]] #c2

xlabels = ['Charge efficiency ' + r'$\eta_c$' + ' [%]',
           'Self-discharge time ' + r'$\tau_{SD}$' + ' [days]',
           'Discharge efficiency ' +r'$\eta_d$' + ' [%]',
           'Cost of charging \n power capacity ' + r'$c_c$' + ' [€/kW]',
           'Cost of energy capacity \n' + r'$\hat{c}$' + ' [€/kWh]',
           'Cost of discharging \n power capacity ' + r'$c_d$' + ' [€/kW]'
           ]

dfs = [df2,df7,df3,df4,df6,df5]
i = 0
for ax in axes:
    df_i = dfs[i]
    
    to_plot = df_i.index[19:25]
    df_i_x = indexes[i]
    
    sm4 = ax.plot(df_i_x,df_i.loc['E [GWh]']/factor,marker='.',color='lightgrey')
    sm5 = ax.plot(df_i_x,df_i.loc[to_plot[1]]/factor,marker='.',color=tech_colors('battery'))
    sm6 = ax.plot(df_i_x,df_i.loc[to_plot[3]]/factor,marker='.',color=tech_colors('hydro'))

    ax.set_xlim(xlims[i])
    ax.set_xticks(xticks[i])
    ax.set_xticklabels(xticklabels[i])
    ax.set_ylim([0,6])
    if i == 0 or i == 3:
        ax.set_ylabel('Energy capacity [TWh]')
    ax.set_xlabel(xlabels[i])
    ax.grid(axis='y')
    ax_t = ax.twinx()
    ax_t.plot(df_i_x,df_i.loc['G_discharge [GW]'],marker='.',ls='--',color='lightgrey')
    ax_t.plot(df_i_x,df_i.loc[to_plot[0]],marker='.',ls='--',color=tech_colors('battery'))
    ax_t.plot(df_i_x,df_i.loc[to_plot[2]],marker='.',ls='--',color=tech_colors('hydro'))
    ax_t.set_ylim([0,200])
    
    if i == 2:
        print('Battery: ','-',round(max(df_i.loc[to_plot[0]]) - min(df_i.loc[to_plot[0]]),2))
        print('Storage-X: ','+',round(max(df_i.loc['G_discharge [GW]']) - min(df_i.loc['G_discharge [GW]']),2))
        
    if i == 2 or i == 5:
        ax_t.set_ylabel('Power capacity [GW]')    
    
    i += 1

fig3.tight_layout() 
fig3.legend(sm1+sm2+sm4+sm5+sm6,['Energy capacity','Power capacity','Storage-X','Battery','PHS'], bbox_to_anchor=(0.2, -0.1), loc=3,
            ncol=3,frameon=True,prop={'size': fs},borderaxespad=0)
fig3.savefig('../figures/Storage_single_sweep_battery_PHS_hydrogen' + sector + '.pdf', transparent=True,
            bbox_inches="tight")
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
sector = '-'            # Power system
# sector = 'T-H'        # Land transport (T) + heating (H) 
# sector = 'T-H-I-B'    # Land transport (T) + heating (H) + industry, shipping, and aviation (I) + biomass (B)

#%% Read files
filename = path + 'sspace_eta_c.csv'
df2 = pd.read_csv(filename,index_col=0).T
df2['sector'] = df2['sector'].fillna('-')
df2 = df2.query('sector == @sector')
df2 = df2.drop(columns='sector').astype(float).sort_values(by='eta1 [-]').T
df_ref = df2[df2.loc['eta1 [-]'][df2.loc['eta1 [-]'] == 0.5].index]

filename = path + 'sspace_eta_d.csv'
df3 = pd.read_csv(filename,index_col=0).T
df3['sector'] = df3['sector'].fillna('-')
df3 = df3.query('sector == @sector')
df3 = df3.drop(columns='sector').astype(float).sort_values(by='eta2 [-]').T

filename = path + 'sspace_c_c.csv' 
df4 = pd.read_csv(filename,index_col=0).T
df4['sector'] = df4['sector'].fillna('-')
df4 = df4.query('sector == @sector')
df4 = df4.drop(columns='sector').astype(float).sort_values(by='c1').T

filename = path + 'sspace_c_d.csv'
df5 = pd.read_csv(filename,index_col=0).T
df5['sector'] = df5['sector'].fillna('-')
df5 = df5.query('sector == @sector')
df5 = df5.drop(columns='sector').astype(float).sort_values(by='c2').T

filename = path + 'sspace_chat.csv' 
df6 = pd.read_csv(filename,index_col=0).T
df6['sector'] = df6['sector'].fillna('-')
df6 = df6.query('sector == @sector')
df6 = df6.drop(columns='sector').astype(float).sort_values(by='c_hat [EUR/kWh]').T

filename = path + 'sspace_tau.csv'
df7 = pd.read_csv(filename,index_col=0).T
df7['sector'] = df7['sector'].fillna('-')
df7 = df7.query('sector == @sector')
df7 = df7.drop(columns='sector').astype(float).sort_values(by='tau [n_days]').T

#%% Plotting
cp = ['darkblue','crimson','darkblue','crimson','saddlebrown','saddlebrown']

axes_t = {}

fig1,axes = plt.subplots(2,3,figsize=[14,8],sharey=True)
output1 = output
output2 = 'G_discharge [GW]'
axes[0,0].set_ylim([0,3.2])
fs = 18
axes[0,0].plot(df2.loc['eta1 [-]']*100,df2.loc[output1]/factor,color=cp[0],marker='.')
sm1 = axes[0,0].plot(-10*df2.loc['eta1 [-]'],df2.loc[output1]/factor,color='k',marker='.',label='Energy capacity ' + r'$E$')
axes[0,0].scatter(50,df2[df2.loc['eta1 [-]'][df2.loc['eta1 [-]'] == 0.5].index].loc[output1]/factor,color='k',marker='X',s=100,zorder=10)
axes[0,0].grid(axis='y')

axes_t[0] = axes[0,0].twinx()
axes_t[0].set_zorder(-1)
axes[0,0].patch.set_alpha(0)
sm2 = axes_t[0].plot(-10*df2.loc['eta1 [-]'],df2.loc[output2],color='k',marker='.',ls='--',label='Power capacity ' + r'$G$')
axes_t[0].plot(df2.loc['eta1 [-]']*100,df2.loc[output2],color=cp[0],marker='.',ls='--')
axes_t[0].set_ylim([0,40])
axes_t[0].set_yticks([0,10,20,30,40])
axes_t[0].set_yticklabels([])
axes[0,0].set_ylabel(r'$E^x$' + ' [TWh]',fontsize=fs)
axes[0,0].set_xlim([30,95])
axes[0,0].set_xlabel('Charge efficiency ' + r'$\eta_c$' + ' [%]',fontsize=fs)
axes[0,0].set_xticks([30,50,70,95])
axes[0,0].tick_params(axis='both', which='major', labelsize=fs)

axes[1,0].plot(490-df4.loc['c1'],df4.loc[output1]/factor,color=cp[2],marker='.')
axes[1,0].set_xlim([0,455])
axes[1,0].set_xlabel('Cost of charging \n power capacity ' + r'$c_c$' + ' [€/kW]',fontsize=fs)
axes[1,0].set_ylabel(r'$E^x$' + ' [TWh]',fontsize=fs)
axes[1,0].scatter(490-350,df4[df4.loc['c1'][df4.loc['c1'] == 350].index].loc[output1]/factor,color='k',marker='X',s=100,zorder=10)
axes[1,0].grid(axis='y')

axes_t[2] = axes[1,0].twinx()
axes_t[2].set_zorder(-1)
axes[1,0].patch.set_alpha(0)
axes_t[2].plot(490-df4.loc['c1'],df4.loc[output2],color=cp[2],marker='.',ls='--')

axes_t[2].set_ylim([0,40])
axes_t[2].set_yticks([0,10,20,30,40])
axes_t[2].set_yticklabels([])
axes[1,0].set_xticks([0,140,300,390,455])
axes[1,0].set_xticklabels([490,350,490-300,490-390,35])
axes[1,0].tick_params(axis='both', which='major', labelsize=fs)

axes[1,1].plot(20-df6.loc['c_hat [EUR/kWh]'],df6.loc[output1]/factor,color=cp[4],marker='.')
axes[1,1].set_xlabel('Cost of energy capacity \n' + r'$\hat{c}$' + ' [€/kWh]',fontsize=fs)
axes[1,1].set_xlim([0,20])
axes[1,1].grid(axis='y')

sm3 = axes[1,1].scatter(20 - 3,df6[df6.loc['c_hat [EUR/kWh]'][df6.loc['c_hat [EUR/kWh]'] == 3].index].loc[output1]/factor,color='k',marker='X',s=100,zorder=10,label='Reference storage-X parameters')
axes_t[4] = axes[1,1].twinx()
axes_t[4].set_zorder(-1)
axes[1,1].patch.set_alpha(0)
axes_t[4].plot(20-df6.loc['c_hat [EUR/kWh]'],df6.loc[output2],color=cp[4],marker='.',ls='--')
axes_t[4].set_ylim([0,40])
axes_t[4].set_yticks([0,10,20,30,40])
axes_t[4].set_yticklabels([])
axes[1,1].set_xticks([0,5,10,15,18,20])
axes[1,1].set_xticklabels([20,15,10,5,2,0])
axes[1,1].tick_params(axis='both', which='major', labelsize=fs)

axes[0,1].plot(df7.loc['tau [n_days]'],df7.loc[output]/factor,color=cp[5],marker='.')
axes[0,1].set_xlabel('Self-discharge time ' + r'$\tau_{SD}$' + ' [days]',fontsize=fs)
axes[0,1].set_xlim([0,57])
axes[0,1].scatter(30,df7[df7.loc['tau [n_days]'][df7.loc['tau [n_days]'] == 30].index].loc[output1]/factor,color='k',marker='X',s=100,zorder=10)
axes_t[5] = axes[0,1].twinx()
axes_t[5].set_zorder(-1)
axes[0,1].patch.set_alpha(0)
axes_t[5].plot(df7.loc['tau [n_days]'],df7.loc[output2],color=cp[5],marker='.',ls='--')
axes_t[5].set_ylim([0,40])
axes_t[5].set_yticklabels([])
axes[0,1].set_xticks([0,10,18,30,45,57])
axes[0,1].tick_params(axis='both', which='major', labelsize=fs)
axes[0,1].grid(axis='y')

axes[0,2].plot(df3.loc['eta2 [-]']*100,df3.loc[output1]/factor,color=cp[1],marker='.')
axes[0,2].scatter(50,df3[df3.loc['eta2 [-]'][df3.loc['eta2 [-]'] == 0.5].index].loc[output1]/factor,color='k',marker='X',s=100,zorder=10)
axes_t[1] = axes[0,2].twinx()
axes_t[1].set_zorder(-1)
axes[0,2].patch.set_alpha(0)
axes[0,2].grid(axis='y')
axes_t[1].plot(df3.loc['eta2 [-]']*100,df3.loc[output2],color=cp[1],marker='.',ls='--')
axes_t[1].set_ylim([0,40])
axes_t[1].set_yticks([0,10,20,30,40])
axes_t[1].set_yticklabels([0,10,20,30,40],fontsize=fs)
axes_t[1].set_ylabel(r'$G_d^x$'+ ' [GW' + r'$_e$' + ']',fontsize=fs)

axes[0,2].set_xlabel('Discharge efficiency ' +r'$\eta_d$' + ' [%]',fontsize=fs)
axes[0,2].set_xlim([30,95])
axes[0,2].set_xticks([30,50,70,95])
axes[0,2].tick_params(axis='both', which='major', labelsize=fs)

axes[1,2].plot(490-df5.loc['c2'],df5.loc[output1]/factor,color=cp[3],marker='.')
axes[1,2].set_xlim([0,455])
axes[1,2].set_xlabel('Cost of discharging \n power capacity ' + r'$c_d$' + ' [€/kW]',fontsize=fs)
axes[1,2].scatter(490-350,df5[df5.loc['c2'][df5.loc['c2'] == 350].index].loc[output1]/factor,color='k',marker='X',s=100,zorder=10)
axes_t[3] = axes[1,2].twinx()
axes_t[3].set_zorder(-1)
axes[1,2].patch.set_alpha(0)
axes_t[3].plot(490-df5.loc['c2'],df5.loc[output2],color=cp[3],marker='.',ls='--')
axes_t[3].set_ylim([0,40])
axes_t[3].set_yticks([0,10,20,30,40])
axes_t[3].set_yticklabels([0,10,20,30,40],fontsize=fs)
axes[1,2].set_xticks([0,140,300,390,455])
axes[1,2].set_xticklabels([490,350,490-300,490-390,35])
axes[1,2].tick_params(axis='both', which='major', labelsize=fs)
axes_t[3].set_ylabel(r'$G_d^x$'+ ' [GW' + r'$_e$' + ']',fontsize=fs)
axes[1,2].grid(axis='y')

fig1.legend(sm1+sm2+[sm3],['Energy capacity','Power capacity','Reference storage-X parameters'], bbox_to_anchor=(0.15, -0.05), loc=3,
           ncol=4,frameon=True,prop={'size': fs},borderaxespad=0)
fig1.tight_layout() 
fig1.savefig('../figures/Storage_parameter_hierarchi_' + variable + '_absolute_values_subplot.pdf', transparent=True,
            bbox_inches="tight")
#%%
fig2,axes = plt.subplots(2,3,figsize=[14,8],sharey=True)

fs = 18
axes[0,0].plot(df2.loc['eta1 [-]'],df2.loc['E [GWh]']/df2.loc['G_discharge [GW]'],color=cp[0],label=r'$\eta_c$',marker='.')
axes[0,0].plot(0.5,df2[df2.loc['eta1 [-]'][df2.loc['eta1 [-]'] == 0.5].index].loc['E [GWh]']/df2[df2.loc['eta1 [-]'][df2.loc['eta1 [-]'] == 0.5].index].loc['G_discharge [GW]'],color='k',marker='X',markersize=10)
axes[0,0].set_ylabel('Full load hours',fontsize=fs)
axes[0,0].set_xlim([0.3,0.95])
axes[0,0].set_xlabel(r'$\eta_c$' + ' [-]',fontsize=fs)
axes[0,0].tick_params(axis='both', which='major', labelsize=fs)

axes[0,2].plot(df3.loc['eta2 [-]'],df3.loc['E [GWh]']/df3.loc['G_discharge [GW]'],color=cp[1],label=r'$\eta_d$',marker='.')
axes[0,2].plot(0.5,df3[df3.loc['eta2 [-]'][df3.loc['eta2 [-]'] == 0.5].index].loc['E [GWh]']/df3[df3.loc['eta2 [-]'][df3.loc['eta2 [-]'] == 0.5].index].loc['G_discharge [GW]'],color='k',marker='X',markersize=10)
axes[0,2].set_xlabel(r'$\eta_d$' + ' [-]',fontsize=fs)
axes[0,2].set_xlim([0.3,0.95])
axes[0,2].tick_params(axis='both', which='major', labelsize=fs)

axes[1,1].plot(20-df6.loc['c_hat [EUR/kWh]'],df6.loc['E [GWh]']/df6.loc['G_discharge [GW]'],color=cp[4],label=r'$\hat{c}$',marker='.')
axes[1,1].set_xlabel(r'$\hat{c}$' + ' [€/kWh]',fontsize=fs)
axes[1,1].set_xlim([0,20])
axes[1,1].plot(20 - 3,df6.T[df6.loc['c_hat [EUR/kWh]'] == 3].T.loc['E [GWh]']/df6.T[df6.loc['c_hat [EUR/kWh]'] == 3].T.loc['G_discharge [GW]'],color='k',marker='X',markersize=10,label='Reference storage-X parameters')
axes[1,1].set_xticks([0,5,10,15,20])
axes[1,1].set_xticklabels([20,15,10,5,0])
axes[1,1].tick_params(axis='both', which='major', labelsize=fs)

axes[0,1].plot(df7.loc['tau [n_days]'],df7.loc['E [GWh]']/df7.loc['G_discharge [GW]'],color=cp[5],marker='.')
axes[0,1].set_xlabel(r'$\tau_{SD}$' + ' [days]',fontsize=fs)
axes[0,1].set_xlim([0,57])
axes[0,1].plot(30,df7.T[df7.loc['tau [n_days]'] == 30].T.loc['E [GWh]']/df7.T[df7.loc['tau [n_days]'] == 30].T.loc['G_discharge [GW]'],color='k',marker='X',markersize=10)
axes[0,1].tick_params(axis='both', which='major', labelsize=fs)

axes[1,0].plot(490-df4.loc['c1'],df4.loc['E [GWh]']/df4.loc['G_discharge [GW]'],color=cp[2],label=r'$c_c$',marker='.')
axes[1,0].set_xlim([0,455])
axes[1,0].set_xlabel(r'$c_c$' + ' [€/kW]',fontsize=fs)
axes[1,0].set_ylabel('Full load hours',fontsize=fs)
axes[1,0].plot(490-350,df4[df4.loc['c1'][df4.loc['c1'] == 350].index].loc['E [GWh]']/df4[df4.loc['c1'][df4.loc['c1'] == 350].index].loc['G_discharge [GW]'],color='k',marker='X',markersize=10)
axes[1,0].set_xticks([0,100,200,300,400,455])
axes[1,0].set_xticklabels([490,490-100,490-200,490-300,490-400,35])
axes[1,0].tick_params(axis='both', which='major', labelsize=fs)

axes[1,2].plot(490-df5.loc['c2'],df5.loc['E [GWh]']/df5.loc['G_discharge [GW]'],color=cp[3],label=r'$c_d$',marker='.')
axes[1,2].set_xlim([0,455])
axes[1,2].set_xlabel(r'$c_d$' + ' [€/kW]',fontsize=fs)
axes[1,2].plot(490-350,df5[df5.loc['c2'][df5.loc['c2'] == 350].index].loc['E [GWh]']/df5[df5.loc['c2'][df5.loc['c2'] == 350].index].loc['G_discharge [GW]'],color='k',marker='X',markersize=10)
axes[1,2].set_xticks([0,100,200,300,400,455])
axes[1,2].set_xticklabels([490,490-100,490-200,490-300,490-400,35])
axes[1,2].tick_params(axis='both', which='major', labelsize=fs)

for ax_ij in axes:
    for ax_i in ax_ij:   
        ax_i.axhline(24,color='darkgrey',ls='--',lw=1)
        ax_i.axhline(24*4,color='darkgrey',ls='--',lw=1)
        ax_i.axhline(168,color='darkgrey',ls='--',lw=1)
        ax_i.axhline(2*168,color='darkgrey',ls='--',lw=1)
        if ax_i == axes[-1][-1]:
            ax_i.text(ax_i.get_xlim()[0] + (ax_i.get_xlim()[1] - ax_i.get_xlim()[0])*0.5,24+6,'1 day',fontsize=fs,color='darkgrey')
            ax_i.text(ax_i.get_xlim()[0] + (ax_i.get_xlim()[1] - ax_i.get_xlim()[0])*0.5,24*4+6,'4 days',fontsize=fs,color='darkgrey')
            ax_i.text(ax_i.get_xlim()[0] + (ax_i.get_xlim()[1] - ax_i.get_xlim()[0])*0.5,168+6,'1 week',fontsize=fs,color='darkgrey')
            ax_i.text(ax_i.get_xlim()[0] + (ax_i.get_xlim()[1] - ax_i.get_xlim()[0])*0.5,2*168+6,'2 weeks',fontsize=fs,color='darkgrey')

fig2.legend(bbox_to_anchor=(0.3, -0.09), loc=3,
           ncol=4,frameon=True,prop={'size': fs},borderaxespad=0)
fig2.tight_layout() 

fig2.savefig('../figures/Storage_parameter_hierarchi_full_load_hours.pdf', transparent=True,
            bbox_inches="tight")
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
    ax.set_ylim([0,4])
    if i == 0 or i == 3:
        ax.set_ylabel('E [TWh]')
    
    ax.set_xlabel(xlabels[i])
    
    ax_t = ax.twinx()
    ax_t.plot(df_i_x,df_i.loc['G_discharge [GW]'],marker='.',ls='--',color='lightgrey')
    ax_t.plot(df_i_x,df_i.loc[to_plot[0]],marker='.',ls='--',color=tech_colors('battery'))
    ax_t.plot(df_i_x,df_i.loc[to_plot[2]],marker='.',ls='--',color=tech_colors('hydro'))
    ax_t.set_ylim([0,60])
    
    if i == 2 or i == 5:
        ax_t.set_ylabel('G [GW]')    
    
    i += 1

fig3.tight_layout() 
fig3.legend(sm1+sm2+sm4+sm5+sm6,['Energy capacity','Power capacity','Storage-X','Battery','PHS'], bbox_to_anchor=(0.2, -0.1), loc=3,
            ncol=3,frameon=True,prop={'size': fs},borderaxespad=0)
fig3.savefig('../figures/Storage_single_sweep_battery_PHS_hydrogen.pdf', transparent=True,
            bbox_inches="tight")
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:25:34 2022

@author: au485969
"""

def plot_single_sweep(fig,axes,sector,output,output_text,factor,fs,lw1,ms1,dfs):
    color_sec = {'-':'green',
                 'T-H':'blue',
                 'T-H-I-B':'orange'}
    
    zorder_sec = {'-':3,
                 'T-H':2,
                 'T-H-I-B':1}
    
    df2 = dfs[0]
    df3 = dfs[1]
    df4 = dfs[2]
    df5 = dfs[3]
    df6 = dfs[4]
    df7 = dfs[5]
    
    sm1 = axes[0,0].plot(df2.loc['eta1 [-]']*100,df2.loc[output]/factor,color=color_sec[sector],zorder=zorder_sec[sector],marker='.',lw=lw1,ms=ms1)
    # axes[0,0].scatter(50,df2[df2.loc['eta1 [-]'][df2.loc['eta1 [-]'] == 0.5].index].loc[output]/factor,color='k',marker='X',s=100,zorder=10)
    axes[0,0].grid(axis='y')
    
    axes[0,0].patch.set_alpha(0)
    axes[0,0].set_ylabel(output_text,fontsize=fs)
    axes[0,0].set_xlim([30,95])
    axes[0,0].set_xlabel('Charge efficiency ' + r'$\eta_c$' + ' [%]',fontsize=fs)
    axes[0,0].set_xticks([30,50,70,95])
    axes[0,0].tick_params(axis='both', which='major', labelsize=fs)
    
    axes[1,0].plot(490-df4.loc['c1'],df4.loc[output]/factor,color=color_sec[sector],marker='.',zorder=zorder_sec[sector],lw=lw1,ms=ms1,label='2013')
    axes[1,0].set_xlim([0,455])
    axes[1,0].set_xlabel('Cost of charging \n power capacity ' + r'$c_c$' + ' [€/kW]',fontsize=fs)
    axes[1,0].set_ylabel(output_text,fontsize=fs)
    # axes[1,0].scatter(490-350,df4[df4.loc['c1'][df4.loc['c1'] == 350].index].loc[output]/factor,color='k',marker='X',s=100,zorder=10)
    axes[1,0].grid(axis='y')
    
    axes[1,0].patch.set_alpha(0)
    axes[1,0].set_xticks([0,140,300,390,455])
    axes[1,0].set_xticklabels([490,350,490-300,490-390,35])
    axes[1,0].tick_params(axis='both', which='major', labelsize=fs)
    
    axes[1,1].plot(20-df6.loc['c_hat [EUR/kWh]'],df6.loc[output]/factor,color=color_sec[sector],zorder=zorder_sec[sector],marker='.',lw=lw1,ms=ms1)
    axes[1,1].set_xlabel('Cost of energy capacity \n' + r'$\hat{c}$' + ' [€/kWh]',fontsize=fs)
    axes[1,1].set_xlim([0,20])
    axes[1,1].grid(axis='y')
    
    # sm2 = axes[1,1].scatter(20 - 3,df6[df6.loc['c_hat [EUR/kWh]'][df6.loc['c_hat [EUR/kWh]'] == 3].index].loc[output]/factor,color='k',marker='X',s=100,zorder=10,label='Reference storage-X parameters')
    axes[1,1].patch.set_alpha(0)
    axes[1,1].set_xticks([0,5,10,15,18,20])
    axes[1,1].set_xticklabels([20,15,10,5,2,0])
    axes[1,1].tick_params(axis='both', which='major', labelsize=fs)
    
    axes[0,1].plot(df7.loc['tau [n_days]'],df7.loc[output]/factor,zorder=zorder_sec[sector],color=color_sec[sector],marker='.',lw=lw1,ms=ms1)
    axes[0,1].set_xlabel('Self-discharge time ' + r'$\tau_{SD}$' + ' [days]',fontsize=fs)
    axes[0,1].set_xlim([0,57])
    # axes[0,1].scatter(30,df7[df7.loc['tau [n_days]'][df7.loc['tau [n_days]'] == 30].index].loc[output]/factor,color='k',marker='X',s=100,zorder=10)
    axes[0,1].patch.set_alpha(0)
    axes[0,1].set_xticks([0,10,18,30,45,57])
    axes[0,1].tick_params(axis='both', which='major', labelsize=fs)
    axes[0,1].grid(axis='y')
    
    axes[0,2].plot(df3.loc['eta2 [-]']*100,df3.loc[output]/factor,zorder=zorder_sec[sector],color=color_sec[sector],marker='.',lw=lw1,ms=ms1)
    # axes[0,2].scatter(50,df3[df3.loc['eta2 [-]'][df3.loc['eta2 [-]'] == 0.5].index].loc[output]/factor,color='k',marker='X',s=100,zorder=10)
    axes[0,2].patch.set_alpha(0)
    axes[0,2].grid(axis='y')
    
    axes[0,2].set_xlabel('Discharge efficiency ' +r'$\eta_d$' + ' [%]',fontsize=fs)
    axes[0,2].set_xlim([30,95])
    axes[0,2].set_xticks([30,50,70,95])
    axes[0,2].tick_params(axis='both', which='major', labelsize=fs)
    
    axes[1,2].plot(490-df5.loc['c2'],df5.loc[output]/factor,zorder=zorder_sec[sector],color=color_sec[sector],marker='.',lw=lw1,ms=ms1)
    axes[1,2].set_xlim([0,455])
    axes[1,2].set_xlabel('Cost of discharging \n power capacity ' + r'$c_d$' + ' [€/kW]',fontsize=fs)
    # axes[1,2].scatter(490-350,df5[df5.loc['c2'][df5.loc['c2'] == 350].index].loc[output]/factor,color='k',marker='X',s=100,zorder=10)
    axes[1,2].patch.set_alpha(0)
    axes[1,2].set_xticks([0,140,300,390,455])
    axes[1,2].set_xticklabels([490,350,490-300,490-390,35])
    axes[1,2].tick_params(axis='both', which='major', labelsize=fs)
    axes[1,2].grid(axis='y')
    
    return sm1
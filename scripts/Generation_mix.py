# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:36:37 2022

@author: au485969
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:27:51 2022

@author: au485969
"""
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

def plot_generation_mix(tech_colors,sector):

    # Read result configurations
    filename = 'results/sspace_w_sectorcoupling_merged.csv'
    df = pd.read_csv(filename,index_col=0).fillna(0)
    df_T = df.T
    df_T = df_T.query('sector == @sector').drop(columns='sector').astype(float)
    df_T = df_T.sort_values(by='c_hat [EUR/kWh]')
    df_T['E [TWh]'] = df_T['E [GWh]']*1e-3 # convert GWh to TWh
    df_T = df_T.loc[df_T['eta1 [-]'][df_T['eta1 [-]'] <= 1].index]
    df_T['E_cor [TWh]'] = df_T['E [TWh]']*df_T['eta2 [-]']
    # df_T['G_discharge [GW]'] = df_T['G_discharge [GW]']*df_T['eta2 [-]'] # Discharge efficiency is already multiplied to the discharge capacity 
    df = df_T.reset_index(drop=True).T
    #%%
    lc_95_quantile = df.loc['load_coverage [%]'].quantile(0.95)
    df = df[df.loc['load_coverage [%]'][df.loc['load_coverage [%]'] <= lc_95_quantile].index]
    # df = df[df.loc['E_cor [TWh]'][df.loc['E_cor [TWh]'] <= 15].index]
    rolling_interval = 20
    
    # Threshold that storage configurations should fulfill to be included in the analysis
    threshold_E = 0# 2000 # GWh
    #%% Capacity factors
    CFs_i = df.loc[['E_cor [TWh]','load_coverage [%]',
                  'gas_OCGT_gen [MWh]','cap_gas_OCGT [GW]',
                  'gas_CCGT_gen [MWh]','cap_gas_CCGT [GW]',
                  'gas_CHP_CC_gen [MWh]','cap_gas_CHP_CC [GW]',
                  'gas_CHP_gen [MWh]','cap_gas_CHP [GW]',
                  'biomass_CHP_CC_gen [MWh]','cap_biomass_CHP_CC [GW]',
                  'biomass_CHP_gen [MWh]','cap_biomass_CHP [GW]',
                   'nuclear_gen [MWh]','cap_nuclear [GW]',
                   'coal_gen [MWh]','cap_coal [GW]',
                  ]].T
    # index = 'E_cor [TWh]'
    index = 'load_coverage [%]'
    CFs_i.set_index(index,inplace=True)
    
    col_dic = {'gas_OCGT_gen [MWh]':'OCGT',
              'gas_CCGT_gen [MWh]':'CCGT',
              'gas_CHP_CC_gen [MWh]':'gas CHP CC',
              'gas_CHP_gen [MWh]':'gas CHP',
              'biomass_CHP_CC_gen [MWh]':'biomass CHP CC',
              'biomass_CHP_gen [MWh]':'biomass CHP',
               'nuclear_gen [MWh]':'nuclear',
               'coal_gen [MWh]':'coal'
              }
    
    CFs = pd.DataFrame(index=CFs_i.index)
    for i in np.arange(len(col_dic))*2+1:
        if ((CFs_i[CFs_i.columns[i]]/1000).max() > 1): #and (CFs_i[CFs_i.columns[i+1]] > 0).all():
            CFs[col_dic[CFs_i.columns[i]]] = (CFs_i[CFs_i.columns[i]]/1000)/CFs_i[CFs_i.columns[i+1]]
            # limit = CFs[col_dic[CFs_i.columns[i]]].quantile(0.99)
            # col = CFs.columns[i]
            # CFs = CFs.query('@col <= @limit')
    # print(CFs['OCGT'].describe())
    
    #%% Generation mix
    
    # Variable used for sorting the configurations
    # x_vars = ['load_coverage [%]']
    x_var = 'E_cor [TWh]'
    # x_vars = ['G_discharge [GW]']
    
    # Total generation
    
    gen_tot = df.loc[['offwind_gen [MWh]',
                      'onwind_gen [MWh]',
                      'solar_gen [MWh]',
                      'hydro_gen [MWh]',
                      'gas_OCGT_gen [MWh]',
                      'gas_CCGT_gen [MWh]',
                      'gas_CHP_CC_gen [MWh]',
                      'gas_CHP_gen [MWh]',
                      'biomass_CHP_CC_gen [MWh]',
                      'biomass_CHP_gen [MWh]',
                       'nuclear_gen [MWh]',
                       'coal_gen [MWh]'
                      ]].T.sum(axis=1)
    
    # Storage configurations
    # x_low = df[df.loc['E [GWh]'].idxmin()]
    x = df[df.loc['E [GWh]'][df.loc['E [GWh]'] > threshold_E].index]
    
    lw_raw = 0.2
    
    # Plotting
    # for x_var in x_vars:
    # fig = plt.figure(figsize=(12, 14))
    fig = plt.figure(figsize=(8, 14))
    nrows = 4 # 6
    ncols = 5
    gs = gridspec.GridSpec(nrows, ncols)
    # gs.update(wspace=0)
    gs.update(hspace=0.5)
    # cp = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # ax = plt.subplot(gs[3:,:])
    ax1 = plt.subplot(gs[0:2,:]) #,sharex=ax)
    ax2 = plt.subplot(gs[2:4,:])
    # ax3 = plt.subplot(gs[2:3,:])
    # fig, ax = plt.subplots(figsize=[8,8])
    x_vardic = {'E [GWh]':'E','E_cor [TWh]':'E_cor [TWh]','G_discharge [GW]':'G_d','G_charge [GW]':'G_c','load_coverage [%]':'load_coverage','c_hat [EUR/kWh]':'chat'}
    x_vardic1 = {'E [GWh]':'E [GWh]','E_cor [TWh]':'E_cor [TWh]','G_discharge [GW]':'G_d [GW]','G_charge [GW]':'G_c [GW]','load_coverage [%]':'load coverage ' + r'$LC$' + ' [%]','c_hat [EUR/kWh]':'chat'}
    
    df_plot = pd.DataFrame()
    df_plot['tot'] = gen_tot.loc[x.T.index]
    df_plot['OCGT'] = x.loc['gas_OCGT_gen [MWh]']#/df_plot['tot']*100
    df_plot['CCGT'] = x.loc['gas_CCGT_gen [MWh]']
    df_plot['biomass CHP CC'] = x.loc['biomass_CHP_CC_gen [MWh]']
    df_plot['biomass CHP'] = x.loc['biomass_CHP_gen [MWh]']
    df_plot['coal'] = x.loc['coal_gen [MWh]']
    df_plot['gas CHP CC'] = x.loc['gas_CHP_CC_gen [MWh]']
    df_plot['gas CHP'] = x.loc['gas_CHP_gen [MWh]']
    # df_plot['gas'] = x.loc['gas_OCGT_gen [MWh]'] + x.loc['gas_CCGT_gen [MWh]']
    df_plot['OCGT'] = x.loc['gas_OCGT_gen [MWh]']
    df_plot['CCGT'] = + x.loc['gas_CCGT_gen [MWh]']
    df_plot['nuclear'] = x.loc['nuclear_gen [MWh]']
    df_plot['hydro'] = x.loc['hydro_gen [MWh]']#/df_plot['tot']*100
    df_plot['wind'] = x.loc['onwind_gen [MWh]'] + x.loc['offwind_gen [MWh]']
    df_plot['solar'] = x.loc['solar_gen [MWh]']
    df_plot['Ecap'] = x.loc['E [GWh]']
    df_plot['Gccap'] = x.loc['G_charge [GW]']
    df_plot['Gdcap'] = x.loc['G_discharge [GW]']
    df_plot['chat'] = x.loc['c_hat [EUR/kWh]']
    df_plot['cc'] = x.loc['c1']
    df_plot['cd'] = x.loc['c2']
    df_plot['etac'] = x.loc['eta1 [-]']
    df_plot['etad'] = x.loc['eta2 [-]']
    df_plot['syscost'] = x.loc['c_sys [bEUR]']
    df_plot['bat_lc'] = x.loc['battery_load_coverage [%]']
    
    df_plot['x'] = x.loc[x_var]
    
    df_plot.sort_values(by = 'x',inplace=True)
    df_plot.set_index('x',inplace=True)
    # tot = df_plot['tot']
    # Ecap =  df_plot['Ecap']
    # Gccap =  df_plot['Gccap']
    # Gdcap =  df_plot['Gdcap']
    # etac =  df_plot['etac']
    # etad =  df_plot['etad']
    # cc =  df_plot['cc']
    # cd =  df_plot['cd']
    # chat = df_plot['chat']
    # syscost = df_plot['syscost']
    # bat_lc = df_plot['bat_lc']
    df_plot.drop(columns=['Ecap','Gccap','Gdcap','etac','etad','cc','cd','chat','syscost','bat_lc'],inplace=True)
    #%%
    if df_plot['coal'].max()/(df_plot.sum(axis=1).min()) < 1e-3:
        df_plot.drop(columns=['coal'],inplace=True)
        
    if df_plot['nuclear'].max()/(df_plot.sum(axis=1).min()) < 1e-3:
        df_plot.drop(columns=['nuclear'],inplace=True)
    
    if df_plot['gas CHP CC'].max()/(df_plot.sum(axis=1).min()) < 1e-3:
        df_plot.drop(columns=['gas CHP CC'],inplace=True)
    
    if df_plot['gas CHP'].max()/(df_plot.sum(axis=1).min()) < 1e-3:
        df_plot.drop(columns=['gas CHP'],inplace=True)
        
    if df_plot['biomass CHP CC'].max()/(df_plot.sum(axis=1).min()) < 1e-3:
        df_plot.drop(columns=['biomass CHP CC'],inplace=True)
    
    if df_plot['biomass CHP'].max()/(df_plot.sum(axis=1).min()) < 1e-3:
        df_plot.drop(columns=['biomass CHP'],inplace=True)
    
    df_caps = pd.DataFrame()
    df_caps['battery'] = x.loc['G_battery [GW]'] + x.loc['G_homebattery [GW]']
    df_caps['battery_E'] = (x.loc['E_battery [GWh]'] + x.loc['E_homebattery [GWh]'])/1e3
    df_caps['X_power'] = x.loc['G_discharge [GW]']
    df_caps['X_energy'] = x.loc['E_cor [TWh]']
    df_caps['OCGT'] = x.loc['cap_gas_OCGT [GW]']#/df_plot['tot']*100
    df_caps['CCGT'] = x.loc['cap_gas_CCGT [GW]']
    # df_caps['gas'] = df_caps['gas_OCGT'] + df_caps['gas_CCGT']
    df_caps['nuclear'] =  x.loc['cap_nuclear [GW]']
    df_caps['gas_CHP_CC'] =  x.loc['cap_gas_CHP_CC [GW]']
    df_caps['gas_CHP'] =  x.loc['cap_gas_CHP [GW]']
    df_caps['biomass_CHP_CC'] =  x.loc['cap_biomass_CHP_CC [GW]']
    df_caps['biomass_CHP'] =  x.loc['cap_biomass_CHP [GW]']
    df_caps['coal'] =  x.loc['cap_coal [GW]']
    df_caps['hydro'] = x.loc['cap_hydro [GW]']#/df_plot['tot']*100
    df_caps['onwind'] = x.loc['cap_onwind [GW]']#/df_plot['tot']*100
    df_caps['offwind'] = x.loc['cap_offwind [GW]']#/df_plot['tot']*100
    df_caps['wind'] = df_caps['onwind'] + df_caps['offwind']
    df_caps['solar'] = x.loc['cap_solar [GW]']#/df_plot['tot']*100
    df_caps['wind+solar'] = x.loc[['cap_onwind [GW]','cap_offwind [GW]','cap_solar [GW]']].sum()
    df_caps['LC'] = x.loc['load_coverage [%]']
    df_caps['x'] = x.loc[x_var]
    df_caps.sort_values(by = 'x',inplace=True)
    
    df_caps.set_index('x',inplace=True)
    
    #%% 
    fig_gen, ax_gen = plt.subplots()
    ax_gen.stackplot(df_plot.drop(columns=['tot']).index,df_plot.drop(columns=['tot']).T/1e6,colors=[tech_colors[str(i)] for i in list(df_plot.drop(columns=['tot']).columns)],labels=[str(i) for i in list(df_plot.drop(columns=['tot']).columns)],lw=0)
    #%%
    # xlims = ax_gen.get_xlim()
    ax_gen.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    ax_gen.set_ylim([0,8500])
    ax_gen.set_xlabel('Storage ' + x_vardic1[x_var])
    ax_gen.set_ylabel('Electricity generation [TWh]')
    ax_gen.grid(zorder=100)
    ax_gen.set_axisbelow(False)
    ax_gen.set_xlabel('Storage ' + x_vardic1[x_var])
    
    if sector == 0:
        fig_gen.legend(bbox_to_anchor=(0.95, 0),ncol=3,prop={'size':18},frameon=True)
    else:
        fig_gen.legend(bbox_to_anchor=(1.05, -0.05),ncol=3,prop={'size':18},frameon=True)
    fig_gen.savefig('figures/Actual_generation mix_' + x_vardic[x_var] + '_' + str(sector) + '.png',bbox_inches="tight",dpi=300)
    
    # ax_gen.plot(df_caps['solar'],color=tech_colors['solar'],lw=lw_raw,alpha=0.5)
    # ax_gen.plot(df_caps['wind'],color=tech_colors['onwind'],lw=lw_raw,alpha=0.5)
    # ax_gen.plot((df_caps['solar'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors['solar'],lw=2,label='solar')
    # ax_gen.plot((df_caps['wind'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors['onwind'],lw=2,label='wind')
    #%%
    if df_caps['gas_CHP_CC'].max() > 1:
        ax1.plot(df_caps['gas_CHP_CC'],color=tech_colors['gas CHP CC'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['gas_CHP_CC'].rolling(rolling_interval).mean()),color=tech_colors['gas CHP CC'],lw=2,label='gas CHP CC (' + str(int(CFs['gas CHP CC'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['gas CHP CC'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)')
    if df_caps['gas_CHP'].max() > 1:
        ax1.plot(df_caps['gas_CHP'],color=tech_colors['gas CHP'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['gas_CHP'].rolling(rolling_interval).mean()),color=tech_colors['gas CHP'],lw=2,label='gas CHP (' + str(int(CFs['gas CHP'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['gas CHP'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)')
    if df_caps['biomass_CHP'].max() > 1:
        ax1.plot(df_caps['biomass_CHP'],color=tech_colors['biomass CHP'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['biomass_CHP'].rolling(rolling_interval).mean()),color=tech_colors['biomass CHP'],lw=2,label='biomass CHP (' + str(int(CFs['biomass CHP'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['biomass CHP'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)')
    if (df_caps['OCGT'].max() > 1):
        ax1.plot(df_caps['OCGT'],color=tech_colors['OCGT'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['OCGT'].rolling(rolling_interval).mean()),color=tech_colors['OCGT'],lw=2,label='OCGT (' + str(int(CFs['OCGT'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['OCGT'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)') # .iloc[0:-5]
    if df_caps['CCGT'].max() > 1:
        ax1.plot(df_caps['CCGT'],color=tech_colors['CCGT'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['CCGT'].rolling(rolling_interval).mean()),color=tech_colors['CCGT'],lw=2,label='CCGT (' + str(int(CFs['CCGT'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['CCGT'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)')
    if df_caps['nuclear'].max() > 1:
        ax1.plot(df_caps['nuclear'],color=tech_colors['nuclear'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['nuclear'].rolling(rolling_interval).mean()),color=tech_colors['nuclear'],lw=2,label='nuclear (' + str(int(CFs['nuclear'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['nuclear'].replace({np.inf:np.nan}).dropna().std())) +  ' hrs)')
    if df_caps['coal'].max() > 1:
        ax1.plot(df_caps['coal'],color=tech_colors['coal'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['coal'].rolling(rolling_interval).mean()),color=tech_colors['coal'],lw=2,label='coal (' + str(int(CFs['coal'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['coal'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)')
    if df_caps['biomass_CHP_CC'].max() > 1:
        ax1.plot(df_caps['biomass_CHP_CC'],color=tech_colors['biomass CHP CC'],lw=lw_raw,alpha=0.5)
        ax1.plot((df_caps['biomass_CHP_CC'].rolling(rolling_interval).mean()),color=tech_colors['biomass CHP CC'],lw=2,label='biomass CHP CC (' + str(int(CFs['biomass CHP CC'].replace({np.inf:np.nan}).dropna().mean())) + r'$\pm$' + str(int(CFs['biomass CHP CC'].replace({np.inf:np.nan}).dropna().std())) + ' hrs)')
        
    df_lc = df_caps.copy()
    df_lc.set_index('LC',inplace=True)
    df_lc.sort_index(inplace=True)
    ax2.plot(df_lc['battery'],color=tech_colors['battery'],lw=lw_raw,alpha=0.5)
    ax2.plot(df_lc['X_power'],color=tech_colors['storage X'],lw=lw_raw,alpha=0.5)
    ax2.plot((df_lc['battery'].rolling(rolling_interval).mean()),color=tech_colors['battery'],lw=2,label='Battery')
    ax2.plot((df_lc['X_power'].rolling(rolling_interval).mean()),color=tech_colors['storage X'],lw=2,label='Storage-X')
    
    ax1.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    # ax.set_xticklabels([])
    
    # ax.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    # ax.set_xticklabels([])
    # 
    # ax.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    # ax3.set_xticklabels([])
    
    # ax.set_ylabel('Baseload power \n capacities [GW]')
    ax1.set_ylabel('Backup power \n capacities [GW]')
    ax2.set_ylabel('Storage discharge \n capacities [GW]')
    ax1.set_xlabel('Storage-X Energy capacity [TWh]')
    ax2.set_xlabel('Storage-X Load coverage [%]')
    
    ax1.axvline(2,ls='--',lw=2,color='k')
    
    ax1.set_ylim([0,100])
    ax2.set_ylim([0,ax2.get_ylim()[1]])
        
    ax1.set_xlim([0,df_caps.index.max()])
    ax2.set_xlim([0,df_lc.index.max()])
    
    # ax2.set_ylim([0,500])
    if sector == 'T-H':
        ax1.legend(prop={'size':18},ncol=1,bbox_to_anchor=(0.3,0.085))
    else:
        ax1.legend(prop={'size':18},ncol=1)
        
    ax2.legend(prop={'size':18})
    
    # handles, labels = ax.get_legend_handles_labels()
    # handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # handles3, labels3 = ax3.get_legend_handles_labels()
    
    # fig.legend(list(reversed(handles)) + handles3, list(reversed(labels)) + labels3, bbox_to_anchor=(0.88, 0.04),ncol=4,prop={'size':fs},frameon=True)
    # fig.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(0.95, 0.04),ncol=3,prop={'size':fs},frameon=True)
    # fig.legend(handles, labels, bbox_to_anchor=(0.95, 0.04),ncol=3,prop={'size':fs},frameon=True)
    
    fig.savefig('figures/Generation mix_' + x_vardic[x_var] + '_' + str(sector) + '.png',bbox_inches="tight",dpi=300)
    
    #%%
    # Wind share
    wind_share = df_plot['wind']/df_plot['tot']*100
    solar_share = df_plot['solar']/df_plot['tot']*100
    hydro_share = df_plot['hydro']/df_plot['tot']*100
    
    print('Sector: ' + str(sector))
    print('Wind: ' + str(round(wind_share.mean(),1)) + ' pm ' + str(round(wind_share.std(),1)))
    print('Solar: ' + str(round(solar_share.mean(),1)) + ' pm ' + str(round(solar_share.std(),1)))
    print('Hydro: ' + str(round(hydro_share.mean(),1)) + ' pm ' + str(round(hydro_share.std(),1)))
    #%%
    # if x_var == 'load_coverage [%]':
    #     fig,ax = plt.subplots(figsize=[10,5])
    #     ax.plot(df_caps['battery_E'],color=tech_colors['battery'],lw=lw_raw,alpha=0.5)
    #     ax.plot(df_caps['X_energy'],color=tech_colors['storage X'],lw=lw_raw,alpha=0.5)
    #     ax.plot((df_caps['battery_E'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors['battery'],lw=2,label='Battery')
    #     ax.plot((df_caps['X_energy'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors['storage X'],lw=2,label='Storage-X')
    #     ax.set_xlabel('Load coverage [%]')
    #     ax.set_ylabel('Energy capacity [TWh]')
    #     ax.set_ylim([0,max(df_caps['X_energy'])])
    #     ax.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    #     fig.savefig('figures/Energy_capacity_vs_LC_' + str(sector) + '.png',bbox_inches="tight",dpi=300)
    
    #     fig,ax = plt.subplots(figsize=[10,5])
    #     ax.plot(df_caps['battery'],color=tech_colors['battery'],lw=lw_raw,alpha=0.5)
    #     ax.plot(df_caps['X_power'],color=tech_colors['storage X'],lw=lw_raw,alpha=0.5)
    #     ax.plot((df_caps['battery'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors['battery'],lw=2,label='Battery')
    #     ax.plot((df_caps['X_power'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors['storage X'],lw=2,label='Storage-X')
    #     ax.set_xlabel('Load coverage [%]')
    #     ax.set_ylabel('Power capacity [GW]')
    #     ax.set_ylim([0,max(df_caps['X_power'])])
    #     ax.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    #     fig.savefig('figures/Power_capacity_vs_LC_' + str(sector) + '.png',bbox_inches="tight",dpi=300)

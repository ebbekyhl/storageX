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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tech_colors import tech_colors
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Plotting layout
fs = 17
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

# Read result configurations
# filename = '../results/sspace_w_sectorcoupling.csv'
filename = '../results/sspace_w_sectorcoupling_merged.csv'
df = pd.read_csv(filename,index_col=0).fillna(0)
df_T = df.T
df_T = df_T.query('sector == @sector').drop(columns='sector').astype(float)
df_T = df_T.sort_values(by='c_hat [EUR/kWh]')
df_T['E [TWh]'] = df_T['E [GWh]']*1e-3 # convert GWh to TWh
df_T = df_T.loc[df_T['eta1 [-]'][df_T['eta1 [-]'] <= 1].index]
df_T['E_cor [TWh]'] = df_T['E [TWh]']*df_T['eta2 [-]']
df = df_T.reset_index(drop=True).T
#%%
# df = df[df.loc['load_coverage [%]'][df.loc['load_coverage [%]'] < 2].index]
df = df[df.loc['E_cor [TWh]'][df.loc['E_cor [TWh]'] <= 15].index]
rolling_interval = 20

# Threshold that storage configurations should fulfill to be included in the analysis
threshold_E = 0# 2000 # GWh

#%% Generation mix

# Variable used for sorting the configurations
# x_vars = ['load_coverage [%]']
x_vars = ['E_cor [TWh]']

# Total generation
gen_tot = (df.loc['gas_OCGT_gen [MWh]'] + df.loc['gas_CCGT_gen [MWh]']
            + df.loc['coal_gen [MWh]'] + df.loc['nuclear_gen [MWh]']
            + df.loc['onwind_gen [MWh]'] + df.loc['offwind_gen [MWh]'] 
            + df.loc['solar_gen [MWh]'] + df.loc['hydro_gen [MWh]']
            + df.loc['gas_CHP_CC_gen [MWh]'] + df.loc['gas_CHP_gen [MWh]'])

# Storage configurations
x_low = df[df.loc['E [GWh]'].idxmin()]
x = df[df.loc['E [GWh]'][df.loc['E [GWh]'] > threshold_E].index]

# Plotting
for x_var in x_vars:
    fig = plt.figure(figsize=(12, 14))
    nrows = 6
    ncols = 5
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0)
    gs.update(hspace=0.4)
    cp = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = plt.subplot(gs[3:,:])
    ax1 = plt.subplot(gs[0:1,:]) #,sharex=ax)
    ax2 = plt.subplot(gs[1:2,:])
    ax3 = plt.subplot(gs[2:3,:])
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
    df_plot['gas'] = x.loc['gas_OCGT_gen [MWh]'] + x.loc['gas_CCGT_gen [MWh]']
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
    tot = df_plot['tot']
    Ecap =  df_plot['Ecap']
    Gccap =  df_plot['Gccap']
    Gdcap =  df_plot['Gdcap']
    etac =  df_plot['etac']
    etad =  df_plot['etad']
    cc =  df_plot['cc']
    cd =  df_plot['cd']
    chat = df_plot['chat']
    syscost = df_plot['syscost']
    bat_lc = df_plot['bat_lc']
    df_plot.drop(columns=['tot','Ecap','Gccap','Gdcap','etac','etad','cc','cd','chat','syscost','bat_lc'],inplace=True)
    
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
    df_caps['gas_OCGT'] = x.loc['cap_gas_OCGT [GW]']#/df_plot['tot']*100
    df_caps['gas_CCGT'] = x.loc['cap_gas_CCGT [GW]']
    df_caps['gas'] = df_caps['gas_OCGT'] + df_caps['gas_CCGT']
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
    
    df_caps['x'] = x.loc[x_var]
    df_caps.sort_values(by = 'x',inplace=True)
    
    df_caps.set_index('x',inplace=True)
    
    ax.stackplot(df_plot.index,df_plot.T/1e6,colors=[tech_colors(str(i)) for i in list(df_plot.columns)],labels=[str(i) for i in list(df_plot.columns)])
    
    genmix = df_plot.T/tot*100
    xlims = ax.get_xlim()

    ax.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    ax.set_ylim([0,8500])
    ax.set_xlabel('Storage ' + x_vardic1[x_var])
    ax.set_ylabel('Electricity generation [TWh]')
    
    ax.grid(zorder=100)
    ax.set_axisbelow(False)
    
    ax1.plot(df_caps['solar'],color=tech_colors('solar'),lw=0.05,alpha=0.5)
    ax1.plot(df_caps['wind'],color=tech_colors('onwind'),lw=0.05,alpha=0.5)
    ax1.plot((df_caps['solar'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('solar'),lw=2)
    ax1.plot((df_caps['wind'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('onwind'),lw=2)

    ax2.plot(df_caps['gas'],color=tech_colors('gas'),lw=0.05,alpha=0.5)
    ax2.plot((df_caps['gas'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('gas'),lw=2)
    
    if df_caps['nuclear'].max() > 1:
        ax2.plot(df_caps['nuclear'],color=tech_colors('nuclear'),lw=0.05,alpha=0.5)
        ax2.plot((df_caps['nuclear'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('nuclear'),lw=2)
    if df_caps['coal'].max() > 1:
        ax2.plot(df_caps['coal'],color=tech_colors('coal'),lw=0.05,alpha=0.5)
        ax2.plot((df_caps['coal'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('coal'),lw=2)
    if df_caps['gas_CHP_CC'].max() > 1:
        ax2.plot(df_caps['gas_CHP_CC'],color=tech_colors('gas CHP CC'),lw=0.05,alpha=0.5)
        ax2.plot((df_caps['gas_CHP_CC'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('gas CHP CC'),lw=2)
    if df_caps['gas_CHP'].max() > 1:
        ax2.plot(df_caps['gas_CHP'],color=tech_colors('gas CHP'),lw=0.05,alpha=0.5)
        ax2.plot((df_caps['gas_CHP'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('gas CHP'),lw=2)
    if df_caps['biomass_CHP_CC'].max() > 1:
        ax2.plot(df_caps['biomass_CHP_CC'],color=tech_colors('biomass CHP CC'),lw=0.05,alpha=0.5)
        ax2.plot((df_caps['biomass_CHP_CC'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('biomass CHP CC'),lw=2)
    if df_caps['biomass_CHP'].max() > 1:
        ax2.plot(df_caps['biomass_CHP'],color=tech_colors('biomass CHP'),lw=0.05,alpha=0.5)
        ax2.plot((df_caps['biomass_CHP'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('biomass CHP'),lw=2)

    ax3.plot(df_caps['battery'],color=tech_colors('battery'),lw=0.05,alpha=0.5)
    ax3.plot(df_caps['X_power'],color=tech_colors('storage X'),lw=0.05,alpha=0.5)
    ax3.plot((df_caps['battery'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('battery'),lw=2,label='Battery')
    ax3.plot((df_caps['X_power'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('storage X'),lw=2,label='Storage-X')
    
    ax1.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    ax1.set_xticklabels([])
    
    ax2.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    ax2.set_xticklabels([])
    
    ax3.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
    ax3.set_xticklabels([])
    
    ax1.set_ylabel('Baseload power \n capacities [GW]')
    ax2.set_ylabel('Backup power \n capacities [GW]')
    ax3.set_ylabel('Storage discharge \n capacities [GW]')
    
    if ax2.get_ylim()[0] < 0:
        ax2.set_ylim([0,ax2.get_ylim()[1]])
    if ax3.get_ylim()[0] < 0:
        ax3.set_ylim([0,ax3.get_ylim()[1]])
    
    handles, labels = ax.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    
    fig.legend(list(reversed(handles)) + handles3, list(reversed(labels)) + labels3, bbox_to_anchor=(0.88, 0.04),ncol=4,prop={'size':fs},frameon=True)
    
    fig.savefig('../figures/Generation mix_' + x_vardic[x_var] + '_' + str(sector) + '.pdf',bbox_inches="tight")

#%%
# fig,ax = plt.subplots(figsize=[10,5])
# ax.plot(df_caps['battery_E'],color=tech_colors('battery'),lw=0.05,alpha=0.5)
# ax.plot(df_caps['X_energy'],color=tech_colors('storage X'),lw=0.05,alpha=0.5)
# ax.plot((df_caps['battery_E'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('battery'),lw=2,label='Battery')
# ax.plot((df_caps['X_energy'].rolling(rolling_interval).mean()).iloc[0:-5],color=tech_colors('storage X'),lw=2,label='Storage-X')
# ax.set_ylabel('Energy capacity [TWh]')
# ax.set_ylim([0,max(df_caps['X_energy'])])
# ax.set_xlim([min(x.loc[x_var]),max(x.loc[x_var])])
# fig.savefig('../figures/Energy_capacity_vs_LC_' + str(sector) + '.png',bbox_inches="tight",dpi=300)

#%%
# nsamples = 15
# lw = 0.1

# fig = plt.figure(figsize=(11, 6))
# nrows = 1
# ncols = 1
# gs = gridspec.GridSpec(nrows, ncols)
# gs.update(wspace=0)
# gs.update(hspace=0)
# cp = plt.rcParams['axes.prop_cycle'].by_key()['color']

# x_temp_df = pd.DataFrame(x.loc['load_coverage [%]'])
# x_temp_df['E_cor [TWh]'] = x.loc['E [TWh]']*x.loc['eta2 [-]']
# x_temp_df.set_index('load_coverage [%]',inplace=True)
# x_temp_df.sort_index(inplace=True)

# ax = plt.subplot(gs[:,:])
# ax.step(x_temp_df.index,x_temp_df['E_cor [TWh]']*1e3/Gdcap,lw=lw,color=tech_colors('storage X'),alpha=0.5)
# ax.plot(x_temp_df.index,(x_temp_df['E_cor [TWh]']*1e3/Gdcap).rolling(nsamples).mean(),lw=2,color=tech_colors('storage X'),label='Storage-X')
# ax.step(df_caps.index,df_caps['battery_E']*1e3/df_caps['battery'],lw=lw,color=tech_colors('battery'),alpha=0.5)
# ax.plot(df_caps.index,(df_caps['battery_E']*1e3/df_caps['battery']).rolling(nsamples).mean(),lw=2,color=tech_colors('battery'),label='Battery')
# ax.set_xlabel('Storage load coverage ' + r'$LC^X$' + ' [%]')
# ax.set_ylabel('Storage discharge time [h]')

# ax.set_xlim([0,18])

# ax.axhline(2,color='darkgrey',ls='--',lw=1,zorder=-1)
# ax.axhline(6,color='darkgrey',ls='--',lw=1,zorder=-1)
# ax.axhline(24,color='darkgrey',ls='--',lw=1,zorder=-1)
# ax.axhline(24*4,color='darkgrey',ls='--',lw=1,zorder=-1)

# ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.9,2.2,'2 hours',fontsize=fs,color='darkgrey')
# ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.9,6.5,'6 hours',fontsize=fs,color='darkgrey')
# ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.9,24+2,'1 day',fontsize=fs,color='darkgrey')
# ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.9,24*4+6,'4 days',fontsize=fs,color='darkgrey')

# ax.set_yscale('log')

# ax.set_ylim([1,200])

# fig.legend(bbox_to_anchor=(0.7, 0),ncol=2,prop={'size':fs},frameon=True)

# fig.savefig('../figures/load_hours_' + str(sector) + '.png',bbox_inches="tight",dpi=300)
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:27:50 2023

@author: au485969
"""

import matplotlib.pyplot as plt
plt.close('all')
import pandas as pd
import numpy as np
import seaborn as sn
sn.set_theme(style="ticks")
import warnings
warnings.filterwarnings('ignore')
# import statsmodels.api as sm

fs = 15
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'


#%% Read data
file = '../Results/sspace_w_sectorcoupling_wo_duplicates.csv'
# file = '../Results/sspace_3888.csv'

if file == '../Results/sspace_w_sectorcoupling_wo_duplicates.csv':
    sectors = ['T-H-I-B','T-H','-']
    sector_names = ['Fully sector-coupled','Electricity + Heating\n+ Land Transport', 'Electricity']
else:
    sectors = ['']
    sector_names = ['Electricity']

sspace_og = pd.read_csv(file,index_col=0)

E_sector = pd.DataFrame(index=np.arange(1082))
G_sector = pd.DataFrame(index=np.arange(1082))
bat_E_sector = pd.DataFrame(index=np.arange(1082))
bat_G_sector = pd.DataFrame(index=np.arange(1082))
LC_sector = pd.DataFrame(index=np.arange(1082))

# Multivariate regression using GLM for all sectors
i = 0
for sector in sectors:
    if file == '../Results/sspace_w_sectorcoupling_wo_duplicates.csv':
        sspace = sspace_og.T
        sspace['sector'] = sspace['sector'].fillna('-')
        sspace = sspace.query('sector == @sector')
        sspace = sspace.drop(columns='sector').astype(float).T
    else:
        sspace = sspace_og.copy()
    
    # Input
    df1 = pd.DataFrame(columns=['c_hat'])
    df1['c_hat'] = sspace.loc['c_hat [EUR/kWh]'].astype(float)
    df1['c1'] = sspace.loc['c1'].astype(float)
    df1['eta1'] = sspace.loc['eta1 [-]'].astype(float)
    df1['c2'] = sspace.loc['c2'].astype(float)
    df1['eta2'] = sspace.loc['eta2 [-]'].astype(float)
    df1['tau_SD'] = sspace.loc['tau [n_days]'].astype(float)
    
    # Output
    df1['E_cor'] = sspace.loc['E [GWh]'].astype(float)*df1['eta2']
    df1['LC'] = sspace.loc['load_coverage [%]'].astype(float)
    df1['G_d'] = sspace.loc['G_discharge [GW]'].astype(float) # Is already in units of electricity (see "networks/Make_sspace_new.py")
    df1['Bat_G'] = sspace.loc['G_battery [GW]'].astype(float)
    df1['Bat_E'] = sspace.loc['E_battery [GWh]'].astype(float)
    df1 = df1.sort_values(['c_hat','c1','eta1','c2','eta2','tau_SD'])
    
    E_sector[sector] = df1['E_cor'].values/1000 # convert GWh to TWh
    G_sector[sector] = df1['G_d'].values # convert GWh to TWh
    
    LC_sector[sector] = df1['LC'].values
    
    bat_E_sector[sector] = df1['Bat_E'].values/1000 # convert GWh to TWh
    bat_G_sector[sector] = df1['Bat_G'].values # convert GWh to TWh

    i += 1
    
#%% ENERGY CAPACITY FOR ALL THREE SYSTEMS
threshold = 2

fig,ax = plt.subplots()

sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
ratio = r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[2],color='green')

r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector3,inplace=True)
ratio = r_df[sector2] #/r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[1],color='blue')

r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector2,inplace=True)
ratio = r_df[sector3] #/r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[0],color='orange')

ax.set_xlim([0,60])

ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])

ax.legend(prop={'size':fs})
ax.set_xlabel('Storage-X energy capacity (TWh)')
fig.savefig('../figures/Storagex_energycap.png', dpi=600, bbox_inches='tight')
#%% POWER CAPACITY FOR ALL THREE SYSTEMS
fig,ax = plt.subplots()

sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
ratio = r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[2],color='green')

r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector3,inplace=True)
ratio = r_df[sector2] #/r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[1],color='blue')

r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector2,inplace=True)
ratio = r_df[sector3] #/r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[0],color='orange')

ecolor_dic = {'-':'green',
              'T-H':'blue',
              'T-H-I-B':'orange'}

ax.set_xlim([0,800])

ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])

ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Storage-X power capacity (GW)')
fig.savefig('../figures/Storagex_powercap.png', dpi=600, bbox_inches='tight')
#%% ENERGY CAPACITY SCALE UP BY INCLUDING SECTORS
fig,ax = plt.subplots()

sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector3,inplace=True)
ratio = r_df[sector2]/r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[1],color='blue')

r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = E_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector2,inplace=True)
ratio = r_df[sector3]/r_df[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[0],color='orange')

# ax.set_xlim([0,6])

ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])

ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Storage-X energy capacity change (' + r'$\times$' + ' $E^{Electricity}$' + ')')
fig.savefig('../figures/Storagex_energycap_change.png', dpi=600, bbox_inches='tight')

#%% POWER CAPACITY SCALE UP BY INCLUDING SECTORS
fig,ax = plt.subplots()

sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector3,inplace=True)
ratio = r_df[sector2]/r_df[sector1]
ratio = ratio.replace({np.inf:0})
sn.distplot(ratio,ax=ax,label=sector_names[1],color='blue')

r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
# r_df = G_sector.loc[E_sector[sector1][E_sector[sector1] < threshold].index]
r_df.drop(columns = sector2,inplace=True)
ratio = r_df[sector3]/r_df[sector1]
ratio = ratio.replace({np.inf:0})
sn.distplot(ratio,ax=ax,label=sector_names[0],color='orange')

ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])

ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Storage-X power capacity change (' + r'$\times$' + ' $G_d^{Electricity}$' + ')')
fig.savefig('../figures/Storagex_powercap_change.png', dpi=600, bbox_inches='tight')

#%% DURATION
fig,ax = plt.subplots()
sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

E_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
G_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
duration = E_df*1000/G_df
sn.distplot(duration['-'],ax=ax,label=sector_names[2],color='green')
sn.distplot(duration['T-H'],ax=ax,label=sector_names[1],color='blue')
sn.distplot(duration['T-H-I-B'],ax=ax,label=sector_names[0],color='orange')
ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Storage-X duration (hours)')
ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])
print('Max duration (storage-X):')
print(str(pd.concat([duration['-'],duration['T-H'],duration['T-H-I-B']]).quantile(1).round(2)),' hours')
fig.savefig('../figures/Storagex_duration.png', dpi=600, bbox_inches='tight')

fig,ax = plt.subplots()
bat_E_df = bat_E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
bat_G_df = bat_G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
bat_duration = bat_E_df*1000/bat_G_df
sn.distplot(bat_duration['-'],ax=ax,label=sector_names[2],color='green')
sn.distplot(bat_duration['T-H'],ax=ax,label=sector_names[1],color='blue')
sn.distplot(bat_duration['T-H-I-B'],ax=ax,label=sector_names[0],color='orange')
ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Battery duration (hours)')
ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])
print('Max battery duration:')
print(str(pd.concat([bat_duration['-'],bat_duration['T-H'],bat_duration['T-H-I-B']]).quantile(1).round(2)),' hours')
fig.savefig('../figures/Battery_duration.png', dpi=600, bbox_inches='tight')

#%% BATTERY CAPACITY
fig,ax = plt.subplots()
sn.distplot(1000*bat_E_df['-'][bat_E_df['-']*1000 > 0.01],ax=ax,label=sector_names[2],color='green')
sn.distplot(1000*bat_E_df['T-H'][bat_E_df['T-H']*1000 > 0.01],ax=ax,label=sector_names[1],color='blue')
sn.distplot(1000*bat_E_df['T-H-I-B'][bat_E_df['T-H-I-B']*1000 > 0.01],ax=ax,label=sector_names[0],color='orange')
ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Battery energy capacity [GWh]')
ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])
fig.savefig('../figures/Battery_energycap.png', dpi=600, bbox_inches='tight')

fig,ax = plt.subplots()
sn.distplot(bat_G_df['-'][bat_G_df['-']>0.005],ax=ax,label=sector_names[2],color='green')
sn.distplot(bat_G_df['T-H'][bat_G_df['T-H']>0.005],ax=ax,label=sector_names[1],color='blue')
sn.distplot(bat_G_df['T-H-I-B'][bat_G_df['T-H-I-B']>0.005],ax=ax,label=sector_names[0],color='orange')
ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Battery power capacity [GW]')
ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])
fig.savefig('../figures/Battery_powercap.png', dpi=600, bbox_inches='tight')
#%% DURATION CHANGE
fig,ax = plt.subplots()

sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

E_df = E_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
G_df = G_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]

duration = E_df*1000/G_df

ratio = duration[sector2]/duration[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[1],color='blue')

ratio = duration[sector3]/duration[sector1]
sn.distplot(ratio,ax=ax,label=sector_names[0],color='orange')

ax.set_xlim([0.2,2.5])

ax.set_yticklabels([ "{:0.3f}".format(x) for x in ax.get_yticks()])

ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Storage-X duration change (' + r'$\times$' + ' duration$^{Electricity}$' + ')')
fig.savefig('../figures/Storagex_duration_change.png', dpi=600, bbox_inches='tight')

#%% LOAD COVERAGE
fig,ax = plt.subplots()
sector1 = '-'
sector2 = 'T-H'
sector3 = 'T-H-I-B'

LC_df = LC_sector.loc[E_sector[sector1][E_sector[sector1] >= threshold].index]
sn.distplot(LC_df['-'],ax=ax,label=sector_names[2],color='green')
sn.distplot(LC_df['T-H'],ax=ax,label=sector_names[1],color='blue')
sn.distplot(LC_df['T-H-I-B'],ax=ax,label=sector_names[0],color='orange')
ax.legend(prop={'size':fs},frameon=True)
ax.set_xlabel('Storage-X load coverage [%]')
print('Max load coverage (95% quantile):')
print(str(pd.concat([LC_df['-'],LC_df['T-H'],LC_df['T-H-I-B']]).quantile(0.95).round(2)),' %')
fig.savefig('../figures/Storagex_load_coverage.png', dpi=600, bbox_inches='tight')
#%%
sspace_ref = sspace[sspace.loc['eta1 [-]'][sspace.loc['eta1 [-]'] == 0.5].index]
sspace_ref = sspace_ref[sspace_ref.loc['c_hat [EUR/kWh]'][sspace_ref.loc['c_hat [EUR/kWh]'] == 2].index]
sspace_ref = sspace_ref[sspace_ref.loc['tau [n_days]'][sspace_ref.loc['tau [n_days]'] == 30].index]
sspace_ref = sspace_ref[sspace_ref.loc['c1'][sspace_ref.loc['c1'] == 350].index]
sspace_ref = sspace_ref[sspace_ref.loc['c2'][sspace_ref.loc['c2'] == 350].index]


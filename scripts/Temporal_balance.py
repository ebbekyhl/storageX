# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:32:25 2022

@author: au485969
"""
import matplotlib.gridspec as gridspec
from tech_colors import tech_colors
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from read_system_files import read_system_files
import numpy as np
import pypsa
plt.close('all')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define here the system %%%%%%%%%%%%%%%%%%%%%%%%%%%%
sector = '-'          # no sector-coupling
# sector = '-T-H'     # Land transport and heating
# sector = '-T-H-I-B' # Land transport, heating, industry, shipping, aviation, and biomass

# value = '0.0' # 0% CO2 emissions relative to 1990-levels
# value = '1.0' # 1% CO2 emissions relative to 1990-levels
# value = '3.0' # 3% CO2 emissions relative to 1990-levels
value = '5.0'   # 5% CO2 emissions relative to 1990-levels
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Plotting layout
fs = 20
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

# Plotting function
def plot_energy_balance(X1,X2,c,period,xcolor,xlabel):
    onwind_y_c = X1[0]
    offwind_y_c = X1[1]
    solar_y_c = X1[2]
    hydro_y_c = X1[3] + X1[-1] #res + ror + PHS
    gas_OCGT_y_c = X1[4]
    gas_CCGT_y_c = X1[5]
    coal_y_c = X1[6]
    nuclear_y_c = X1[7]
    gas_CHP_CC_y_c = X1[8]
    gas_CHP_y_c = X1[9]
    biomass_CC_y_c = X1[10]
    biomass_y_c = X1[11]
    
    discharge_y_c = X1[12]
    battery_discharge_y_c = X1[13]
    import_dc_y_c = X1[14]
    import_ac_y_c = X1[15]
    
    load_c = X2[0]
    export_dc_y_c = X2[1]
    export_ac_y_c = X2[2]
    charge_y_c = X2[3]
    H2_charge_y_c = X2[4]
    battery_charge_y_c = X2[5]
    phs_charge_y_c = X2[6]
    
    fig = plt.figure(figsize=(14, 10))
    ncol = 4
    nrow = 6
    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=0.2)
    gs.update(hspace=0.4) 
    
    ax = plt.subplot(gs[0:5,0:])
    
    ax.fill_between(onwind_y_c.index,onwind_y_c,lw=0,color=tech_colors('onwind'),label='Onwind')
    ax.fill_between(onwind_y_c.index,
                    onwind_y_c,
                    onwind_y_c+offwind_y_c,lw=0,color=tech_colors('offwind'),label='Offwind')
    ax.fill_between(onwind_y_c.index,
                    onwind_y_c+offwind_y_c,
                    onwind_y_c+offwind_y_c+solar_y_c,lw=0,color=tech_colors('solar'),label='Solar')
    ax.fill_between(onwind_y_c.index,
                    onwind_y_c+offwind_y_c+solar_y_c,
                    onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c,lw=0,color=tech_colors('hydro'),label='Hydro')
    if gas_OCGT_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c,lw=0,color=tech_colors('OCGT'),label='OCGT')
    if gas_CCGT_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c,lw=0,color=tech_colors('CCGT'),label='CCGT')
    if coal_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c,lw=0,color=tech_colors('coal'),label='coal')
    if nuclear_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c,lw=0,color=tech_colors('nuclear'),label='nuclear')
    if gas_CHP_CC_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c,lw=0,color=tech_colors('gas CHP CC'),label='gas CHP-CC')
    if gas_CHP_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c,lw=0,color=tech_colors('gas CHP'),label='gas CHP')
    
    if biomass_CC_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c,lw=0,color=tech_colors('biomass CHP CC'),label='biomass CHP CC')

    if biomass_y_c.sum() > 10:
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c,lw=0,color=tech_colors('biomass CHP'),label='biomass CHP')
    
    ax.fill_between(onwind_y_c.index,
                    onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c,
                    onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c,lw=0,color=xcolor,label=xlabel)

    ax.fill_between(onwind_y_c.index,
                    onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c,
                    onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c + gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c+battery_discharge_y_c,lw=0,color=tech_colors('battery'),label='Battery')
    
    if c != 'EU':
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c+ gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c+battery_discharge_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c+ gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c+battery_discharge_y_c+import_dc_y_c,lw=0,color=tech_colors('transmission-dc'),label='DC Import/Export')
        ax.fill_between(onwind_y_c.index,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c+ gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c+battery_discharge_y_c+import_dc_y_c,
                        onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c+ gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c+battery_discharge_y_c+import_dc_y_c+import_ac_y_c,lw=0,color=tech_colors('transmission-ac'),label='AC Import/Export')
    
    ax.fill_between(load_c.index,
                    -load_c,lw=0,color='k',alpha=1,label='Load')
    
    ax.fill_between(load_c.index,
                    -load_c,
                    -load_c+export_dc_y_c,lw=0,color=tech_colors('transmission-dc'))
    
    ax.fill_between(load_c.index,
                    -load_c+export_dc_y_c,
                    -load_c+export_dc_y_c+export_ac_y_c,lw=0,color=tech_colors('transmission-ac'))
    
    ax.fill_between(load_c.index,
                    -load_c+export_dc_y_c+export_ac_y_c,
                    -load_c+export_dc_y_c+export_ac_y_c-charge_y_c,lw=0,color=xcolor)
    
    if H2_charge_y_c.sum() > 10:
        ax.fill_between(load_c.index,
                        -load_c+export_dc_y_c+export_ac_y_c-charge_y_c,
                        -load_c+export_dc_y_c+export_ac_y_c-charge_y_c-H2_charge_y_c,lw=0,color=tech_colors('H2'),label='H2')
    
    ax.fill_between(load_c.index,
                    -load_c+export_dc_y_c+export_ac_y_c-charge_y_c-H2_charge_y_c,
                    -load_c+export_dc_y_c+export_ac_y_c-charge_y_c-H2_charge_y_c-battery_charge_y_c,lw=0,color=tech_colors('battery'))
    
    ax.fill_between(load_c.index,
                    -load_c+export_dc_y_c+export_ac_y_c-charge_y_c-H2_charge_y_c-battery_charge_y_c,
                    -load_c+export_dc_y_c+export_ac_y_c-charge_y_c-H2_charge_y_c-battery_charge_y_c+phs_charge_y_c,lw=0,color=tech_colors('hydro'))
    
    y = solar_y_c.index.year[0]
    
    ax.plot(load_c,lw=1,color='k')
    ax.tick_params(axis='x', which='major', pad=10)
    
    date_lim_lower = pd.to_datetime(period[0] + str(y))
    date_lim_upper = pd.to_datetime(period[1] + str(y))
    
    ax.set_ylabel('Electricity [GW]')
    if c == 'EU' and sector == '-':
        ax.set_ylim([-500,500])
    elif c == 'EU' and sector == '-T-H':
        ax.set_ylim([-1000,1000])
    elif c == 'EU' and sector == '-T-H-I-B':
        ax.set_ylim([-1400,1400])
    else:
        ax.set_ylim([-1.05*(onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+gas_CCGT_y_c+coal_y_c+nuclear_y_c+ gas_CHP_CC_y_c + gas_CHP_y_c + biomass_CC_y_c + biomass_y_c +discharge_y_c+battery_discharge_y_c+import_dc_y_c+import_ac_y_c).max(),1.05*(onwind_y_c+offwind_y_c+solar_y_c+hydro_y_c+gas_OCGT_y_c+discharge_y_c+battery_discharge_y_c+import_dc_y_c+import_ac_y_c).max()])
    
    ax.set_xlim([date_lim_lower,date_lim_upper])
    
    ax.tick_params(axis='both', which='major', direction='out')
        
    fig.legend(loc='upper center',ncol=5,borderaxespad=1,fontsize=15)
    
    if c != 'EU':
        fig.suptitle(c,fontproperties ={'size':15,'weight':'bold'})
        
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    return fig,ax

# Setup
countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 
            'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 
            'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 
            'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 
            'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 
            'SE', 'SI', 'SK']

cdic = {'EU':'Europe',
        'GB':'Great Britain',
        'DE':'Germany',
        'DK':'Denmark',
        'FR':'France',
        'SE': 'Sweden',
        'CH':'Switzerland',
        'ES':'Spain',
        'NO':'Norway'}

if sector == '' or sector == '-':
    ylimdic = {'EU':[-170,100],
                'GB':[-65,90],
                'DE':[-100,100],
                'DK':[-250,250],
                'FR':[-100,100],
                'SE':[-100,100],
                'CH':[-450,450],
                'ES':[-200,200],
                'NO':[-150,150],
                'SK':[-150,150]}
elif sector == '-T-H':
    ylimdic = {'EU':[-170,100],
                'GB':[-150,150],
                'DE':[-100,100],
                'DK':[-250,250],
                'FR':[-100,100],
                'SE':[-100,100],
                'CH':[-450,450],
                'ES':[-200,200],
                'NO':[-150,150],
                'SK':[-150,150]}
elif sector == '-T-H-I-B':
    ylimdic = {'EU':[-170,100],
                'GB':[-400,400],
                'DE':[-100,100],
                'DK':[-500,500],
                'FR':[-100,100],
                'SE':[-100,100],
                'CH':[-450,450],
                'ES':[-200,200],
                'NO':[-150,150],
                'SK':[-300,300]}

freq = 1 # averaging after being sorted (technically, not the correct way)
freq1 = 1 # averaging before being sorted (more correct)

path = '../results/temporal/'

sector_dic = {'-T-H':'TH','-T-H-I-B':'fullsector'}

tres = '3h'
fs_title = 18

for c in ['GB']: #,'DE','DK','CH','ES','SE','EU']:
    
    # Residual load figure
    fig_dur, ax_dur = plt.subplots(1,1,figsize=[10,7])
    count = 0

    [load_y_c, onwind_y_c, offwind_y_c, solar_y_c, 
     phs_charge_y_c, phs_discharge_y_c, hydro_y_c, charge_y_c, 
        discharge_y_c, H2_charge_y_c, gas_OCGT_y_c, gas_CCGT_y_c, coal_y_c, nuclear_y_c, gas_CHP_CC_y_c,gas_CHP_y_c, biomass_CC_y_c, biomass_y_c, battery_discharge_y_c, battery_charge_y_c, import_dc_y_c, import_ac_y_c,  
        export_dc_y_c, export_ac_y_c, h2_y_c, h2_e_caps_y_i, e_battery_y_c, battery_e_caps_y_i, ror_y_c] = read_system_files(path, c, value+sector,tres)

    load_y_c = load_y_c.rolling(freq1).mean()
    onwind_y_c = onwind_y_c.rolling(freq1).mean()
    offwind_y_c = offwind_y_c.rolling(freq1).mean()
    solar_y_c = solar_y_c.rolling(freq1).mean()
    phs_charge_y_c = phs_charge_y_c.rolling(freq1).mean()
    phs_discharge_y_c = phs_discharge_y_c.rolling(freq1).mean()
    hydro_y_c = hydro_y_c.rolling(freq1).mean()
    charge_y_c = charge_y_c.rolling(freq1).mean()
    discharge_y_c = discharge_y_c.rolling(freq1).mean()
    H2_charge_y_c = H2_charge_y_c.rolling(freq1).mean()
    gas_OCGT_y_c = gas_OCGT_y_c.rolling(freq1).mean()
    gas_CCGT_y_c = gas_CCGT_y_c.rolling(freq1).mean()
    coal_y_c = coal_y_c.rolling(freq1).mean()
    nuclear_y_c = nuclear_y_c.rolling(freq1).mean()
    gas_CHP_CC_y_c = gas_CHP_CC_y_c.rolling(freq1).mean()
    gas_CHP_y_c = gas_CHP_y_c.rolling(freq1).mean()
    biomass_CC_y_c = biomass_CC_y_c.rolling(freq1).mean()
    biomass_y_c = biomass_y_c.rolling(freq1).mean()
    battery_discharge_y_c = battery_discharge_y_c.rolling(freq1).mean()
    battery_charge_y_c = battery_charge_y_c.rolling(freq1).mean()
    import_dc_y_c = import_dc_y_c.rolling(freq1).mean()
    import_ac_y_c = import_ac_y_c.rolling(freq1).mean()
    export_dc_y_c = export_dc_y_c.rolling(freq1).mean()
    export_ac_y_c = export_ac_y_c.rolling(freq1).mean()
    h2_y_c = h2_y_c.rolling(freq1).mean()
    e_battery_y_c = e_battery_y_c.rolling(freq1).mean()
    ror_y_c = ror_y_c.rolling(freq1).mean()

    if sector == '-T-H' or sector == '-T-H-I-B':
        n = pypsa.Network('../Results/17thJune_cross_sectoral_solution_space/n_' + sector_dic[sector] + '.nc')
    
        if c != 'EU':
            buses = n.buses.query('carrier == "AC"').index
            buses = buses[list(np.where([c in buses[i] for i in range(len(buses))])[0])]
            links = n.links
            generators = n.generators.loc[n.generators.bus.loc[[n.generators.bus.iloc[i] in buses for i in range(len(n.generators.bus))]].index]
            storage_units = n.storage_units.loc[n.storage_units.bus.loc[[n.storage_units.bus.iloc[i] in buses for i in range(len(n.storage_units.bus))]].index]
            
            loads = n.loads.query('carrier == "industry electricity"').index
            loads = n.loads.loc[loads[np.where([c in loads[i] for i in range(len(loads))])].tolist()]
            
            xss_0 = [links.bus0[links.bus0 == i].loc[[i in links.bus0[links.bus0 == i].index[j] for j in range(len(links.bus0[links.bus0 == i].index))]].drop([i + ' X Charge',i + ' battery charger',i + ' electricity distribution grid',i + ' H2 Electrolysis']).index.tolist() for i in buses]
            loads_0 = [x for xs in xss_0 for x in xs] # links using electricity from HVAC bus
            loads_1 = [links[links.index == i + ' electricity distribution grid'].index.item() for i in buses]
            loads_t_0 = n.links_t.p0[loads_0].sum(axis=1)
            loads_t_1 = n.links_t.p0[loads_1][n.links_t.p0[loads_1] > 0].fillna(0).sum(axis=1)
            loads_t_2 = n.loads_t.p[loads.index].sum(axis=1)
        
            loads_sum = (loads_t_0 + loads_t_1 + loads_t_2)/1e3
        
            # Electrolysis removed from the load in the line below:
            xss1 = [links.bus1[links.bus1 == i].loc[[i in links.bus1[links.bus1 == i].index[j] for j in range(len(links.bus1[links.bus1 == i].index))]].drop([i + ' X Discharge',i + ' battery discharger']).index.tolist() for i in buses]
            supply_0 = [x for xs in xss1 for x in xs] # links supplying electricity to HVAC bus
            supply_t_0 = -n.links_t.p1[supply_0].sum(axis=1)
            supply_t_1 = n.links_t.p1[loads_1][n.links_t.p1[loads_1] > 0].fillna(0).sum(axis=1)
            supply_t_2 = n.generators_t.p[generators.query('carrier == ["offwind-ac","offwind-dc","onwind","ror","solar","solar rooftop"]').index].sum(axis=1)
            supply_t_3 = n.storage_units_t.p[storage_units.query('carrier == "hydro"').index].sum(axis=1)
        
            supply_sum = (supply_t_0 + supply_t_1 + supply_t_2 + supply_t_3)/1e3
            
            load_y_c = loads_sum #supply_sum
        else:
            buses = n.buses.query('carrier == "AC"').index
            buses = buses #[list(np.where([c in buses[i] for i in range(len(buses))])[0])]
            links = n.links
            generators = n.generators.loc[n.generators.bus.loc[[n.generators.bus.iloc[i] in buses for i in range(len(n.generators.bus))]].index]
            storage_units = n.storage_units.loc[n.storage_units.bus.loc[[n.storage_units.bus.iloc[i] in buses for i in range(len(n.storage_units.bus))]].index]
            
            loads = n.loads.query('carrier == "industry electricity"').index
            loads = n.loads.loc[loads[np.where([c in loads[i] for i in range(len(loads))])].tolist()]
            
            xss_0 = [links.bus0[links.bus0 == i].loc[[i in links.bus0[links.bus0 == i].index[j] for j in range(len(links.bus0[links.bus0 == i].index))]].drop([i + ' X Charge',i + ' battery charger',i + ' electricity distribution grid',i + ' H2 Electrolysis']).index.tolist() for i in buses]
            loads_0 = [x for xs in xss_0 for x in xs] # links using electricity from HVAC bus
            loads_1 = [links[links.index == i + ' electricity distribution grid'].index.item() for i in buses]
            loads_t_0 = n.links_t.p0[loads_0].sum(axis=1)
            loads_t_1 = n.links_t.p0[loads_1][n.links_t.p0[loads_1] > 0].fillna(0).sum(axis=1)
            loads_t_2 = n.loads_t.p[loads.index].sum(axis=1)
        
            loads_sum = (loads_t_0 + loads_t_1 + loads_t_2)/1e3
        
            # Electrolysis removed from the load in the line below:
            xss1 = [links.bus1[links.bus1 == i].loc[[i in links.bus1[links.bus1 == i].index[j] for j in range(len(links.bus1[links.bus1 == i].index))]].drop([i + ' X Discharge',i + ' battery discharger']).index.tolist() for i in buses]
            supply_0 = [x for xs in xss1 for x in xs] # links supplying electricity to HVAC bus
            supply_t_0 = -n.links_t.p1[supply_0].sum(axis=1)
            supply_t_1 = n.links_t.p1[loads_1][n.links_t.p1[loads_1] > 0].fillna(0).sum(axis=1)
            supply_t_2 = n.generators_t.p[generators.query('carrier == ["offwind-ac","offwind-dc","onwind","ror","solar","solar rooftop"]').index].sum(axis=1)
            supply_t_3 = n.storage_units_t.p[storage_units.query('carrier == "hydro"').index].sum(axis=1)
        
            supply_sum = (supply_t_0 + supply_t_1 + supply_t_2 + supply_t_3)/1e3
            
            load_y_c = loads_sum
            
    import_y_c = import_dc_y_c + import_ac_y_c
    export_y_c = export_dc_y_c + export_ac_y_c
    
    REN = onwind_y_c + offwind_y_c + solar_y_c + hydro_y_c
    LOA = load_y_c
    delta_g = LOA - REN
    delta_g_crit = delta_g
        
    if value == '3.0' and sector == '-':
        df_sort_delta_g = delta_g.sort_values()[::-1]
        sort_delta_g = df_sort_delta_g.rolling(freq).mean().values/load_y_c.max()*100
    elif value == '3.0' and sector != '-':
        df_sort_delta_g = delta_g.sort_values()[::-1]
        sort_delta_g = df_sort_delta_g.rolling(freq).mean().values/load_y_c.max()*100
    elif sector == '-':
        df_sort_delta_g = delta_g.sort_values()[::-1]
        sort_delta_g = df_sort_delta_g.rolling(freq).mean().values/load_y_c.max()*100
    else:
        df_sort_delta_g = delta_g.sort_values()[::-1]
        sort_delta_g = df_sort_delta_g.rolling(freq).mean().values/load_y_c.max()*100
        
    df_sort_delta_g_crit = delta_g_crit.sort_values()[::-1]
    sort_delta_g_crit = df_sort_delta_g_crit.rolling(freq).mean().values/load_y_c.max()*100
    
    sort_phs = phs_discharge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_X = discharge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_battery = battery_discharge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_gas_OCGT = gas_OCGT_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_gas_CCGT = gas_CCGT_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_coal = coal_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_nuclear = nuclear_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_gas_CHP_CC = gas_CHP_CC_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_gas_CHP = gas_CHP_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_biomass_CC = biomass_CC_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_biomass = biomass_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_import = import_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    exceedence = np.arange(1.,len(sort_import)+1) / len(REN)
    
    sort_X_charge = charge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_H2_charge = H2_charge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_battery_charge = battery_charge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_phs_charge = phs_charge_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    sort_export = export_y_c.loc[df_sort_delta_g_crit.index].rolling(freq).mean().values/load_y_c.max()*100
    
    ax_dur.fill_between(exceedence*100, sort_X, lw=0,zorder=2,color='purple',label='Storage X')
    ax_dur.fill_between(exceedence*100, sort_X + sort_battery, sort_X, lw=0,zorder=2,color=tech_colors('battery'),label='Battery')
    ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs, sort_X + sort_battery, lw=0,zorder=2,color=tech_colors('hydro'),label='PHS')
    ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT, sort_battery + sort_X + sort_phs, lw=0,zorder=2,color=tech_colors('OCGT'),label='OCGT')
    ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT, sort_battery + sort_X + sort_phs + sort_gas_OCGT, lw=0,zorder=2,color=tech_colors('CCGT'),label='CCGT')
    if sort_coal.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal, sort_battery + sort_X + sort_phs + sort_gas_OCGT + sort_gas_CCGT, lw=0,zorder=2,color=tech_colors('coal'),label='Coal')
    if sort_nuclear.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear, sort_battery + sort_X + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal, lw=0,zorder=2,color=tech_colors('nuclear'),label='Nuclear')
    if sort_gas_CHP_CC.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC, sort_battery + sort_X + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear, lw=0,zorder=2,color=tech_colors('gas CHP CC'),label='Gas CHP-CC')
    if sort_gas_CHP.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP, sort_battery + sort_X + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC, lw=0,zorder=2,color=tech_colors('gas CHP'),label='Gas CHP')
    if sort_biomass_CC.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP + sort_biomass_CC, sort_battery + sort_X + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP, lw=0,zorder=2,color=tech_colors('biomass CHP CC'),label='Biomass CHP CC')
    if sort_biomass.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, sort_X + sort_battery + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP + sort_biomass_CC + sort_biomass, sort_battery + sort_X + sort_phs + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP + sort_biomass_CC, lw=0,zorder=2,color=tech_colors('biomass CHP'),label='Biomass CHP')

    if c != 'EU':
        ax_dur.fill_between(exceedence*100, sort_phs + sort_battery + sort_X + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP + sort_biomass_CC + sort_biomass + sort_import, sort_phs + sort_battery + sort_X + sort_gas_OCGT + sort_gas_CCGT + sort_coal + sort_nuclear + sort_gas_CHP_CC + sort_gas_CHP + sort_gas_CHP + sort_biomass_CC, lw=0,zorder=2,color = tech_colors('transmission-dc'),label='Transmission',alpha=0.8)
    
    ax_dur.axhline(0,ls='--',color='k')
    ax_dur.plot(exceedence*100, sort_delta_g_crit, lw=2,zorder=2,color='k',label = 'Residual load ' + r'$\Delta \bar{g}$')
    
    sort_charge = (charge_y_c+battery_charge_y_c+phs_charge_y_c.abs()).loc[df_sort_delta_g_crit.index].rolling(freq).mean().values
    
    exceedence = np.arange(1.,len(REN)+1) / len(REN)
    ax_dur.plot(exceedence*100, sort_delta_g, lw=2,zorder=2,color='k',ls='--')
    
    ax_dur.fill_between(exceedence*100, -sort_X_charge, lw=0,zorder=2,color='purple')
    ax_dur.fill_between(exceedence*100, -sort_X_charge-sort_battery_charge, -sort_X_charge,lw=0,zorder=2,color=tech_colors('battery'))
    ax_dur.fill_between(exceedence*100, -sort_X_charge-sort_battery_charge+sort_phs_charge,-sort_X_charge-sort_battery_charge, lw=0,zorder=2,color=tech_colors('hydro'))
    if sort_H2_charge.sum() > 1e-1:
        ax_dur.fill_between(exceedence*100, -sort_X_charge-sort_battery_charge+sort_phs_charge-sort_H2_charge,-sort_X_charge-sort_battery_charge+sort_phs_charge, lw=0,zorder=2,color=tech_colors('H2'),label='H2')

    if c != 'EU':
        ax_dur.fill_between(exceedence*100, -sort_X_charge-sort_battery_charge+sort_phs_charge-sort_H2_charge+sort_export, -sort_X_charge-sort_battery_charge+sort_phs_charge-sort_H2_charge, lw=0,zorder=2,color = tech_colors('transmission-dc'),alpha=0.8)
    
    ax_dur.fill_between(exceedence[sort_delta_g < 0]*100, sort_delta_g[sort_delta_g < 0], lw=0.5,zorder=2,color='none',alpha=0.5,hatch='//',edgecolor='k',label='Excess renewable \n generation')
    
    ax_dur.set_ylabel('% of peak load')
    ax_dur.set_xlim([0,100])
    ax_dur.set_ylim([ylimdic[c][0],ylimdic[c][1]]) #[-max([delta_g.max(),abs(delta_g.min())]),max([delta_g.max(),abs(delta_g.min())])])
    
    ax_dur.set_xlabel('% of time')

    ax_dur.axvline(x=exceedence[np.where(abs(sort_delta_g) == abs(sort_delta_g).min())[0][0]]*100,ymin=0,ymax=1,lw=0.75,color='k',zorder=20)
    ax_dur.text(exceedence[np.where(abs(sort_delta_g) == abs(sort_delta_g).min())[0][0]]*100+1,-60,r'$\Delta \bar{g} = 0$',fontsize=fs)    

    ax_dur.grid()
    
    fig_dur.legend(prop={'size':fs},bbox_to_anchor=(0.85, 0),ncol=2)#,borderaxespad=0)
    fig_dur.savefig('../figures/Balancing_duration_curve_' + c + '_' + value+sector + '.png',
                bbox_inches="tight",dpi=300)
    
    freq = 8
    X1 = [onwind_y_c.rolling(freq).mean(),offwind_y_c.rolling(freq).mean(),solar_y_c.rolling(freq).mean(),
            hydro_y_c.rolling(freq).mean(), gas_OCGT_y_c.rolling(freq).mean(),gas_CCGT_y_c.rolling(freq).mean(),
            coal_y_c.rolling(freq).mean(),nuclear_y_c.rolling(freq).mean(),
            gas_CHP_CC_y_c.rolling(freq).mean(),gas_CHP_y_c.rolling(freq).mean(),
            biomass_CC_y_c.rolling(freq).mean(),biomass_y_c.rolling(freq).mean(),
            discharge_y_c.rolling(freq).mean(),battery_discharge_y_c.rolling(freq).mean(), import_dc_y_c.rolling(freq).mean(),
            import_ac_y_c.rolling(freq).mean(),phs_discharge_y_c.rolling(freq).mean()]
    
    X2 = [load_y_c.rolling(freq).mean(),export_dc_y_c.rolling(freq).mean(),export_ac_y_c.rolling(freq).mean(),
          charge_y_c.rolling(freq).mean(),H2_charge_y_c.rolling(freq).mean(),battery_charge_y_c.rolling(freq).mean(),
          phs_charge_y_c.rolling(freq).mean()]
    
    # Energy balance figure
    fig1,ax1 = plot_energy_balance(X1,X2,c,['1/1/','12/31/'],'purple','Storage-X')

    if freq > 1:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig1.savefig('../figures/Balancing_' + c + '_' + value+sector + '.png',
                     bbox_inches="tight",dpi=300)
    
    count += 1

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:19:41 2022

@author: au485969
"""

def investment_map(networks_opt, scen, moving_average, tech_colors):
    from scripts.plotting import plot_investment_map
    import pypsa 
    import matplotlib.pyplot as plt
    plt.close('all')
    
    networks_path = networks_opt['path']
    wyear = networks_opt['wyear']
    eta1 = networks_opt['eta1']
    eta2 = networks_opt['eta2']
    c1 = networks_opt['c1']
    c2 = networks_opt['c2']
    chat = networks_opt['chat']
    
    n = pypsa.Network(networks_path + 'elec_s_y' + wyear + '_n37_lv1.0__Co2L0.05-3H' + scen +'-solar+p3-dist1-X Charge+e' + eta1 + '-X Charge+c' + c1 + '-X Discharge+e' + eta2 + '-X Discharge+c' + c2 + '-X Store+c' + chat + '.nc')
    fig_investment = plot_investment_map(n, tech_colors, threshold=10,bus_size_factor=4.3e10)
    fig_investment.savefig('figures/Investment_map' + scen + '_eta_d_factor' + eta2 + '.png',bbox_inches="tight",dpi=600)

def storage_map(networks_opt, scen, moving_average, tech_colors):
    from scripts.plotting import plot_storage_map
    import pypsa 
    import matplotlib.pyplot as plt
    plt.close('all')
    
    networks_path = networks_opt['path']
    wyear = networks_opt['wyear']
    eta1 = networks_opt['eta1']
    eta2 = networks_opt['eta2']
    c1 = networks_opt['c1']
    c2 = networks_opt['c2']
    chat = networks_opt['chat']
    
    n = pypsa.Network(networks_path + 'elec_s_y' + wyear + '_n37_lv1.0__Co2L0.05-3H' + scen +'-solar+p3-dist1-X Charge+e' + eta1 + '-X Charge+c' + c1 + '-X Discharge+e' + eta2 + '-X Discharge+c' + c2 + '-X Store+c' + chat + '.nc')
    fig_storage = plot_storage_map(n, tech_colors, threshold=10,bus_size_factor=1e6)
    fig_storage.savefig('figures/Storage_capacity_map' + scen + '_eta_d_factor' + eta2 + '.png',bbox_inches="tight",dpi=600)

def temporal(networks_opt, scen, moving_average, tech_colors):
    from scripts.plotting import plot_series
    import pandas as pd
    import pypsa 
    import matplotlib.pyplot as plt
    plt.close('all')
    
    SOC_X = {}
    discharge_t_X = {}
    SOC_bat = {}
    discharge_t_bat = {}
    
    moving_average_steps = moving_average
    
    networks_path = networks_opt['path']
    wyear = networks_opt['wyear']
    eta1 = networks_opt['eta1']
    eta2 = networks_opt['eta2']
    c1 = networks_opt['c1']
    c2 = networks_opt['c2']
    chat = networks_opt['chat']
    
    # Moving average equal to, e.g, 24 takes the first 24 values (with 3-hourly resoulution, this is equivalent to a 3-daily moving average)
    global_constraint = {}
    # for scen in scens:
    n = pypsa.Network(networks_path + 'elec_s_y' + wyear + '_n37_lv1.0__Co2L0.05-3H' + scen +'-solar+p3-dist1-X Charge+e' + eta1 + '-X Charge+c' + c1 + '-X Discharge+e' + eta2 + '-X Discharge+c' + c2 + '-X Store+c' + chat + '.nc')
    global_constraint[scen] = n.global_constraints.loc['CO2Limit']
    
    c = 'EU'
    fig_AC_1year,supply = plot_series(n,country=c,dstart=pd.to_datetime('1/1/2013'),dend=pd.to_datetime('31/12/2013'),tech_colors=tech_colors,moving_average=moving_average_steps,carrier="AC")
    fig_AC_1year.savefig('figures/Timeseries_' + scen + '_eta_d_factor' + eta2 + '_' + c + '.png',bbox_inches="tight",dpi=600)

    dstart = pd.to_datetime('1/20/2013')
    dend = pd.to_datetime('3/1/2013')
    
    fig_AC_worst,supply = plot_series(n,country=c,dstart=dstart,dend=dend,tech_colors=tech_colors,moving_average=moving_average_steps,carrier="AC")
    fig_AC_worst.savefig('figures/Timeseries_worst_week' + scen + '_eta_d_factor' + eta2 + '_' + c +  '.png',bbox_inches="tight",dpi=600)

    # State-of-charge for storage-X
    SOC_X[scen] = n.stores_t.e[n.stores.query('carrier == "X"').index].sum(axis=1)/n.stores_t.e[n.stores.query('carrier == "X"').index].sum(axis=1).max()
    # Dispatch of storage-X
    discharge_t_X[scen] = -n.links_t.p1[n.links.query('carrier == "X Discharge"').index].sum(axis=1)/(-n.links_t.p1[n.links.query('carrier == "X Discharge"').index].sum(axis=1)).max()
    
    # State of charge for battery
    SOC_bat[scen] = n.stores_t.e[n.stores.query('carrier == "battery"').index].sum(axis=1)/n.stores_t.e[n.stores.query('carrier == "battery"').index].sum(axis=1).max()
    # Dispatch of battery
    discharge_t_bat[scen] = -n.links_t.p1[n.links.query('carrier == "battery discharger"').index].sum(axis=1)/-n.links_t.p1[n.links.query('carrier == "battery discharger"').index].sum(axis=1).max()

    return [fig_AC_1year, fig_AC_worst]

def state_of_charge(networks_opt, scen):
    import matplotlib.pyplot as plt
    import pypsa
    
    scen_dic = {'':'El.','-T-H':'El.-T-H','-T-H-I-B':'El.-T-H-I-B'}
    
    SOC_X = {}
    discharge_t_X = {}
    SOC_bat = {}
    discharge_t_bat = {}
    
    networks_path = networks_opt['path']
    wyear = networks_opt['wyear']
    eta1 = networks_opt['eta1']
    eta2 = networks_opt['eta2']
    c1 = networks_opt['c1']
    c2 = networks_opt['c2']
    chat = networks_opt['chat']
    
    n = pypsa.Network(networks_path + 'elec_s_y' + wyear + '_n37_lv1.0__Co2L0.05-3H' + scen +'-solar+p3-dist1-X Charge+e' + eta1 + '-X Charge+c' + c1 + '-X Discharge+e' + eta2 + '-X Discharge+c' + c2 + '-X Store+c' + chat + '.nc')
    
    # State-of-charge for storage-X
    SOC_X[scen] = n.stores_t.e[n.stores.query('carrier == "X"').index].sum(axis=1)/n.stores_t.e[n.stores.query('carrier == "X"').index].sum(axis=1).max()
    # Dispatch of storage-X
    discharge_t_X[scen] = -n.links_t.p1[n.links.query('carrier == "X Discharge"').index].sum(axis=1)/(-n.links_t.p1[n.links.query('carrier == "X Discharge"').index].sum(axis=1)).max()
    # State of charge for battery
    SOC_bat[scen] = n.stores_t.e[n.stores.query('carrier == "battery"').index].sum(axis=1)/n.stores_t.e[n.stores.query('carrier == "battery"').index].sum(axis=1).max()
    # Dispatch of battery
    discharge_t_bat[scen] = -n.links_t.p1[n.links.query('carrier == "battery discharger"').index].sum(axis=1)/-n.links_t.p1[n.links.query('carrier == "battery discharger"').index].sum(axis=1).max()

    # State of charge for batteries and stoage-X throughout the year
    fig1,ax1 = plt.subplots(figsize=[14,5])
    fig2,ax2 = plt.subplots(figsize=[14,5])
    fig3,ax3 = plt.subplots(figsize=[14,5])
    fig4,ax4 = plt.subplots(figsize=[14,5])

    SOC_X[scen].plot(ax=ax1,label=scen_dic[scen],linewidth=0.5)
    discharge_t_X[scen].plot(ax=ax2,label=scen_dic[scen])
    SOC_bat[scen].plot(ax=ax3,label=scen_dic[scen],linewidth=0.5)
    discharge_t_bat[scen].plot(ax=ax4,label=scen_dic[scen])
    plt.legend()
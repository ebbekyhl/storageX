# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:19:41 2022

@author: au485969
"""
import pandas as pd
import pypsa
from plotting import plot_series, worst_best_week, plot_storage_map, plot_investment_map
import yaml
import matplotlib.pyplot as plt
plt.close('all')
fs = 18
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

with open('tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']
tech_colors['CO2 capture'] = tech_colors['DAC']
tech_colors['domestic demand'] = '#050000'
tech_colors['industry demand'] = '#423737'
tech_colors['BEV'] = 'c'
tech_colors['EV battery'] = 'c'
tech_colors['heat pump'] = '#b52f2f'
tech_colors['resistive heater'] = '#c45c5c'
tech_colors['V2G'] = '#38f2d9'
tech_colors['transmission lines'] = '#6c9459'
tech_colors['storage-X'] = '#610555'
tech_colors['X'] = '#610555'
tech_colors['pumped hydro'] = tech_colors['hydroelectricity']
#%%
SOC_X = {}
delta_e_plus_X = {}
duration_X = {}
SOC_bat = {}
delta_e_plus_bat = {}
duration_bat = {}

# eta2 = '0.6'
# eta2 = '1.0'
eta2 = '1.9'

# chat = '3.0'
chat = '0.01'

# scens = ['','-T-H','-T-H-I-B']
# scens = ['-T-H-I-B']
scens = ['','-T-H','-T-H-I-B']

# Moving average equal to 24 takes the first 24 values (with 3-hourly resoulution, this is equivalent to a 3-daily moving average)

global_constraint = {}

scen_dic = {'':'El.','-T-H':'El.-T-H','-T-H-I-B':'El.-T-H-I-B'}
for scen in scens:
    n = pypsa.Network('../networks/high_efficiency/elec_s_y2003_n37_lv1.0__Co2L0.05-3H' + scen +'-solar+p3-dist1-X Charge+e1.0-X Charge+c1.0-X Discharge+e' + eta2 + '-X Discharge+c1.0-X Store+c0.15.nc')
    global_constraint[scen] = n.global_constraints.loc['CO2Limit']
    c = 'EU'
    fig_AC_1year,supply = plot_series(n,country=c,dstart=pd.to_datetime('1/1/2013'),dend=pd.to_datetime('31/12/2013'),tech_colors=tech_colors,moving_average=12,carrier="AC")
    fig_AC_1year.savefig('../figures/Timeseries_' + scen + '_eta_d_factor' + eta2 + '_' + c + '.png',bbox_inches="tight",dpi=600)

    dstart = pd.to_datetime('1/20/2013')
    dend = pd.to_datetime('3/1/2013')
    
    fig_AC_worst,supply = plot_series(n,country=c,dstart=dstart,dend=dend,tech_colors=tech_colors,moving_average=12,carrier="AC")
    fig_AC_worst.savefig('../figures/Timeseries_worst_week' + scen + '_eta_d_factor' + eta2 + '_' + c +  '.png',bbox_inches="tight",dpi=600)

    fig_storage = plot_storage_map(n, tech_colors, threshold=10,bus_size_factor=1e6)
    fig_storage.savefig('../figures/Storage_capacity_map' + scen + '_eta_d_factor' + eta2 + '.png',bbox_inches="tight",dpi=600)

    fig_investment = plot_investment_map(n, tech_colors, threshold=10,bus_size_factor=4.3e10)
    fig_investment.savefig('../figures/Investment_map' + scen + '_eta_d_factor' + eta2 + '.png',bbox_inches="tight",dpi=600)

    SOC_X[scen] = n.stores_t.e[n.stores.query('carrier == "X"').index].sum(axis=1)/n.stores_t.e[n.stores.query('carrier == "X"').index].sum(axis=1).max()#.plot()
    delta_e_plus_X[scen] = -n.links_t.p1[n.links.query('carrier == "X Discharge"').index].sum(axis=1)/(-n.links_t.p1[n.links.query('carrier == "X Discharge"').index].sum(axis=1)).max()
    SOC_bat[scen] = n.stores_t.e[n.stores.query('carrier == "battery"').index].sum(axis=1)/n.stores_t.e[n.stores.query('carrier == "battery"').index].sum(axis=1).max()#.plot()
    delta_e_plus_bat[scen] = -n.links_t.p1[n.links.query('carrier == "battery discharger"').index].sum(axis=1)/-n.links_t.p1[n.links.query('carrier == "battery discharger"').index].sum(axis=1).max()
    
    p_nom_opt = n.links.query('carrier == "X Discharge"').p_nom_opt*n.links.query('carrier == "X Discharge"').efficiency
    e_nom_opt = n.stores.query("carrier == 'X'").e_nom_opt
    
    duration_X[scen] = (e_nom_opt/p_nom_opt.values).mean()
#%%    
# fig1,ax1 = plt.subplots(figsize=[14,5])
# fig2,ax2 = plt.subplots(figsize=[14,5])
# fig3,ax3 = plt.subplots(figsize=[14,5])
# fig4,ax4 = plt.subplots(figsize=[14,5])
# for scen in scens:
#     SOC_X[scen].plot(ax=ax1,label=scen_dic[scen],linewidth=0.5)
#     delta_e_plus_X[scen].plot(ax=ax2,label=scen_dic[scen])
#     SOC_bat[scen].plot(ax=ax3,label=scen_dic[scen],linewidth=0.5)
#     delta_e_plus_bat[scen].plot(ax=ax4,label=scen_dic[scen])
    
# plt.legend()
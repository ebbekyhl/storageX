# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:54:09 2022

@author: au485969
"""

import matplotlib.gridspec as gridspec
# from tech_colors import tech_colors
import matplotlib.pyplot as plt
import pandas as pd
# import matplotlib.dates as mdates
# from read_system_files import read_system_files
# import numpy as np
# import pypsa

plt.close('all')

fs = 20
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

output = 'E [GWh]'
xco2 = 5
factor = 1e3

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

# variable = r'$\hat{c}$'
variable = 'CO2'

# year = '2000' 
# year = '2004'
# year = '2013'

# sector = ''
sector = '-'
# sector = '-T-H'
# sector = '-T-H-I-B'

# value = '0.0'
# value = '1.0'
# value = '3.0'
value = '5.0'

# unit = '€/kWh'
unit = '%'

size = 'small'
# size = 'big'
freq = 1 # averaging after being sorted (technically, not the correct way)
freq1 = 1 # averaging before being sorted (more correct)
# path = 'C:\\Users/au485969/OneDrive - Aarhus Universitet/PhD/WP1/Results/weather_year/small_ref/'

# path = 'C:\\Users/au485969/OneDrive - Aarhus Universitet/PhD/WP1/Results/17thJune_cross_sectoral_solution_space/simple_sweep/'
path = '../results/simple_sweep/'
# path = 'C:\\Users/au485969/OneDrive - Aarhus Universitet/PhD/WP1/Results/17thJune_cross_sectoral_solution_space/chat/'

# if size == 'big':
#     tres = '3h'
# else:
#     tres = 'h'

sector_dic = {'-T-H':'TH','-T-H-I-B':'fullsector'}

ecolor_dic = {'-':'green',
              'T-H':'blue',
              'T-H-I-B':'orange'}

tres = '3h'
fs_title = 18

name_dic = {'eta2 [-]': 'eta_d',
            'c_hat [EUR/kWh]':'c_hat',
            'c1':'c_c',
            'eta1 [-]':'eta_c',
            'tau [n_days]':'tau_SD',
            'c2':'c_d',
            'E [TWh]':'E'}

ref_dic = {'co2_cap [%]':5,
           'nnodes':37,
           'tres':3,
           'weatheryear':2013}

variables = ['co2_cap [%]','nnodes','tres','weatheryear']
variable_names = {'co2_cap [%]':'CO2',
                  'nnodes':'nodes',
                  'tres':'time',
                  'weatheryear':'weather'}

xlabs = {'co2_cap [%]':'CO2-cap relative to 1990-levels [%]',
         'nnodes':'Number of nodes',
         'tres':'Temporal resolution [hours]',
         'weatheryear':'Weather year'}

for var in variables:
    df1 = pd.read_csv(path + 'sspace_' + variable_names[var] + '.csv',index_col=0).fillna(0).T
    
    sector = 0 # no sector-coupling
    df1_elec = df1.query('sector == @sector').drop(columns='sector').astype(float)
    
    sector = 'T-H' # Transportation and heating
    df1_th = df1.query('sector == @sector').drop(columns='sector').astype(float)
    
    sector = 'T-H-I-B' # Full sector coupling
    df1_full = df1.query('sector == @sector').drop(columns='sector').astype(float)
    
    df1_elec = df1_elec.sort_values(by=var)
    df1_elec['E [GWh]'] = df1_elec['E [GWh]']*df1_elec['eta2 [-]']*1e-3 # convert GWh to TWh
    df1_elec.rename(columns={'E [GWh]':'E [TWh]'}, inplace=True)
    df1_elec = df1_elec.loc[df1_elec['eta1 [-]'][df1_elec['eta1 [-]'] <= 1].index]
    df1_elec = df1_elec.reset_index(drop=True)
    
    df1_th = df1_th.sort_values(by=var)
    df1_th['E [GWh]'] = df1_th['E [GWh]']*df1_th['eta2 [-]']*1e-3 # convert GWh to TWh
    df1_th.rename(columns={'E [GWh]':'E [TWh]'}, inplace=True)
    df1_th = df1_th.loc[df1_th['eta1 [-]'][df1_th['eta1 [-]'] <= 1].index]
    df1_th = df1_th.reset_index(drop=True)
    
    df1_full = df1_full.sort_values(by=var)
    df1_full['E [GWh]'] = df1_full['E [GWh]']*df1_full['eta2 [-]']*1e-3 # convert GWh to TWh
    df1_full.rename(columns={'E [GWh]':'E [TWh]'}, inplace=True)
    df1_full = df1_full.loc[df1_full['eta1 [-]'][df1_full['eta1 [-]'] <= 1].index]
    df1_full = df1_full.reset_index(drop=True)
    
    df1_elec = df1_elec[[var,'eta2 [-]','c_hat [EUR/kWh]','c1','eta1 [-]','tau [n_days]','c2','E [TWh]']]
    df1_elec.rename(columns=name_dic,inplace=True)
    
    df1_elec.set_index(var,inplace=True)
    
    df1_th = df1_th[[var,'eta2 [-]','c_hat [EUR/kWh]','c1','eta1 [-]','tau [n_days]','c2','E [TWh]']]
    df1_th.rename(columns=name_dic,inplace=True)
    
    df1_th.set_index(var,inplace=True)
    
    df1_full = df1_full[[var,'eta2 [-]','c_hat [EUR/kWh]','c1','eta1 [-]','tau [n_days]','c2','E [TWh]']]
    df1_full.rename(columns=name_dic,inplace=True)
    
    df1_full.set_index(var,inplace=True)
    
    fig = plt.figure(figsize=(12, 6))
    ncol = 4
    nrow = 6
    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=0.2)
    gs.update(hspace=0.4) 
    
    ax = plt.subplot(gs[0:,0:])
    
    if var == 'co2_cap [%]':
        ax.set_yscale('log')
        ax.set_ylim([0.01,40])
        ax.set_xlim([0,10])
    elif var == 'weatheryear':
        ax.set_xlim([1997,2013])
    else:
        ax.set_ylim([0,0.55])
        
    ax.plot(df1_elec['E'],'-',marker='.',color='green',label='Electricity')
    ax.plot(df1_th['E'],'-',marker='.',color='blue',label='Electricity + Heating + Land Transport')
    ax.plot(df1_full['E'],'-',marker='.',color='orange',label='Fully sector-coupled')
    
    fig.legend(prop={'size':fs},frameon=True,bbox_to_anchor=(0, -0.10), loc=3,
               ncol=3,borderaxespad=0)
    
    ax.set_ylabel('E [TWh]')
    ax.set_xlabel(xlabs[var])
    ax.grid(which='both',axis='y')
    
    fig.savefig('../figures/Reference_' + var + '_sectors.png',
                          bbox_inches="tight",dpi=300)
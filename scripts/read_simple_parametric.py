# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:26:51 2022

@author: au485969
"""

import pandas as pd

def normalize_system_cost(dataframe,fixed,fixed_par,fixed_par_val):
    c_sys = pd.DataFrame(dataframe['c_sys [bEUR]'])
    c_sys['weatheryear'] = dataframe['weatheryear']
    c_sys.sort_values(by='weatheryear',inplace=True)
    c_sys_fixed = pd.DataFrame(fixed.loc[fixed[fixed_par][fixed[fixed_par] == fixed_par_val].index]['c_sys [bEUR]']) # Define reference (here one that does not include storage-X)
    c_sys_fixed['weatheryear'] = fixed.loc[c_sys_fixed.index].weatheryear
    c_sys_normalized = pd.concat([c_sys[c_sys.weatheryear == c_sys_fixed.weatheryear[i]]['c_sys [bEUR]']/c_sys_fixed['c_sys [bEUR]'][i] for i in range(len(c_sys_fixed))])
    return c_sys_normalized

def read_simple_parametric(path,sector):
    filename = path + 'sspace_eta_c.csv'
    df2 = pd.read_csv(filename,index_col=0).T
    df2['sector'] = df2['sector'].fillna('-')
    df2 = df2.query('sector == @sector')
    df2 = df2.drop(columns='sector').astype(float).sort_values(by='eta1 [-]').T
    
    df2 = df2.T
    df2['E_el [GWh]'] = df2['E [GWh]']*df2['eta2 [-]']
    df2['duration'] = df2['E_el [GWh]']/df2['G_discharge [GW]']
    fixed = df2
    fixed_par = 'eta1 [-]'
    fixed_par_val = 0.3
    df2['c_sys_reduction'] = normalize_system_cost(df2,fixed,fixed_par,fixed_par_val)
    
    filename = path + 'sspace_eta_d.csv'
    df3 = pd.read_csv(filename,index_col=0).T
    df3['sector'] = df3['sector'].fillna('-')
    df3 = df3.query('sector == @sector')
    df3 = df3.drop(columns='sector').astype(float).sort_values(by='eta2 [-]').T
    
    df3 = df3.T
    df3['E_el [GWh]'] = df3['E [GWh]']*df3['eta2 [-]']
    df3['duration'] = df3['E_el [GWh]']/df3['G_discharge [GW]']
    df3['c_sys_reduction'] = normalize_system_cost(df3,fixed,fixed_par,fixed_par_val)
    df3 = df3.T
    
    filename = path + 'sspace_c_c.csv' 
    df4 = pd.read_csv(filename,index_col=0).T
    df4['sector'] = df4['sector'].fillna('-')
    df4 = df4.query('sector == @sector')
    df4 = df4.drop(columns='sector').astype(float).sort_values(by='c1').T
    
    df4 = df4.T
    df4['E_el [GWh]'] = df4['E [GWh]']*df4['eta2 [-]']
    df4['duration'] = df4['E_el [GWh]']/df4['G_discharge [GW]']
    df4['c_sys_reduction'] = normalize_system_cost(df4,fixed,fixed_par,fixed_par_val)
    df4 = df4.T
    
    filename = path + 'sspace_c_d.csv'
    df5 = pd.read_csv(filename,index_col=0).T
    df5['sector'] = df5['sector'].fillna('-')
    df5 = df5.query('sector == @sector')
    df5 = df5.drop(columns='sector').astype(float).sort_values(by='c2').T
    
    df5 = df5.T
    df5['E_el [GWh]'] = df5['E [GWh]']*df5['eta2 [-]']
    df5['duration'] = df5['E_el [GWh]']/df5['G_discharge [GW]']
    df5['c_sys_reduction'] = normalize_system_cost(df5,fixed,fixed_par,fixed_par_val)
    df5 = df5.T
    
    filename = path + 'sspace_chat.csv' 
    df6 = pd.read_csv(filename,index_col=0).T
    df6['sector'] = df6['sector'].fillna('-')
    df6 = df6.query('sector == @sector')
    df6 = df6.drop(columns='sector').astype(float).sort_values(by='c_hat [EUR/kWh]').T
    
    df6 = df6.T
    df6['E_el [GWh]'] = df6['E [GWh]']*df6['eta2 [-]']
    df6['duration'] = df6['E_el [GWh]']/df6['G_discharge [GW]']
    df6['c_sys_reduction'] = normalize_system_cost(df6,fixed,fixed_par,fixed_par_val)
    df6 = df6.T
    
    filename = path + 'sspace_tau.csv'
    df7 = pd.read_csv(filename,index_col=0).T
    df7['sector'] = df7['sector'].fillna('-')
    df7 = df7.query('sector == @sector')
    df7 = df7.drop(columns='sector').astype(float).sort_values(by='tau [n_days]').T
    
    df7 = df7.T
    df7['E_el [GWh]'] = df7['E [GWh]']*df7['eta2 [-]']
    df7['duration'] = df7['E_el [GWh]']/df7['G_discharge [GW]']
    df7['c_sys_reduction'] = normalize_system_cost(df7,fixed,fixed_par,fixed_par_val)
    df7 = df7.T
    df2 = df2.T
    
    return [df2,df3,df4,df5,df6,df7]
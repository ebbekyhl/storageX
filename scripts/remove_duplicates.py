# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:12:16 2023

@author: au485969
"""

import pandas as pd
file = '../results/sspace_w_sectorcoupling_merged.csv'
# file = '../results/sspace_w_sectorcoupling_wo_duplicates.csv'
sectors = ['T-H-I-B','T-H','-']
sspace_all = pd.read_csv(file,index_col=0)

#%%
sector = 'T-H-I-B'
print('Length of sspace is initially ', len(sspace_all.columns))
for sector in sectors:
    sspace = sspace_all.T
    sspace['sector'] = sspace['sector'].fillna('-')
    sspace = sspace.query('sector == @sector')
    sspace = sspace.drop(columns='sector').astype(float).T
    
    sspace_new = sspace.T[['c_hat [EUR/kWh]','c1','c2','eta1 [-]','eta2 [-]','tau [n_days]','E [GWh]']]
    sspace_new.columns = ['c_hat','c1','c2','eta1','eta2','tau','E']
    sspace_new_sorted = sspace_new.sort_values(by=['c_hat','c1','c2','eta1','eta2','tau'])[['c_hat','c1','c2','eta1','eta2','tau']]
    
    print('Omitting ', len(sspace_new_sorted[sspace_new_sorted.duplicated()].index), ' duplicate configurations')
    sspace_all = sspace_all.drop(columns=sspace_new_sorted[sspace_new_sorted.duplicated()].index)
    print('Length of sspace is now ', len(sspace_all.columns))
    
    # sspace_all.to_csv('../results/sspace_w_sectorcoupling_wo_duplicates.csv')
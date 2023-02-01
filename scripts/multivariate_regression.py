# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:16:50 2023

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
import statsmodels.api as sm

#%% Read data
file = '../Results/sspace_w_sectorcoupling_merged.csv'
# file = '../Results/sspace_3888.csv'

if file == '../Results/sspace_w_sectorcoupling_merged.csv':
    sectors = ['T-H-I-B','T-H','-']
    sector_names = ['Fully sector-coupled','Electricity \n + Heating \n + Land Transport', 'Electricity']
else:
    sectors = ['']
    sector_names = ['Electricity']

sspace_og = pd.read_csv(file,index_col=0)

# Remove eta1 > 1
# sspace_man = sspace_og.T.drop(columns='sector').astype(float)
# sspace_man = sspace_man['eta1 [-]'][sspace_man['eta1 [-]'] < 1]
# sspace_og = sspace_og[sspace_man.index]

params_E = np.zeros([len(sectors),5])
pvalues_E = np.zeros([len(sectors),5])
params_LC = np.zeros([len(sectors),5])
pvalues_LC = np.zeros([len(sectors),5])
params_SCR = np.zeros([len(sectors),5])
pvalues_SCR = np.zeros([len(sectors),5])

E_sector = pd.DataFrame(index=np.arange(1082))

# Multivariate regression using GLM for all sectors
i = 0
for sector in sectors:
    if file == '../Results/sspace_w_sectorcoupling_merged.csv':
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
    df1['SCR'] = sspace.loc['c_sys [bEUR]'].astype(float)/(sspace.loc['c_sys [bEUR]'].astype(float).max())
    
    df1 = df1.sort_values(['c_hat','c1','eta1','c2','eta2','tau_SD'])
    
    # Regression
    df_data = df1[['c_hat', 'c1','c2','eta1','eta2']] # "tau_SD" is here not included in the linear regression
    X = df_data
    X = (X - X.min())/(X.max()-X.min()) # We do a min-max scaling of the variables
    
    # E
    y = df1['E_cor'].values
    X_val = sm.add_constant(X.values)
    est = sm.GLM(y, X_val) # Using Generelized Linear Model for regression
    est2 = est.fit()
    # print(sector_names[i])
    # print(est2.summary())
    params_E[i,:] = (est2.params[1:]/(np.abs(est2.params[1:]).max()))
    pvalues_E[i,:] = est2.pvalues[1:]
    
    # LC
    y = df1['LC'].values
    X_val = sm.add_constant(X.values)
    est = sm.GLM(y, X_val) # Using Generelized Linear Model for regression
    est2 = est.fit()
    # print(sector_names[i])
    # print(est2.summary())
    params_LC[i,:] = (est2.params[1:]/(np.abs(est2.params[1:]).max()))
    pvalues_LC[i,:] = est2.pvalues[1:]
    
    # System cost reduction
    y = (1 - df1['SCR'].values)*100
    X_val = sm.add_constant(X.values)
    est = sm.GLM(y, X_val) # Using Generelized Linear Model for regression
    est2 = est.fit()
    # print(sector_names[i])
    # print(est2.summary())
    params_SCR[i,:] = (est2.params[1:]/(np.abs(est2.params[1:]).max()))
    pvalues_SCR[i,:] = est2.pvalues[1:]
    
    i += 1

#%% Plotting coefficients of the GLM estimating "E_cor"
fig,ax = plt.subplots(figsize=[10,6])
extent = [0, 5, 0, len(sectors)]
im = ax.imshow(np.array([params_E[2],params_E[1],params_E[0]]), cmap="cool",extent=extent)
ax.set_xticks(np.arange(5)+0.5)
ax.set_xticklabels([r'$\hat{c}$', r'$c_c$',r'$c_d$',r'$\eta_c$',r'$\eta_d$'],fontsize=18)
ax.set_yticks(np.arange(len(sectors))+0.5)
ax.set_yticklabels(sector_names,fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

for i in range(len(sectors)):
    for j in range(5):
        ax.text(0.5+j,0.5+i,str(params_E[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center')
        pvalue_float = pvalues_E[i,j]
        if pvalue_float < 1e-3:
            pvalue_str = 'p < ' + r'$10^{-3}$'
        else:
            pvalue_str = 'p = ' + str(pvalue_float.round(3))
        ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
# fig.colorbar(im)
fig.savefig('../figures/GLM_coefficients_sectors_E.png',
            bbox_inches="tight",dpi=300)

#%% Plotting coefficients of the GLM estimating "LC"
fig,ax = plt.subplots(figsize=[10,6])
extent = [0, 5, 0, len(sectors)]
im = ax.imshow(np.array([params_LC[2],params_LC[1],params_LC[0]]), cmap="cool",extent=extent)
ax.set_xticks(np.arange(5)+0.5)
ax.set_xticklabels([r'$\hat{c}$', r'$c_c$',r'$c_d$',r'$\eta_c$',r'$\eta_d$'],fontsize=18)
ax.set_yticks(np.arange(len(sectors))+0.5)
ax.set_yticklabels(sector_names,fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

for i in range(len(sectors)):
    for j in range(5):
        ax.text(0.5+j,0.5+i,str(params_LC[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center')
        pvalue_float = pvalues_LC[i,j]
        if pvalue_float < 1e-3:
            pvalue_str = 'p < ' + r'$10^{-3}$'
        else:
            pvalue_str = 'p = ' + str(pvalue_float.round(3))
        ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
# fig.colorbar(im)
fig.savefig('../figures/GLM_coefficients_sectors_LC.png',
            bbox_inches="tight",dpi=300)

#%% Plotting coefficients of the GLM estimating "SCR"
fig,ax = plt.subplots(figsize=[10,6])
extent = [0, 5, 0, len(sectors)]
im = ax.imshow(np.array([params_SCR[2],params_SCR[1],params_SCR[0]]), cmap="cool",extent=extent)
ax.set_xticks(np.arange(5)+0.5)
ax.set_xticklabels([r'$\hat{c}$', r'$c_c$',r'$c_d$',r'$\eta_c$',r'$\eta_d$'],fontsize=18)
ax.set_yticks(np.arange(len(sectors))+0.5)
ax.set_yticklabels(sector_names,fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

for i in range(len(sectors)):
    for j in range(5):
        ax.text(0.5+j,0.5+i,str(params_SCR[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center')
        pvalue_float = pvalues_LC[i,j]
        if pvalue_float < 1e-3:
            pvalue_str = 'p < ' + r'$10^{-3}$'
        else:
            pvalue_str = 'p = ' + str(pvalue_float.round(3))
        ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
# fig.colorbar(im)
fig.savefig('../figures/GLM_coefficients_sectors_SCR.png',
            bbox_inches="tight",dpi=300)
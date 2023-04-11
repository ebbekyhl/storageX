# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:41:10 2021
@author: au485969
"""

#################### SCRIPT USED TO REDUCE THE SAMPLE SPACE ###################
# The initial storage-X sample space consists of 2,016 configurations. 
# Ideally, these would be used as inputs in the PyPSA-Eur-Sec model to evaluate 
# the market potential of each single configuration. This would have to be repeated
# for the three system compositions: 
    # SC1: Electricity only
    # SC2: Coupling with land transport and heating sectors
    # SC3: Fully sector-coupled
    
# As it is computationally heavy to run 3 x 2,016 scenarios, we reduce the initial 
# sample space based on the first iteration of 2,016 calculations for SC1 since 
# this can already identify storage that are lacking too much either in cost or 
# efficiency to be considered in the cost-optimal system design. 
# This script is used to make this reduction.

#%% Import modules and standardize figure layout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

fs = 18
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.title_fontsize'] = fs

#%% Read csv file containing results of initial sample space for SC1
visual = '_threshold' # ''
visual_2 = '' # 'numbers'
colorcode = 'round-trip'
path = '../results/'
filename = 'sspace_2016.csv' # .csv file containing results for 2,016 configurations 
df1 = pd.read_csv(path + filename,index_col=0)
df1_T = df1.T.sort_values(by='c_hat [EUR/kWh]')
df1_T['E [GWh]'] = df1_T['E [GWh]']*1e-3 # convert GWh to TWh
df1_T.rename(columns={'E [GWh]':'E [TWh]'}, inplace=True)
# df1_T = df1_T.loc[df1_T['eta1 [-]'][df1_T['eta1 [-]'] < 1].index]
df1 = df1_T.reset_index(drop=True).T
#%% Fitting
threshold_E = 2 # Threshold of 2 TWh energy capacity of storage-X
threshold_L = 0 # We do not impose any requirement of load coverage
df_update = df1.T[df1.T['eta2 [-]']*df1.T['E [TWh]'] >= threshold_E]
tau = df_update['tau [n_days]']
RTE = df_update['eta1 [-]']*df_update['eta2 [-]']
LEN = []
LAM1 = [] # list of lambda1s
LAM2 = [] # list of lambda2s

lambdaa1_min = 50 # <------ These were estimated from an iterative search
lambdaa1_max = 70 # <------ These were estimated from an iterative search
lambdaa2_min = 160 # <------ These were estimated from an iterative search
lambdaa2_max = 180 # <------ These were estimated from an iterative search

delta = 1 # Step size

# Introduce metric M and fit it to the results by sweeping across a range of lambda1s and lambda2s 
for lambdaa1 in np.arange(lambdaa1_min,lambdaa1_max,delta):
    for lambdaa2 in np.arange(lambdaa2_min,lambdaa2_max,delta):
            # M_succesful is the "M" metric calculated only based on the configurations fulfilling E >= 2 TWh
            M_succesful = ((df_update['c1']/lambdaa1+df_update['c2']/lambdaa2) + df_update['c_hat [EUR/kWh]']) / (RTE)
            # M is the "M" metric calculated based on all configurations
            M = ((df1.T['c1']/lambdaa1+df1.T['c2']/lambdaa2) + df1.T['c_hat [EUR/kWh]']) / (df1.T['eta1 [-]']*df1.T['eta2 [-]'])
            LAM1.append(lambdaa1)
            LAM2.append(lambdaa2)
            LEN.append(len(M[M <= M_succesful.max()]) - len(M_succesful)) # Error
#%% Find the lambdas leading to the best fit
A = np.array(LEN)
B = np.array(LAM1)
C = np.array(LAM2)

A_pos = A[A >= 0]
B_pos = B[A >= 0]
C_pos = C[A >= 0]

lambda1_o = B_pos[A_pos == A_pos.min()].mean()
lambda2_o = C_pos[A_pos == A_pos.min()].mean()
#%% Calculate the maximum M value at E = 2 TWh
df = pd.DataFrame([M, df1.loc['E [TWh]']]).T
df.columns = ['M', 'E']
M_max = df.loc[df['E'][df['E']>=2].index]['M'].max()

# Exclude configurations which has M > max(M(E=2))
M = (df1.T['c1']/lambda1_o + df1.T['c2']/lambda2_o + df1.T['c_hat [EUR/kWh]']) / (df1.T['eta1 [-]']*df1.T['eta2 [-]'])
filt = M[M <= M_max]
#%% Plot the reduction of the sample space
fig,ax = plt.subplots()
ax.plot(M.values,df1.loc['E [TWh]'].values,'.',markersize=4,alpha=0.4)
ax.axvline(M_max,ls='--',color='k')
ax.axhline(2,(M_max+0)/(250+0),1,ls='--',color='k')
ax.set_ylim([-0.3,20])
ax.set_xlim([0,250])
ax.set_ylabel('$E$ [TWh]')

ydelta = 0.5
ax.text(75,2+ydelta,'$E$ = 2 TWh',fontsize=16)
ax.text(50,10,r'$\max{\{M(E=2)\}}$',fontsize=16,rotation=90)
ax.set_xlabel('$M$ [-]')
ax.fill_between([0,M_max],[-1,-1],[20,20],color='lightgrey',zorder=0,label='Reduced sample space')
ax.legend(prop={'size':16})
fig.savefig('../figures/Filter.png',dpi=600,bbox_inches="tight")

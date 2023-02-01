# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:41:10 2021
@author: au485969
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import multiprocessing as mp
#%%
visual = '_threshold' # ''
visual_2 = '' # 'numbers'
colorcode = 'round-trip'

path = '../results/'
filename = path + '../results/sspace.csv'

df1 = pd.read_csv(path + 'sspace.csv',index_col=0)
df1_T = df1.T.sort_values(by='c_hat [EUR/kWh]')
df1_T['E [GWh]'] = df1_T['E [GWh]']*1e-3 # convert GWh to TWh
df1_T.rename(columns={'E [GWh]':'E [TWh]'}, inplace=True)
df1_T.loc[df1_T['eta1 [-]'][df1_T['eta1 [-]'] < 1].index]
df1 = df1_T.reset_index(drop=True).T
#%%
threshold_E = 2
threshold_L = 0

df_update = df1.T[df1.T['eta2 [-]']*df1.T['E [TWh]'] >= threshold_E]
tau = df_update['tau [n_days]']
RTE = df_update['eta1 [-]']*df_update['eta2 [-]']

LEN = []
LAM1 = []
LAM2 = []

for lambdaa1 in np.arange(50,70,1):
    for lambdaa2 in np.arange(160,180,1):
            XCOS = ((df_update['c1']/lambdaa1+df_update['c2']/lambdaa2) + df_update['c_hat [EUR/kWh]']) / (RTE)
            xcos = ((df1.T['c1']/lambdaa1+df1.T['c2']/lambdaa2) + df1.T['c_hat [EUR/kWh]']) / (df1.T['eta1 [-]']*df1.T['eta2 [-]'])
            LAM1.append(lambdaa1)
            LAM2.append(lambdaa2)
            LEN.append(len(xcos[xcos <= XCOS.max()]) - len(XCOS))
#%%
A = np.array(LEN)
B = np.array(LAM1)
C = np.array(LAM2)

A_pos = A[A >= 0]
B_pos = B[A >= 0]
C_pos = C[A >= 0]

lambda1_o = B_pos[A_pos == A_pos.min()].mean()
lambda2_o = C_pos[A_pos == A_pos.min()].mean()
#%%
df = pd.DataFrame([xcos, df1.loc['E [TWh]']]).T
df.columns = ['M', 'E']
M_max = df.loc[df['E'][df['E']>=2].index]['M'].max()

filtered = (df1.T['c1']/lambda1_o + df1.T['c2']/lambda2_o + df1.T['c_hat [EUR/kWh]']) / (df1.T['eta1 [-]']*df1.T['eta2 [-]'])
filt = filtered[filtered <= M_max]
#%%
fig,ax = plt.subplots()
ax.plot(xcos.values,df1.loc['E [TWh]'].values,'.',markersize=4,alpha=0.4)
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
fig.savefig('../figures/Filter.png',dpi=300,bbox_inches="tight")

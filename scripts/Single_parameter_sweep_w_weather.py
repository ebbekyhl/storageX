# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 08:31:10 2022

@author: au485969
"""
from read_simple_parametric import read_simple_parametric
from plot_single_sweep import plot_single_sweep
import matplotlib.pyplot as plt
# from tech_colors import tech_colors
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
plt.close('all')
fs = 18
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

variable = 'E'
# variable = 'G_discharge'
# variable = 'lc'
# variable = 'duration'
# variable = 'system_cost'

output_dic = {'E':'E_el [GWh]',
              'G_charge':'G_charge [GW]',
              'G_discharge':'G_discharge [GW]',
              'lc':'load_coverage [%]',
              'duration':'duration',
              'system_cost':'c_sys_reduction'}

output_text_dic = {'E':'Energy capacity ' + r'$E$' + ' [TWh]' ,
                   'G_charge':' Power capacity ' + r'$G_c$' + ' [GW]',
                   'G_discharge': ' Power capacity ' + r'$G_d$' +' [GW]',
                   'lc':'Load coverage ' + r'$LC$' +' [%]',
                   'duration':'Duration [h]',
                   'system_cost':'System cost [-]'}

factor_dic = {'E':1e3,
              'G_charge':1,
              'G_discharge': 1,
              'lc':1,
              'duration':1,
              'system_cost':1}

output = output_dic[variable]
output_text = output_text_dic[variable]
factor = factor_dic[variable]

#%% Plotting
path = '../results/simple_sweep/'
fs = 18
lw1 = 3
ms1 = 10
sm = []
fig1,axes = plt.subplots(2,3,figsize=[14,8],sharey=True)
for sector in ['-','T-H','T-H-I-B']:
    [df2,df3,df4,df5,df6,df7] = read_simple_parametric(path,sector)
    dfs = [df2,df3,df4,df5,df6,df7]
    sm1 = plot_single_sweep(fig1,axes,sector,output,output_text,factor,fs,lw1,ms1,dfs)
    
    sm.append(sm1[0])
#%%
path = '../results/weather_sensitivity/'

count = 0
count_sector = 0

hdic = {'-':'//',
        'T-H':'\\',
        'T-H-I-B':'--'}

fcolor = 'lightgrey'
ecolor_dic = {'-':'green',
              'T-H':'blue',
              'T-H-I-B':'orange'}

sm3 = []

df2_df_w_sector = pd.DataFrame()
df3_df_w_sector = pd.DataFrame()
df4_df_w_sector = pd.DataFrame()
df5_df_w_sector = pd.DataFrame()
df6_df_w_sector = pd.DataFrame()
df7_df_w_sector = pd.DataFrame()

color_y_dic = {2002: 'red',
               2010: 'purple',
               2003: 'blue',
               2008: 'green',
               2006: 'orange'}

# Weather year and sector
for sector in ['-','T-H','T-H-I-B']:
    
    sm4 = []
    
    df2_df = pd.DataFrame()
    df3_df = pd.DataFrame()
    df4_df = pd.DataFrame()
    df5_df = pd.DataFrame()
    df6_df = pd.DataFrame()
    df7_df = pd.DataFrame()
    
    [df2,df3,df4,df5,df6,df7] = read_simple_parametric(path,sector)
    
    df_list = [df2,df3,df4,df5,df6,df7]

    wys = df2.loc['weatheryear'].unique()
    
    for wy in wys:
        
        df2_i = df2[df2.loc['weatheryear'][df2.loc['weatheryear'] == wy].index]
        df3_i = df3[df3.loc['weatheryear'][df3.loc['weatheryear'] == wy].index]
        df4_i = df4[df4.loc['weatheryear'][df4.loc['weatheryear'] == wy].index]
        df5_i = df5[df5.loc['weatheryear'][df5.loc['weatheryear'] == wy].index]
        df6_i = df6[df6.loc['weatheryear'][df6.loc['weatheryear'] == wy].index]
        df7_i = df7[df7.loc['weatheryear'][df7.loc['weatheryear'] == wy].index]
        
        df2_df = pd.concat([df2_df,pd.DataFrame([df2_i.loc['eta1 [-]'],df2_i.loc[output],df2_i.loc['weatheryear']]).T])
        df3_df = pd.concat([df3_df,pd.DataFrame([df3_i.loc['eta2 [-]'],df3_i.loc[output],df3_i.loc['weatheryear']]).T])
        df4_df = pd.concat([df4_df,pd.DataFrame([df4_i.loc['c1'],df4_i.loc[output],df4_i.loc['weatheryear']]).T])
        df5_df = pd.concat([df5_df,pd.DataFrame([df5_i.loc['c2'],df5_i.loc[output],df5_i.loc['weatheryear']]).T])
        df6_df = pd.concat([df6_df,pd.DataFrame([df6_i.loc['c_hat [EUR/kWh]'],df6_i.loc[output],df6_i.loc['weatheryear']]).T])
        df7_df = pd.concat([df7_df,pd.DataFrame([df7_i.loc['tau [n_days]'],df7_i.loc[output],df7_i.loc['weatheryear']]).T])
        
        df2_df_w_sector = pd.concat([df2_df_w_sector,pd.DataFrame([df2_i.loc['eta1 [-]'],df2_i.loc[output],df2_i.loc['weatheryear'],pd.Series([sector]*df2_i.shape[1],index=df2_i.columns,name='sector')]).T])
        df3_df_w_sector = pd.concat([df3_df_w_sector,pd.DataFrame([df3_i.loc['eta2 [-]'],df3_i.loc[output],df3_i.loc['weatheryear'],pd.Series([sector]*df3_i.shape[1],index=df3_i.columns,name='sector')]).T])
        df4_df_w_sector = pd.concat([df4_df_w_sector,pd.DataFrame([df4_i.loc['c1'],df4_i.loc[output],df4_i.loc['weatheryear'],pd.Series([sector]*df4_i.shape[1],index=df4_i.columns,name='sector')]).T])
        df5_df_w_sector = pd.concat([df5_df_w_sector,pd.DataFrame([df5_i.loc['c2'],df5_i.loc[output],df5_i.loc['weatheryear'],pd.Series([sector]*df5_i.shape[1],index=df5_i.columns,name='sector')]).T])
        df6_df_w_sector = pd.concat([df6_df_w_sector,pd.DataFrame([df6_i.loc['c_hat [EUR/kWh]'],df6_i.loc[output],df6_i.loc['weatheryear'],pd.Series([sector]*df6_i.shape[1],index=df6_i.columns,name='sector')]).T])
        df7_df_w_sector = pd.concat([df7_df_w_sector,pd.DataFrame([df7_i.loc['tau [n_days]'],df7_i.loc[output],df7_i.loc['weatheryear'],pd.Series([sector]*df7_i.shape[1],index=df7_i.columns,name='sector')]).T])
        
        count += 1
        
    if variable == 'E':
        ymin = 0
        ymax = 8
        
    elif variable == 'lc':    
        ymin = 0
        ymax = 1.5
        
    elif variable == 'duration':
        ymin = 0
        ymax = 150
        
    if variable == 'system_cost':
        ymin = 0.98
        ymax = 1.01
        
    if variable == 'G_discharge':
        ymin = 0
        ymax = 200
    
    for ax in axes.flatten():
        ax.set_ylim([ymin,ymax])
        
    sm3.append(axes[0,0].fill_between(df2_df['eta1 [-]'].unique()*100,df2_df.groupby('eta1 [-]').min()[output].values/factor,df2_df.groupby('eta1 [-]').max()[output].values/factor,facecolor=fcolor,hatch=hdic[sector],alpha=0.5,edgecolor=ecolor_dic[sector],label=sector))
    axes[0,1].fill_between(df7_df['tau [n_days]'].unique(),df7_df.groupby('tau [n_days]').min()[output].values/factor,df7_df.groupby('tau [n_days]').max()[output].values/factor,facecolor=fcolor,hatch=hdic[sector],alpha=0.5,edgecolor=ecolor_dic[sector])
    axes[0,2].fill_between(df3_df['eta2 [-]'].unique()*100,df3_df.groupby('eta2 [-]').min()[output].values/factor,df3_df.groupby('eta2 [-]').max()[output].values/factor,facecolor=fcolor,hatch=hdic[sector],alpha=0.5,edgecolor=ecolor_dic[sector])
    axes[1,0].fill_between(490 - df4_df['c1'].unique(),df4_df.groupby('c1').min()[output].values/factor,df4_df.groupby('c1').max()[output].values/factor,facecolor=fcolor,hatch=hdic[sector],alpha=0.5,edgecolor=ecolor_dic[sector])
    axes[1,1].fill_between(20 - df6_df['c_hat [EUR/kWh]'].unique(),df6_df.groupby('c_hat [EUR/kWh]').min()[output].values/factor,df6_df.groupby('c_hat [EUR/kWh]').max()[output].values/factor,facecolor=fcolor,hatch=hdic[sector],alpha=0.5,edgecolor=ecolor_dic[sector])
    axes[1,2].fill_between(490 - df5_df['c2'].unique(),df5_df.groupby('c2').min()[output].values/factor,df5_df.groupby('c2').max()[output].values/factor,facecolor=fcolor,hatch=hdic[sector],alpha=0.5,edgecolor=ecolor_dic[sector])
    
    count_sector += 1

sms = sm + sm3

fig1.legend([sms[0],sms[3],sms[1],sms[4],sms[2],sms[5]],
             ['Electricity 2013','Electricity (2002-2010)',
              'Electricity + Heating + Land Transport 2013','Electricity + Heating + Land Transport (2002-2010)',
              'Fully sector-coupled 2013','Fully sector-coupled (2002-2010)'], bbox_to_anchor=(0, -0.10), loc=3,
            ncol=3,frameon=True,prop={'size': fs},borderaxespad=0)

fig1.tight_layout() 
fig1.savefig('../figures/Storage_parameter_hierarchi_' + variable + '_absolute_values_subplot_weather_and_sector.pdf', transparent=True,
            bbox_inches="tight")

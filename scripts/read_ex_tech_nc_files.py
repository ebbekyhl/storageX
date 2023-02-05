# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:00:00 2022

@author: au485969
"""
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
from glob import glob

fs = 16
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

def calculate_load_coverage(network,x_dic,PyPSA_Eur_Sec_v):
    n = network.copy()
    buses = n.buses.query('carrier == "AC"').index
    links = n.links

    storage_discharge_t = -n.links_t.p1[n.links[n.links.carrier == x_dic[PyPSA_Eur_Sec_v][0]].index].sum().sum() # MWe (after conversion to electricity)

    xss_0 = [links.bus0[links.bus0 == i].loc[[i in links.bus0[links.bus0 == i].index[j] for j in range(len(links.bus0[links.bus0 == i].index))]].drop([i + ' ' + x_dic[PyPSA_Eur_Sec_v][1],i + ' battery charger',i + ' electricity distribution grid']).index.tolist() for i in buses]
    loads_0 = [x for xs in xss_0 for x in xs] # links using electricity from HVAC bus
    loads_1 = [links[links.index == i + ' electricity distribution grid'].index.item() for i in buses]
    loads_t_0 = n.links_t.p0[loads_0].sum().sum()/1e6*8760/n.snapshots.shape[0]
    loads_t_1 = n.links_t.p0[loads_1][n.links_t.p0[loads_1] > 0].fillna(0).sum().sum()/1e6*8760/n.snapshots.shape[0]
    loads_t_2 = n.loads_t.p[n.loads.query('carrier == "industry electricity"').index].sum().sum()/1e6*8760/n.snapshots.shape[0]

    loads_sum = loads_t_0 + loads_t_1 + loads_t_2

    xss1 = [links.bus1[links.bus1 == i].loc[[i in links.bus1[links.bus1 == i].index[j] for j in range(len(links.bus1[links.bus1 == i].index))]].drop([i + ' ' + x_dic[PyPSA_Eur_Sec_v][0],i + ' battery discharger']).index.tolist() for i in buses]
    supply_0 = [x for xs in xss1 for x in xs] # links supplying electricity to HVAC bus
    supply_t_0 = -n.links_t.p1[supply_0].sum().sum()/1e6*8760/n.snapshots.shape[0]
    supply_t_1 = -n.links_t.p1[loads_1][n.links_t.p1[loads_1] > 0].fillna(0).sum().sum()/1e6*8760/n.snapshots.shape[0]
    supply_t_2 = n.generators_t.p[n.generators.query('carrier == ["offwind-ac","offwind-dc","onwind","ror","solar","solar rooftop"]').index].sum().sum()/1e6*8760/n.snapshots.shape[0]
    supply_t_3 = n.storage_units_t.p[n.storage_units.query('carrier == "hydro"').index].sum().sum()/1e6*8760/n.snapshots.shape[0]

    supply_sum = supply_t_0 + supply_t_1 + supply_t_2 + supply_t_3

    if ('land transport EV' in n.loads.carrier.unique()) and ('industry electricity' not in n.loads.carrier.unique()):
        load_coverage = 8760/n.snapshots.shape[0]*storage_discharge_t.sum()/(max(supply_sum,loads_sum)*1e6)

    elif ('land transport EV' in n.loads.carrier.unique()) and ('industry electricity' in n.loads.carrier.unique()):
        load_coverage = 8760/n.snapshots.shape[0]*storage_discharge_t.sum()/(max(supply_sum,loads_sum)*1e6)

    else:
        load_coverage = storage_discharge_t.sum()/(n.loads_t.p.sum().sum())

    return load_coverage

PyPSA_Eur_Sec_v = '0.4.0'
# PyPSA_Eur_Sec_v = '0.6.0'

x_dic = {'0.4.0':["X Discharge",'X Charge','X'],
         '0.6.0':["Xdischarge",'Xcharge','X']}

files = glob('PyPSA-Eur-Sec-' + PyPSA_Eur_Sec_v + '/*.nc')

csv = pd.read_csv('storage_techs_cost_USD.csv',sep=';',index_col=0)
costs = csv[['c_c','eta_c','c_d','eta_d','c_hat']]
costs['c_c']/=350
costs['c_d']/=350
costs['eta_c'] /= 50
costs['eta_d'] /= 50
costs['c_hat'] /= 20

costs = costs.round(2)

tres = '3H'
# tres = '1H'

ex_tech_dic = {# c_hat,    c_c,     c_d,   eta_c,  eta_d
                "['1.35', '0.93', '1.87', '1.84', '1.30']":'aCAES',
                "['1.60', '1.67', '1.67', '1.54', '1.30']":'LAES',
                "['0.40', '0.11', '2.57', '1.96', '0.76']":'TES',
                "['1.00', '0.97', '1.94', '4.40', '0.50']":'PTES',
                "['0.90', '0.31', '3.09', '1.98', '0.86']":'MSES',
                "['6.00', '0.52', '0.52', '1.70', '1.70']":'RFB',
                "['0.55', '1.34', '3.27', '1.36', '1.00']":'H2'
                }

ren_res = {'0.4.0':'',
           '0.6.0':'370'}

obj_ref = {'37':747488065064,
           '74':754678961550}

i = 0
X_EUs = {}
Ns = ['37','74']
for N in Ns:
    X_EU = pd.DataFrame(index=['E [MWh_e]','G_c [MW_e]','G_d [MW_e]','Duration [h]','Load coverage [%]','Sys_cost [bEur]'])
    for file in files:
        # print(i)
        condition = file.split('\\')[1].startswith('elec_s370_' + N + '_lv1.0__Co2L0.05-' + tres) if PyPSA_Eur_Sec_v == '0.6.0' else file.split('\\')[1].startswith('elec_s_y2013_n' + N + '_lv1.0__Co2L0.05-' + tres)
        
        if not condition:
            continue 
        
        config = file.split('_')[-2].split('-')[-6:-1] if PyPSA_Eur_Sec_v == '0.6.0' else file.split('_')[-1].split('-')[-5:]
        
        sector = 'E' if 'T-H' in file.split('_')[-1] else 'SC1'
        sector = 'SC2' if 'T-H-I-B' in file.split('_')[-1] else sector
        sector = 'SC2' if PyPSA_Eur_Sec_v == '0.6.0' else sector
        
        print(sector)
        
        if PyPSA_Eur_Sec_v == '0.6.0':
            c_hat = config[0][-4:]
            c_c = config[1][-4:]
            c_d = config[2][-4:]
            eta_c = config[3][-4:]
            eta_d = config[4][-4:]
        else:
            c_hat = format(float(config[4][9:-3]),'.2f')
            c_c = format(float(config[1][-4:]),'.2f')
            c_d = format(float(config[3][-4:]),'.2f')
            eta_c = format(float(config[0][10:]),'.2f')
            eta_d = format(float(config[2][13:]),'.2f')
            
        config_list = str([c_hat,c_c,c_d,eta_c,eta_d])
        if config_list not in ex_tech_dic.keys():
            continue
            
        ex_tech = ex_tech_dic[config_list]
        
        n = pypsa.Network(file)
        
        string_d = x_dic[PyPSA_Eur_Sec_v][0]
        string_c = x_dic[PyPSA_Eur_Sec_v][1]
        string_x = x_dic[PyPSA_Eur_Sec_v][2]
        
        X_dischargers = n.links.query('carrier == @string_d')
        X_chargers = n.links.query('carrier == @string_c')
        X_stores = n.stores.query('carrier == @string_x')
        
        G_c_n = X_chargers.p_nom_opt
        G_d_n_0 = X_dischargers.p_nom_opt
        G_d_n_1 = G_d_n_0*X_dischargers.efficiency.unique().item()
        E_n_0 = X_stores.e_nom_opt
        E_n_1 = E_n_0*X_dischargers.efficiency.unique().item()
        
        duration = E_n_0/G_d_n_0.values
        
        # system_cost_percentage_change = (n.objective - obj_ref[N])/obj_ref[N]*100
        system_cost = n.objective/1e9
        
        load_coverage = calculate_load_coverage(n,x_dic,PyPSA_Eur_Sec_v)
        
        X_EU[ex_tech+'-'+sector] = [round(E_n_1.sum(),2),
                                    round(G_c_n.sum(),1),
                                    round(G_d_n_1.sum(),1),
                                    round(duration.mean(),1),
                                    round(load_coverage,1),
                                    round(system_cost,1)
                                    ]
        i += 1
        
    # print(X_EU)
    X_EUs[N] = X_EU
    
#%%
color_dic = {'E':'green',
             'SC1':'blue',
             'SC2':'orange'}

techs = ['H2', 'LAES', 'MSES', 'PTES', 'RFB', 'TES', 'aCAES']
# techs = ['H2', 'PTES', 'TES', 'aCAES']

output = 'E [MWh_e]'
# output = 'G_d [MW_e]'

output_ylab_dic = {'E [MWh_e]':'Energy capacity [GWh' + r'$_e$' + ']',
              'G_d [MW_e]':'Power capacity [GW' + r'$_e$' + ']',}

output_dic = {'E [MWh_e]':'E',
              'G_d [MW_e]':'G_d',}

wi = 0.1
tech = {}
fig,ax = plt.subplots(figsize=[7,12])
count_i = 0
cols = list(X_EUs['37'].columns)
cols.sort()
leg = {}
for tech in techs:
    # print(count_i)
    count_j = 0
    for col in cols:
        if tech == col.split('-')[0]:
            print(count_i+wi*count_j)
            y_val = X_EUs['37'][col][output]/1e3
            leg[col.split('-')[1]] = ax.barh(0.5*count_i+wi*count_j-wi,y_val,height=wi,color=color_dic[col.split('-')[1]])
            count_j += 1
    
    count_i += 1

ax.set_yticks(0.5*np.arange(len(techs)))
ax.set_yticklabels(techs)

ax.set_xlabel(output_ylab_dic[output])

ax.set_xscale('log')
ax.grid(axis='x',which='minor')

ax.legend(list(leg.values()),['Electricity','+ Heating + Land Transport','Fully sector-coupled'],prop={'size':fs})

ax.set_xlim([0,2500])
ax.axvline(2000,color='grey',linewidth=3)

ax.annotate('E ' + r'$\geq$' + ' 2TWh', 
            xy=(0.98, 0.5),  
            xytext=(0.65, 0.5), 
            xycoords=ax.transAxes,
            #textcoords='axes fraction',
            arrowprops=dict(facecolor='grey',color='grey', shrink=0.05),
            fontproperties = {'size':fs},
            color = 'grey',
            horizontalalignment='left', 
            verticalalignment='center',
            )

fig.savefig('Existing_techs_output_' + output_dic[output] + '.png', dpi=300,
             bbox_inches="tight")
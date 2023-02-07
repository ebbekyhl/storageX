# Import packages
import pandas as pd
from six import iteritems
import numpy as np
import pypsa
# from six import iteritems
import os
import warnings
warnings.filterwarnings('ignore')

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus4"] = ["string",np.nan,np.nan,"4th bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency4"] = ["static or series","per unit",1.,"4th bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]
override_component_attrs["Link"].loc["p4"] = ["series","MW",0.,"4th bus output","Output"]
override_component_attrs["StorageUnit"].loc["p_dispatch"] = ["series","MW",0.,"Storage discharging.","Output"]
override_component_attrs["StorageUnit"].loc["p_store"] = ["series","MW",0.,"Storage charging.","Output"]

def make_summaries2(n):
    e_X = n.stores_t.e[n.stores[n.stores.carrier == 'X'].index] #/(n.stores[n.stores.carrier == 'H2'].e_nom_opt) # Filling level 
    e_X['EU X Store'] = n.stores_t.e[n.stores[n.stores.carrier == 'X'].index].sum(axis=1) #/((n.stores[n.stores.carrier == 'X'].e_nom_opt).sum())
    e_X_e_cap = n.stores.e_nom_opt[n.stores[n.stores.carrier == 'X'].index]
    
    e_battery = n.stores_t.e[n.stores[n.stores.carrier == 'battery'].index]
    e_battery['EU battery'] = n.stores_t.e[n.stores[n.stores.carrier == 'battery'].index].sum(axis=1)
    
    e_battery_e_cap = n.stores.e_nom_opt[n.stores[n.stores.carrier == 'battery'].index]
    e_homebattery_e_cap = n.stores.e_nom_opt[n.stores[n.stores.carrier == 'home battery'].index]

    e_battery_p_cap = n.links.p_nom_opt[n.links[n.links.carrier == 'battery discharger'].index]
    e_homebattery_p_cap = n.links.p_nom_opt[n.links[n.links.carrier == 'home battery discharger'].index]

    df_load = n.loads_t.p
    df_offwind_ac = n.generators_t.p[n.generators.index[n.generators.carrier == 'offwind-ac']]
    df_offwind_dc = n.generators_t.p[n.generators.index[n.generators.carrier == 'offwind-dc']]
    df_onwind = n.generators_t.p[n.generators.index[n.generators.carrier == 'onwind']]
    df_solar = n.generators_t.p[n.generators.index[n.generators.carrier == 'solar']]
    df_solar_roof = n.generators_t.p[n.generators.index[n.generators.carrier == 'solar rooftop']]
    df_ror = n.generators_t.p[n.generators.index[n.generators.carrier == 'ror']]
    df_hydro = n.storage_units_t.p[n.storage_units.index[n.storage_units.carrier == 'hydro']]
    df_phs = n.storage_units_t.p[n.storage_units.index[n.storage_units.carrier == 'PHS']]
    
    df_charge = n.links_t.p0[n.links[n.links.carrier == 'X Charge'].index] # Power (electricity) entering storage before conversion of energy
    df_H2_charge = n.links_t.p0[n.links[n.links.carrier == 'H2 Electrolysis'].index] # Power (electricity) entering storage before conversion of energy
    df_battery_charge = n.links_t.p0[n.links[n.links.carrier == 'battery charger'].index] 
    df_homebattery_charge = n.links_t.p0[n.links[n.links.carrier == 'home battery charger'].index]
    
    df_discharge = -n.links_t.p1[n.links[n.links.carrier == 'X Discharge'].index] # Power (electricity) leaving storage after conversion of energy
    df_gas_OCGT = -n.links_t.p1[n.links[n.links.carrier == 'OCGT'].index] # Active power (electricity) injected from gas power plants
    df_gas_CCGT = -n.links_t.p1[n.links[n.links.carrier == 'CCGT'].index] # Active power (electricity) injected from gas power plants
    df_coal = -n.links_t.p1[n.links[n.links.carrier == 'coal'].index] # Active power (electricity) injected from coal power plants
    df_nuclear = -n.links_t.p1[n.links[n.links.carrier == 'nuclear'].index] # Active power (electricity) injected from nuclear power plants
    df_gas_CHP_CC = -n.links_t.p1[n.links[n.links.carrier == 'urban central gas CHP CC'].index] # Active power (electricity) injected from CHP-CC power plants
    df_gas_CHP = -n.links_t.p1[n.links[n.links.carrier == 'urban central gas CHP'].index] # Active power (electricity) injected from CHP power plants
    df_biomass_CHP_CC = -n.links_t.p1[n.links[n.links.carrier == 'urban central solid biomass CHP CC'].index] # Active power (electricity) injected from CHP-CC power plants
    df_biomass_CHP = -n.links_t.p1[n.links[n.links.carrier == 'urban central solid biomass CHP'].index] # Active power (electricity) injected from CHP power plants

    df_battery = -n.links_t.p1[n.links[n.links.carrier == 'battery discharger'].index] 
    df_homebattery = -n.links_t.p1[n.links[n.links.carrier == 'home battery discharger'].index]
    
    load_df = pd.DataFrame()
    offwind_dc_df = pd.DataFrame()
    offwind_ac_df = pd.DataFrame()
    onwind_df = pd.DataFrame()
    solar_df = pd.DataFrame()
    solar_roof_df = pd.DataFrame()
    ror_df = pd.DataFrame()
    hydro_df = pd.DataFrame()
    phs_df = pd.DataFrame()
    X_df = pd.DataFrame()
    
    X_e_cap_df = pd.Series(index = countries)

    X_charge_df = pd.DataFrame()
    X_discharge_df = pd.DataFrame()
    H2_charge_df = pd.DataFrame()

    gas_df_OCGT = pd.DataFrame()
    gas_df_CCGT = pd.DataFrame()
    coal_df = pd.DataFrame()
    nuclear_df = pd.DataFrame()
    gas_CHP_CC_df = pd.DataFrame()
    gas_CHP_df = pd.DataFrame()
    biomass_CHP_CC_df = pd.DataFrame()
    biomass_CHP_df = pd.DataFrame()
    battery_df = pd.DataFrame()
    homebattery_df = pd.DataFrame()
    battery_charge_df = pd.DataFrame()

    e_battery_df = pd.DataFrame()
    
    battery_e_cap_df = pd.Series(index = countries)
    homebattery_e_cap_df = pd.Series(index = countries)
    battery_p_cap_df = pd.Series(index = countries)
    homebattery_p_cap_df = pd.Series(index = countries)

    homebattery_charge_df = pd.DataFrame()
    import_df = pd.DataFrame()
    links_DC_import_df = pd.DataFrame()

    links_to_dist_df = pd.DataFrame()
        
    for country_i in countries:
        matching_solar = [s for s in df_solar.columns if country_i in s]
        df_matching_solar = df_solar[set(matching_solar)]
        solar_df[country_i] = df_matching_solar.sum(axis=1)

        matching_solar_roof = [s for s in df_solar_roof.columns if country_i in s]
        df_matching_solar_roof = df_solar_roof[set(matching_solar_roof)]
        solar_roof_df[country_i] = df_matching_solar_roof.sum(axis=1)
        
        matching_onwind = [s for s in df_onwind.columns if country_i in s]
        df_matching_onwind = df_onwind[set(matching_onwind)]
        onwind_df[country_i] = df_matching_onwind.sum(axis=1)
        
        matching_offwind_ac = [s for s in df_offwind_ac.columns if country_i in s]
        df_matching_offwind_ac = df_offwind_ac[set(matching_offwind_ac)]
        offwind_ac_df[country_i] = df_matching_offwind_ac.sum(axis=1)
        
        matching_offwind_dc = [s for s in df_offwind_dc.columns if country_i in s]
        df_matching_offwind_dc = df_offwind_dc[set(matching_offwind_dc)]
        offwind_dc_df[country_i] = df_matching_offwind_dc.sum(axis=1)

        matching_ror = [s for s in df_ror.columns if country_i in s]
        df_matching_ror = df_ror[set(matching_ror)]
        ror_df[country_i] = df_matching_ror.sum(axis=1)
        
        matching_hydro = [s for s in df_hydro.columns if country_i in s]
        df_matching_hydro = df_hydro[set(matching_hydro)]
        hydro_df[country_i] = df_matching_hydro.sum(axis=1)

        matching_phs = [s for s in df_phs.columns if country_i in s]
        df_matching_phs = df_phs[set(matching_phs)]
        phs_df[country_i] = df_matching_phs.sum(axis=1)

        matching_X = [s for s in e_X.columns if country_i in s]
        df_matching_X = e_X[set(matching_X)]
        X_df[country_i] = df_matching_X.sum(axis=1)

        matching_e_battery = [s for s in e_battery.columns if country_i in s]
        df_matching_e_battery = e_battery[set(matching_e_battery)]
        e_battery_df[country_i] = df_matching_e_battery.sum(axis=1)

        matching_X_e_cap = [s for s in e_X_e_cap.index if country_i in s]
        df_matching_X_e_cap = e_X_e_cap[set(matching_X_e_cap)]
        X_e_cap_df.loc[country_i] = df_matching_X_e_cap.sum()

        matching_battery_e_cap = [s for s in e_battery_e_cap.index if country_i in s]
        df_matching_battery_e_cap = e_battery_e_cap[set(matching_battery_e_cap)]
        battery_e_cap_df.loc[country_i] = df_matching_battery_e_cap.sum()

        matching_homebattery_e_cap = [s for s in e_homebattery_e_cap.index if country_i in s]
        df_matching_homebattery_e_cap = e_homebattery_e_cap[set(matching_homebattery_e_cap)]
        homebattery_e_cap_df.loc[country_i] = df_matching_homebattery_e_cap.sum()

        matching_battery_p_cap = [s for s in e_battery_p_cap.index if country_i in s]
        df_matching_battery_p_cap = e_battery_p_cap[set(matching_battery_p_cap)]
        battery_p_cap_df.loc[country_i] = df_matching_battery_p_cap.sum()

        matching_homebattery_p_cap = [s for s in e_homebattery_p_cap.index if country_i in s]
        df_matching_homebattery_p_cap = e_homebattery_p_cap[set(matching_homebattery_p_cap)]
        homebattery_p_cap_df.loc[country_i] = df_matching_homebattery_p_cap.sum()

        matching_charge = [s for s in df_charge.columns if country_i in s]
        df_matching_charge = df_charge[set(matching_charge)]
        X_charge_df[country_i] = df_matching_charge.sum(axis=1)
        
        matching_discharge = [s for s in df_discharge.columns if country_i in s]
        df_matching_discharge = df_discharge[set(matching_discharge)]
        X_discharge_df[country_i] = df_matching_discharge.sum(axis=1)

        matching_H2_charge = [s for s in df_H2_charge.columns if country_i in s]
        df_matching_H2_charge = df_H2_charge[set(matching_H2_charge)]
        H2_charge_df[country_i] = df_matching_H2_charge.sum(axis=1)

        matching_battery = [s for s in df_battery.columns if country_i in s]
        df_matching_battery = df_battery[set(matching_battery)]
        battery_df[country_i] = df_matching_battery.sum(axis=1)
        
        matching_homebattery = [s for s in df_homebattery.columns if country_i in s]
        df_matching_homebattery = df_homebattery[set(matching_homebattery)]
        homebattery_df[country_i] = df_matching_homebattery.sum(axis=1)

        matching_battery_charge = [s for s in df_battery_charge.columns if country_i in s]
        df_matching_battery_charge = df_battery_charge[set(matching_battery_charge)]
        battery_charge_df[country_i] = df_matching_battery_charge.sum(axis=1)
        
        matching_homebattery_charge = [s for s in df_homebattery_charge.columns if country_i in s]
        df_matching_homebattery_charge = df_homebattery_charge[set(matching_homebattery_charge)]
        homebattery_charge_df[country_i] = df_matching_homebattery_charge.sum(axis=1)
        
        matching_gas_OCGT = [s for s in df_gas_OCGT.columns if country_i in s]
        df_matching_gas_OCGT = df_gas_OCGT[set(matching_gas_OCGT)]
        gas_df_OCGT[country_i] = df_matching_gas_OCGT.sum(axis=1)

        matching_gas_CCGT = [s for s in df_gas_CCGT.columns if country_i in s]
        df_matching_gas_CCGT = df_gas_CCGT[set(matching_gas_CCGT)]
        gas_df_CCGT[country_i] = df_matching_gas_CCGT.sum(axis=1)

        matching_coal = [s for s in df_coal.columns if country_i in s]
        df_matching_coal = df_coal[set(matching_coal)]
        coal_df[country_i] = df_matching_coal.sum(axis=1)

        matching_nuclear = [s for s in df_nuclear.columns if country_i in s]
        df_matching_nuclear = df_nuclear[set(matching_nuclear)]
        nuclear_df[country_i] = df_matching_nuclear.sum(axis=1)

        matching_gas_CHP_CC = [s for s in df_gas_CHP_CC.columns if country_i in s]
        df_matching_gas_CHP_CC = df_gas_CHP_CC[set(matching_gas_CHP_CC)]
        gas_CHP_CC_df[country_i] = df_matching_gas_CHP_CC.sum(axis=1)

        matching_gas_CHP = [s for s in df_gas_CHP.columns if country_i in s]
        df_matching_gas_CHP = df_gas_CHP[set(matching_gas_CHP)]
        gas_CHP_df[country_i] = df_matching_gas_CHP.sum(axis=1)

        matching_biomass_CHP_CC = [s for s in df_biomass_CHP_CC.columns if country_i in s]
        df_matching_biomass_CHP_CC = df_biomass_CHP_CC[set(matching_biomass_CHP_CC)]
        biomass_CHP_CC_df[country_i] = df_matching_biomass_CHP_CC.sum(axis=1)

        matching_biomass_CHP = [s for s in df_biomass_CHP.columns if country_i in s]
        df_matching_biomass_CHP = df_biomass_CHP[set(matching_biomass_CHP)]
        biomass_CHP_df[country_i] = df_matching_biomass_CHP.sum(axis=1)

        df_import_0 = n.lines['bus0']
        df_import_1 = n.lines['bus1']
        matching_import_0 = pd.Series([s for s in df_import_0 if country_i in s]).unique()
        if len(matching_import_0) > 0:
            matching_import_0 = matching_import_0.item()
        else:
            matching_import_0 = 'none'
        matching_import_1 = pd.Series([s for s in df_import_1 if country_i in s]).unique()
        if len(matching_import_1) > 0:
            matching_import_1 = matching_import_1.item()
        else:
            matching_import_1 = 'none'
        df_matching_import_0 = - n.lines_t.p0[n.lines[n.lines['bus0'] == matching_import_0].index] # p0 is positive when power is withdrawn from bus0 (i.e. when bus0 exports)
        df_matching_import_1 = - n.lines_t.p1[n.lines[n.lines['bus1'] == matching_import_1].index] # p1 is positive when power is withdrawn from bus1 (i.e. when bus0 exports)

        import_df[country_i] = df_matching_import_0.sum(axis=1) + df_matching_import_1.sum(axis=1)

        df_import_dc_0 = n.links[n.links.carrier == 'DC']['bus0']
        df_import_dc_1 = n.links[n.links.carrier == 'DC']['bus1']

        link_bus0 = []
        link_bus1 = []

        for dc in range(len(df_import_dc_0)):
            s0 = df_import_dc_0[dc]
            s1 = df_import_dc_1[dc]
            
            if (country_i in s0) and (country_i not in s1):
                link_bus0.append(df_import_dc_0.index[dc]) 
                
            if (country_i in s1) and (country_i not in s0):
                link_bus1.append(df_import_dc_1.index[dc]) 
                
        links_DC_import_0 = - n.links_t.p0[link_bus0]
        links_DC_import_1 = - n.links_t.p1[link_bus1]

        links_DC_import_df[country_i] = links_DC_import_0.sum(axis=1) + links_DC_import_1.sum(axis=1)

        for ci in df_load.columns:
            if country_i in ci:
                try:
                    load_df[country_i]
                    load_df[country_i] = load_df[country_i] + df_load[ci]
                except:
                    load_df[country_i] = df_load[ci]

    return load_df, onwind_df, offwind_ac_df, offwind_dc_df, solar_df, solar_roof_df, ror_df, hydro_df, phs_df, X_df, X_e_cap_df, X_charge_df, X_discharge_df, H2_charge_df, battery_df, e_battery_df, battery_e_cap_df, battery_p_cap_df, homebattery_e_cap_df,homebattery_p_cap_df,homebattery_df,battery_charge_df, homebattery_charge_df,gas_df_OCGT,gas_df_CCGT,coal_df,nuclear_df,gas_CHP_CC_df,gas_CHP_df,biomass_CHP_CC_df,biomass_CHP_df,import_df,links_DC_import_df

c_hat = 20
c1 = 350
c2 = 350

path =  '/home/au485969/project/pypsa-eur-0.3.0/pypsa-eur-sec/results/sspace_addition_boundary_3/'

networks_list = os.listdir(path + 'postnetworks')
print(networks_list)
csv_path = path + 'csvs'

#%%
df_store = pd.DataFrame()
df_discharge_time = pd.DataFrame()
df_link = pd.DataFrame()
df_onwind = pd.DataFrame()
df_offwind = pd.DataFrame()
df_solar = pd.DataFrame()
df_ror = pd.DataFrame()

df_onwind_cap = pd.DataFrame()
df_offwind_cap = pd.DataFrame()
df_solar_cap = pd.DataFrame()
df_ror_cap = pd.DataFrame()
df_gas_OCGT_cap = pd.DataFrame()
df_gas_CCGT_cap = pd.DataFrame()
df_coal_cap = pd.DataFrame()
df_nuclear_cap = pd.DataFrame()
df_gas_CHP_CC_cap = pd.DataFrame()
df_gas_CHP_cap = pd.DataFrame()
df_biomass_CHP_CC_cap = pd.DataFrame()
df_biomass_CHP_cap = pd.DataFrame()
df_hydro_cap = pd.DataFrame()

df_hydro = pd.DataFrame()
df_gas_OCGT = pd.DataFrame()
df_gas_CCGT = pd.DataFrame()
df_coal = pd.DataFrame()
df_nuclear = pd.DataFrame()
df_biomass_CHP_CC = pd.DataFrame()
df_biomass_CHP = pd.DataFrame()
df_battery_loadshift_g = pd.DataFrame()
df_battery_loadshift_l = pd.DataFrame()
df_storage_loadshift_g = pd.DataFrame()
df_storage_loadshift_l = pd.DataFrame()
# df_AClines_ur_mean = pd.DataFrame()
# df_DClinks_ur_mean = pd.DataFrame()
# df_AClines_ur_var = pd.DataFrame()
# df_DClinks_ur_var = pd.DataFrame()

countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 
            'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 
            'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 
            'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 
            'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 
            'SE', 'SI', 'SK']

onwind_n = pd.DataFrame()
load_n = pd.DataFrame()
offwind_ac_n = pd.DataFrame()
offwind_dc_n = pd.DataFrame()
solar_n = pd.DataFrame()
solar_roof_n = pd.DataFrame()
ror_n = pd.DataFrame()
hydro_n = pd.DataFrame()  
phs_n = pd.DataFrame()  
X_state_n = pd.DataFrame() 
X_e_cap_n = pd.DataFrame(index=countries)    
X_charge_n = pd.DataFrame()
X_discharge_n = pd.DataFrame()
H2_charge_n = pd.DataFrame()
battery_n = pd.DataFrame()
e_battery_n = pd.DataFrame()
battery_e_cap_n = pd.DataFrame(index=countries)  
homebattery_e_cap_n = pd.DataFrame(index=countries)  
battery_p_cap_n = pd.DataFrame(index=countries)  
homebattery_p_cap_n = pd.DataFrame(index=countries)  
homebattery_n = pd.DataFrame()
battery_charge_n = pd.DataFrame()
homebattery_charge_n = pd.DataFrame()
gas_n_OCGT = pd.DataFrame()
gas_n_CCGT = pd.DataFrame()
coal_n = pd.DataFrame()
nuclear_n = pd.DataFrame()
gas_CHP_CC_n = pd.DataFrame()
gas_CHP_n = pd.DataFrame()
biomass_CHP_CC_n = pd.DataFrame()
biomass_CHP_n = pd.DataFrame()
import_n = pd.DataFrame()
links_DC_import_n = pd.DataFrame()

j = 0
i = 2
for filename in networks_list:
    n = pypsa.Network(path + 'postnetworks' + '/' + filename,
                        override_component_attrs=override_component_attrs)
    sector_i = []
    for i in filename.split('/')[-1].split('-'):
        if 'X Store' in i:
            print('cost factor:')
            print(i.split('+')[1][1:])
            cost_factor = float(i.split('+')[1][1:])
        if 'X Charge+c' in i:
            print('charge cost factor:')
            print(i.split('+')[1][1:])
            charge_cost_factor = float(i.split('+')[1][1:])
        if 'X Discharge+c' in i:
            print('discharge cost factor')
            print(i.split('+')[1][1:])
            discharge_cost_factor = float(i.split('+')[1][1:])
        if 'elec' in i:
            weatheryear = int(i.split('_')[2][1:])
            nnodes = int(i.split('_')[3][1:])
            trans = float(i.split('_')[4][2:])
            co2_cap = float(i.split('_')[6][4:])

        if i == 'H' or i == 'T' or i == 'B' or i == 'I':
            sector_i.append(i)

    tres = int(filename.split('/')[-1].split('-')[1][0:-1])

    energy_storage_cost = c_hat*cost_factor
    energy_capacity = n.stores[n.stores.carrier == 'X'].e_nom_opt.sum()
    co2_price = n.global_constraints.at["CO2Limit","mu"]
    discharge_efficiency =  n.links[n.links.carrier == 'X Discharge'].efficiency[0]
    charge_efficiency =  n.links[n.links.carrier == 'X Charge'].efficiency[0]
    discharge_capacity =  n.links[n.links.carrier == 'X Discharge'].p_nom_opt.sum()*discharge_efficiency # discharging
    charge_capacity = n.links[n.links.carrier == 'X Charge'].p_nom_opt.sum()
    battery_G_capacity = (n.links.query("carrier == 'battery discharger'").p_nom_opt*n.links.query("carrier == 'battery discharger'").efficiency).sum()
    battery_E_capacity = n.stores.query("carrier == 'battery'").e_nom_opt.sum()
    homebattery_G_capacity = (n.links.query("carrier == 'home battery discharger'").p_nom_opt*n.links.query("carrier == 'home battery discharger'").efficiency).sum()
    homebattery_E_capacity = n.stores.query("carrier == 'home battery'").e_nom_opt.sum()
    PHS_G_capacity = n.storage_units.query("carrier == 'PHS'").p_nom_opt.sum()
    PHS_E_capacity = (n.storage_units.query("carrier == 'PHS'").p_nom_opt * n.storage_units.query("carrier == 'PHS'").max_hours).sum()
    hydrogen_G_capacity = n.links.query('carrier == "H2 Electrolysis"').p_nom_opt.sum()
    hydrogen_E_capacity = n.stores.query('carrier == "H2"').e_nom_opt.sum()
    
    storage_discharge_t = -n.links_t.p1[n.links[n.links.carrier == 'X Discharge'].index].sum().sum() # MWe (after conversion to electricity)

    buses = n.buses.query('carrier == "AC"').index
    links = n.links

    xss_0 = [links.bus0[links.bus0 == i].loc[[i in links.bus0[links.bus0 == i].index[j] for j in range(len(links.bus0[links.bus0 == i].index))]].drop([i + ' X Charge',i + ' battery charger',i + ' electricity distribution grid']).index.tolist() for i in buses]
    loads_0 = [x for xs in xss_0 for x in xs] # links using electricity from HVAC bus
    loads_1 = [links[links.index == i + ' electricity distribution grid'].index.item() for i in buses]
    loads_t_0 = n.links_t.p0[loads_0].sum().sum()/1e6*8760/n.snapshots.shape[0]
    loads_t_1 = n.links_t.p0[loads_1][n.links_t.p0[loads_1] > 0].fillna(0).sum().sum()/1e6*8760/n.snapshots.shape[0]
    loads_t_2 = n.loads_t.p[n.loads.query('carrier == "industry electricity"').index].sum().sum()/1e6*8760/n.snapshots.shape[0]

    loads_sum = loads_t_0 + loads_t_1 + loads_t_2

    xss1 = [links.bus1[links.bus1 == i].loc[[i in links.bus1[links.bus1 == i].index[j] for j in range(len(links.bus1[links.bus1 == i].index))]].drop([i + ' X Discharge',i + ' battery discharger']).index.tolist() for i in buses]
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

    standing_loss = n.stores[n.stores.carrier == 'X'].standing_loss[0]
    tau = (-(np.log(1-standing_loss)**-1)/24).round(1)
    lh_avg = energy_capacity/(n.loads_t.p.sum(axis=1).mean()) # Average load hours
    P = n.links[n.links.carrier == 'X Discharge'].p_nom_opt.sort_values()
    E = n.stores[n.stores.carrier == 'X'].e_nom_opt.sort_values()
    P_mean = P.mean()
    sigma_P_rel = np.std(P)/P_mean*100
    E_mean = E.mean()
    sigma_E_rel = np.std(E)/E_mean*100
    P_max = P_mean + P_mean # Statistical max
    E_max = E_mean + E_mean # Statistical max
    n_P_out = len(P[P > P_max]) # number of outliers (Outlier indicate hydrogen is con)
    n_E_out = len(E[E > E_max]) # number of outliers
    system_cost = n.objective/1e9
    avail = n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt).sum().groupby(n.generators.carrier).sum()
    used = n.generators_t.p.sum().groupby(n.generators.carrier).sum()
    avail_offwind = avail.loc['offwind-ac'] + avail.loc['offwind-dc']
    avail_onwind = avail.loc['onwind']
    avail_solar = avail.loc['solar'] + avail.loc['solar rooftop'] 
    used_offwind = used.loc['offwind-ac'] + used.loc['offwind-dc'] 
    used_onwind = used.loc['onwind']
    used_solar = used.loc['solar'] + used.loc['solar rooftop']
    battery_storage_discharge_t = -n.links_t.p1[n.links[n.links.carrier == 'battery discharger'].index].sum().sum() 
    battery_load_coverage =  battery_storage_discharge_t.sum()/n.loads_t.p.sum().sum()  
    charge_cost = c1*charge_cost_factor
    discharge_cost = c2*discharge_cost_factor
    gen = {}
    test = pd.DataFrame()
    test[0] = n.generators.carrier
    techs = ['onwind','offwind','solar','ror']
    for tech in techs:
        test[1] = [tech in n.generators.carrier[s] for s in range(len(n.generators))]
        gen[tech] = np.round(n.generators_t.p[n.generators[test[1]].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_gas_OCGT = np.round(-n.links_t.p1[n.links[n.links.carrier == 'OCGT'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_gas_CCGT = np.round(-n.links_t.p1[n.links[n.links.carrier == 'CCGT'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_coal = np.round(-n.links_t.p1[n.links[n.links.carrier == 'coal'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_nuclear = np.round(-n.links_t.p1[n.links[n.links.carrier == 'nuclear'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_gas_CHP_CC = np.round(-n.links_t.p1[n.links[n.links.carrier == 'urban central gas CHP CC'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_gas_CHP = np.round(-n.links_t.p1[n.links[n.links.carrier == 'urban central gas CHP'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_biomass_CHP_CC = np.round(-n.links_t.p1[n.links[n.links.carrier == 'urban central solid biomass CHP CC'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_biomass_CHP = np.round(-n.links_t.p1[n.links[n.links.carrier == 'urban central solid biomass CHP'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_hydrores = np.round(n.storage_units_t.p[n.storage_units[n.storage_units.carrier == 'hydro'].index].sum().sum(),1)*8760/n.snapshots.shape[0]
    gen_hydro = gen_hydrores + gen['ror']

    sto_caps = n.storage_units[['p_nom_opt','carrier']].groupby('carrier').sum()
    gen_caps = n.generators[['p_nom_opt','carrier']].groupby('carrier').sum()
    cap_onwind = gen_caps.loc['onwind']
    cap_offwind = gen_caps.loc['offwind-ac'] + gen_caps.loc['offwind-dc']
    cap_solar = gen_caps.loc['solar'] + gen_caps.loc['solar rooftop']
    cap_hydro = gen_caps.loc['ror'] + sto_caps.loc['hydro']
    cap_gas_OCGT = (n.links.query("carrier == 'OCGT'").efficiency*n.links.query("carrier == 'OCGT'").p_nom_opt).sum() 
    cap_gas_CCGT = (n.links.query("carrier == 'CCGT'").efficiency*n.links.query("carrier == 'CCGT'").p_nom_opt).sum()
    cap_coal = (n.links.query("carrier == 'coal'").efficiency*n.links.query("carrier == 'coal'").p_nom_opt).sum()
    cap_nuclear = (n.links.query("carrier == 'nuclear'").efficiency*n.links.query("carrier == 'nuclear'").p_nom_opt).sum()
    cap_gas_CHP_CC = (n.links.query("carrier == 'urban central gas CHP CC'").efficiency*n.links.query("carrier == 'urban central gas CHP CC'").p_nom_opt).sum()
    cap_gas_CHP = (n.links.query("carrier == 'urban central gas CHP'").efficiency*n.links.query("carrier == 'urban central gas CHP'").p_nom_opt).sum()
    cap_biomass_CHP_CC = (n.links.query("carrier == 'urban central solid biomass CHP CC'").efficiency*n.links.query("carrier == 'urban central solid biomass CHP CC'").p_nom_opt).sum()
    cap_biomass_CHP = (n.links.query("carrier == 'urban central solid biomass CHP'").efficiency*n.links.query("carrier == 'urban central solid biomass CHP'").p_nom_opt).sum()

    # A = n.generators_t.p.sum().sum() # This is a bad approach
    # B = n.storage_units_t.p[n.storage_units.carrier[n.storage_units.carrier == 'hydro'].index ].sum().sum()
    # C = (-n.links_t.p1[n.links.carrier[n.links.carrier == 'OCGT'].index]).sum().sum() # CCGT missing

    elec_gen_tot = supply_sum

    Charge = n.links_t.p0[n.links.query('carrier == "X Charge"').index]
    Discharge = -n.links_t.p1[n.links.query('carrier == "X Discharge"').index]

    excessive_dispatch = 0
    for jjj in range(len(Discharge.columns)):
        nodal_discharge = Discharge.T.iloc[jjj]
        nodal_charge = Charge.T.iloc[jjj]  
        
        for iii in range(len(nodal_charge)):
            nodal_discharge_i = nodal_discharge.iloc[iii]
            nodal_charge_i = nodal_charge.iloc[iii] 
            
            if (nodal_discharge_i > 0) and (nodal_charge_i > 0):
                additional_dispatch = nodal_charge_i if nodal_discharge_i > nodal_charge_i else nodal_discharge_i
                excessive_dispatch += additional_dispatch
                
    total_storage_X_dispatch = Discharge.sum().sum()
    percentage_distortion = excessive_dispatch/total_storage_X_dispatch*100

    listt = [weatheryear,
             energy_storage_cost,
             co2_cap*1e2,
             charge_efficiency,
             discharge_efficiency,
             tau,
             charge_cost,
             discharge_cost,
             trans,
             nnodes,
             tres,
             co2_price,
             system_cost,
             storage_discharge_t/1e6,
             load_coverage*1e2,
             battery_load_coverage*1e2,
             discharge_capacity*1e-3,
             energy_capacity*1e-3,
             charge_capacity*1e-3,
             battery_G_capacity*1e-3,
             battery_E_capacity*1e-3,
             homebattery_G_capacity*1e-3,
             homebattery_E_capacity*1e-3,
             PHS_G_capacity*1e-3,
             PHS_E_capacity*1e-3,
             hydrogen_G_capacity*1e-3,
             hydrogen_E_capacity*1e-3,
             avail_onwind,
             avail_offwind,
             avail_solar,
             used_onwind,
             used_offwind,
             used_solar,
             gen_gas_OCGT,
             gen_gas_CCGT,
             gen_coal,
             gen_nuclear,
             gen_gas_CHP_CC,
             gen_gas_CHP,
             gen_biomass_CHP_CC,
             gen_biomass_CHP,
             gen['onwind'],
             gen['offwind'],
             gen['solar'],
             gen_hydro,
             elec_gen_tot,
             cap_onwind.item()/1e3,
             cap_offwind.item()/1e3,
             cap_solar.item()/1e3,
             cap_hydro.item()/1e3,
             cap_gas_OCGT.item()/1e3,
             cap_gas_CCGT.item()/1e3,
             cap_coal.item()/1e3,
             cap_nuclear.item()/1e3,
             cap_gas_CHP_CC.item()/1e3,
             cap_gas_CHP.item()/1e3,
             cap_biomass_CHP_CC.item()/1e3,
             cap_biomass_CHP.item()/1e3,
             percentage_distortion
             ]

    listt = pd.DataFrame(listt).round(decimals=3)
    listt = listt[::].append(['-'.join(sector_i)],ignore_index=True)

    if j == 0:
        df = pd.DataFrame(index=np.arange(len(listt)))

    df[j] = listt

    e_cap = n.stores.query('carrier == "X"').e_nom_opt 
    p_cap = n.links.query('carrier == "X Discharge"').p_nom_opt

    p_cap.index = e_cap.index

    discharge_time = e_cap/p_cap




    year = filename[8:12]


    var = year + '-' + '-'.join(sector_i)

    j += 1

df.index = ['weatheryear','c_hat [EUR/kWh]','co2_cap [%]',
            'eta1 [-]', 'eta2 [-]', 'tau [n_days]',
            'c1','c2','transmission','nnodes','tres',
            'CO2 price [EUR]','c_sys [bEUR]','X Discharge',
            'load_coverage [%]','battery_load_coverage [%]',
            'G_discharge [GW]','E [GWh]','G_charge [GW]',
            'G_battery [GW]','E_battery [GWh]',
            'G_homebattery [GW]','E_homebattery [GWh]',
            'G_PHS [GW]','E_PHS [GWh]',
            'G_electrolysis [GW]','E_H2 [GWh]',
            'avail_onwind [MWh]','avail_offwind [MWh]','avail_solar [MWh]',
            'used_onwind [MWh]','used_offwind [MWh]','used_solar [MWh]',
            'gas_OCGT_gen [MWh]', 'gas_CCGT_gen [MWh]','coal_gen [MWh]','nuclear_gen [MWh]',
            'gas_CHP_CC_gen [MWh]','gas_CHP_gen [MWh]',
            'biomass_CHP_CC_gen [MWh]','biomass_CHP_gen [MWh]',
            'onwind_gen [MWh]', 'offwind_gen [MWh]', 'solar_gen [MWh]','hydro_gen [MWh]','tot_gen [MWh]',
            'cap_onwind [GW]','cap_offwind [GW]','cap_solar [GW]',
            'cap_hydro [GW]','cap_gas_OCGT [GW]','cap_gas_CCGT [GW]','cap_coal [GW]','cap_nuclear [GW]','cap_gas_CHP_CC [GW]',
            'cap_gas_CHP [GW]','cap_biomass_CHP_CC [GW]','cap_biomass_CHP [GW]', 'unintended_storage_cycling [%]',
            'sector']

df.to_csv(csv_path + '/sspace.csv')


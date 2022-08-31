# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:26:57 2022

@author: au485969
"""

import pandas as pd
def read_system_files(path,c,column,tres):
    if len(column) == 4:
        try: 
            year = column
            if ((int(year) % 4 == 0) and (int(year) % 100 != 0)) or (int(year) % 1000 == 0):
                enddate = '12/31/'
                delta = 0
            else:
                enddate = '1/1/'
                delta = 1
        except:
            year = '2013'
            enddate = '1/1/'
            delta = 1
    else:
        year = '2013'
        enddate = '1/1/'
        delta = 1
        
    solar_y = pd.read_csv(path + 'solar_timeseries.csv',index_col=[1,0])
    config = column
    solar_y_i = solar_y[config]/1e3
    solar_y_i.index = solar_y_i.index.set_names(['country', 'date'])
    
    solar_roof_y = pd.read_csv(path + 'solar_roof_timeseries.csv',index_col=[1,0])
    solar_roof_y_i = solar_roof_y[config]/1e3
    solar_roof_y_i.index = solar_roof_y_i.index.set_names(['country', 'date'])
    
    load_y = pd.read_csv(path + 'load_timeseries.csv',index_col=[1,0])
    load_y_i = load_y[config]/1e3
    load_y_i.index = load_y_i.index.set_names(['country', 'date'])
    
    onwind_y = pd.read_csv(path + 'onwind_timeseries.csv',index_col=[1,0])
    onwind_y_i = onwind_y[config]/1e3
    onwind_y_i.index = onwind_y_i.index.set_names(['country', 'date'])
    
    offwind_ac_y = pd.read_csv(path + 'offwind_ac_timeseries.csv',index_col=[1,0])
    offwind_ac_y_i = offwind_ac_y[config]/1e3
    offwind_ac_y_i.index = offwind_ac_y_i.index.set_names(['country', 'date'])
    
    offwind_dc_y = pd.read_csv(path + 'offwind_dc_timeseries.csv',index_col=[1,0])
    offwind_dc_y_i = offwind_dc_y[config]/1e3
    offwind_dc_y_i.index = offwind_dc_y_i.index.set_names(['country', 'date'])
    
    hydro_y = pd.read_csv(path + 'hydro_timeseries.csv',index_col=[1,0])
    hydro_y_i = hydro_y[config]/1e3
    hydro_y_i.index = hydro_y_i.index.set_names(['country', 'date'])
    
    ror_y = pd.read_csv(path + 'ror_timeseries.csv',index_col=[1,0])
    ror_y_i = ror_y[config]/1e3
    ror_y_i.index = ror_y_i.index.set_names(['country', 'date'])
    
    phs_y = pd.read_csv(path + 'phs_timeseries.csv',index_col=[1,0])
    phs_y_i = phs_y[config]/1e3
    phs_y_i.index = phs_y_i.index.set_names(['country', 'date'])
    
    phs_charge_y_i = phs_y_i.copy()
    phs_charge_y_i[phs_charge_y_i > 0] = 0
    phs_charge_y_i = phs_charge_y_i 
    phs_y_i[phs_y_i < 0] = 0
    
    gas_OCGT_y = pd.read_csv(path + 'gas_OCGT_timeseries.csv',index_col=[1,0])
    gas_OCGT_y_i = gas_OCGT_y[config]/1e3
    gas_OCGT_y_i.index = gas_OCGT_y_i.index.set_names(['country', 'date'])
    
    gas_CCGT_y = pd.read_csv(path + 'gas_CCGT_timeseries.csv',index_col=[1,0])
    gas_CCGT_y_i = gas_CCGT_y[config]/1e3
    gas_CCGT_y_i.index = gas_CCGT_y_i.index.set_names(['country', 'date'])
    
    coal_y = pd.read_csv(path + 'coal_timeseries.csv',index_col=[1,0])
    coal_y_i = coal_y[config]/1e3
    coal_y_i.index = coal_y_i.index.set_names(['country', 'date'])
    
    nuclear_y = pd.read_csv(path + 'nuclear_timeseries.csv',index_col=[1,0])
    nuclear_y_i = nuclear_y[config]/1e3
    nuclear_y_i.index = nuclear_y_i.index.set_names(['country', 'date'])
    
    gas_CHP_CC_y = pd.read_csv(path + 'gas_CHP_CC_timeseries.csv',index_col=[1,0])
    gas_CHP_CC_y_i = gas_CHP_CC_y[config]/1e3
    gas_CHP_CC_y_i.index = gas_CHP_CC_y_i.index.set_names(['country', 'date'])
    
    gas_CHP_y = pd.read_csv(path + 'gas_CHP_timeseries.csv',index_col=[1,0])
    gas_CHP_y_i = gas_CHP_y[config]/1e3
    gas_CHP_y_i.index = gas_CHP_y_i.index.set_names(['country', 'date'])
    
    biomass_CHP_CC_y = pd.read_csv(path + 'biomass_CHP_CC_timeseries.csv',index_col=[1,0])
    biomass_CHP_CC_y_i = biomass_CHP_CC_y[config]/1e3
    biomass_CHP_CC_y_i.index = biomass_CHP_CC_y_i.index.set_names(['country', 'date'])
    
    biomass_CHP_y = pd.read_csv(path + 'biomass_CHP_timeseries.csv',index_col=[1,0])
    biomass_CHP_y_i = biomass_CHP_y[config]/1e3
    biomass_CHP_y_i.index = biomass_CHP_y_i.index.set_names(['country', 'date'])
    
    battery_discharge_y = pd.read_csv(path + 'battery_discharge_timeseries.csv',index_col=[1,0])
    battery_discharge_y_i = battery_discharge_y[config]/1e3
    battery_discharge_y_i.index = battery_discharge_y_i.index.set_names(['country', 'date'])
    
    homebattery_discharge_y = pd.read_csv(path + 'homebattery_discharge_timeseries.csv',index_col=[1,0])
    homebattery_discharge_y_i = homebattery_discharge_y[config]/1e3
    homebattery_discharge_y_i.index = homebattery_discharge_y_i.index.set_names(['country', 'date'])
    
    battery_charge_y = pd.read_csv(path + 'battery_charge_timeseries.csv',index_col=[1,0])
    battery_charge_y_i = battery_charge_y[config]/1e3 #/np.sqrt(0.96) # Home battery RTE = 0.96
    battery_charge_y_i.index = battery_charge_y_i.index.set_names(['country', 'date'])

    homebattery_charge_y = pd.read_csv(path + 'homebattery_charge_timeseries.csv',index_col=[1,0])
    homebattery_charge_y_i = homebattery_charge_y[config]/1e3 #/np.sqrt(0.96) # Home battery RTE = 0.96
    homebattery_charge_y_i.index = homebattery_charge_y_i.index.set_names(['country', 'date'])
    
    charge_y = pd.read_csv(path + 'X_charge_timeseries.csv',index_col=[1,0])
    charge_y_i = charge_y[config]/1e3 #/0.8 # Electrolysis charging 
    charge_y_i.index = charge_y_i.index.set_names(['country', 'date'])
    
    discharge_y = pd.read_csv(path + 'X_discharge_timeseries.csv',index_col=[1,0])
    discharge_y_i = discharge_y[config]/1e3
    discharge_y_i.index = discharge_y_i.index.set_names(['country', 'date'])
    
    H2_charge_y = pd.read_csv(path + 'H2_charge_timeseries.csv',index_col=[1,0])
    H2_charge_y_i = H2_charge_y[config]/1e3  
    H2_charge_y_i.index = H2_charge_y_i.index.set_names(['country', 'date'])
    
    h2_y = pd.read_csv(path + 'X_timeseries.csv',index_col=[1,0])
    h2_y_i = h2_y[config]/1e3
    h2_y_i.index = h2_y_i.index.set_names(['country', 'date'])
    
    e_battery_y = pd.read_csv(path + 'e_battery_timeseries.csv',index_col=[1,0])
    e_battery_y_i = e_battery_y[config]/1e3
    e_battery_y_i.index = e_battery_y_i.index.set_names(['country', 'date'])
    
    h2_e_caps_y = pd.read_csv(path + 'X_ecaps.csv',index_col=[0])
    h2_e_caps_y_i = h2_e_caps_y[config]/1e3
    
    battery_e_caps_y = pd.read_csv(path + 'battery_ecaps.csv',index_col=[0])
    battery_e_caps_y_i = battery_e_caps_y[config]/1e3
    
    import_dc_y = pd.read_csv(path + 'import_dc_timeseries.csv',index_col=[1,0])
    import_dc_y_i = import_dc_y[config]/1e3
    import_dc_y_i.index = import_dc_y_i.index.set_names(['country', 'date'])
    
    import_ac_y = pd.read_csv(path + 'import_ac_timeseries.csv',index_col=[1,0])
    import_ac_y_i = import_ac_y[config]/1e3
    import_ac_y_i.index = import_ac_y_i.index.set_names(['country', 'date'])
    
    load_y_EU = load_y_i.groupby('date').sum()  
    load_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta) ,freq=tres)[:-1]
    if c != 'EU':
        load_y_c = load_y_i.loc[(c)]
        load_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    
    onwind_y_EU = onwind_y_i.groupby('date').sum()
    onwind_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        onwind_y_c = onwind_y_i.loc[(c)]
        onwind_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    offwind_y_EU = offwind_ac_y_i.groupby('date').sum() + offwind_dc_y_i.groupby('date').sum()  
    offwind_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        offwind_y_c = offwind_ac_y_i.loc[(c)] + offwind_dc_y_i.loc[(c)]
        offwind_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    solar_y_EU = solar_y_i.groupby('date').sum() + solar_roof_y_i.groupby('date').sum()  
    solar_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        solar_y_c = solar_y_i.loc[(c)] + solar_roof_y_i.loc[(c)]
        solar_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    
    h2_y_EU = h2_y_i.groupby('date').sum()  
    h2_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        h2_y_c = h2_y_i.loc[(c)]
        h2_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    e_battery_y_EU = e_battery_y_i.groupby('date').sum()  
    e_battery_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        e_battery_y_c = e_battery_y_i.loc[(c)]
        e_battery_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    
    phs_charge_y_EU = phs_charge_y_i.groupby('date').sum()  
    phs_charge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        phs_charge_y_c = phs_charge_y_i.loc[(c)]
        phs_charge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    phs_discharge_y_EU = phs_y_i.groupby('date').sum()  
    phs_discharge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        phs_discharge_y_c = phs_y_i.loc[(c)]
        phs_discharge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    ror_y_EU = ror_y_i.groupby('date').sum()
    ror_y_EU.index = pd.to_datetime(ror_y_EU.index)
    
    hydro_y_EU = hydro_y_i.groupby('date').sum() + ror_y_i.groupby('date').sum() #+ phs_y_i.groupby('date').sum()  
    hydro_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        hydro_y_c = hydro_y_i.loc[(c)] + ror_y_i.loc[(c)] #+ phs_y_i.loc[(c)]
        hydro_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
        ror_y_c = ror_y_i.loc[(c)]
        ror_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    charge_y_EU = charge_y_i.groupby('date').sum()  
    charge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        charge_y_c = charge_y_i.loc[(c)]
        charge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    discharge_y_EU = discharge_y_i.groupby('date').sum()  
    discharge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        discharge_y_c = discharge_y_i.loc[(c)]
        discharge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    H2_charge_y_EU = H2_charge_y_i.groupby('date').sum()  
    H2_charge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        H2_charge_y_c = H2_charge_y_i.loc[(c)]
        H2_charge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    gas_OCGT_y_EU = gas_OCGT_y_i.groupby('date').sum()  
    gas_OCGT_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        gas_OCGT_y_c = gas_OCGT_y_i.loc[(c)]
        gas_OCGT_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    
    gas_CCGT_y_EU = gas_CCGT_y_i.groupby('date').sum()  
    gas_CCGT_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        gas_CCGT_y_c = gas_CCGT_y_i.loc[(c)]
        gas_CCGT_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    coal_y_EU = coal_y_i.groupby('date').sum()  
    coal_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        coal_y_c = coal_y_i.loc[(c)]
        coal_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    nuclear_y_EU = nuclear_y_i.groupby('date').sum()  
    nuclear_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        nuclear_y_c = nuclear_y_i.loc[(c)]
        nuclear_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    gas_CHP_CC_y_EU = gas_CHP_CC_y_i.groupby('date').sum()  
    gas_CHP_CC_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        gas_CHP_CC_y_c = gas_CHP_CC_y_i.loc[(c)]
        gas_CHP_CC_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    gas_CHP_y_EU = gas_CHP_y_i.groupby('date').sum()  
    gas_CHP_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        gas_CHP_y_c = gas_CHP_y_i.loc[(c)]
        gas_CHP_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    biomass_CHP_CC_y_EU = biomass_CHP_CC_y_i.groupby('date').sum()  
    biomass_CHP_CC_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        biomass_CHP_CC_y_c = biomass_CHP_CC_y_i.loc[(c)]
        biomass_CHP_CC_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
        
    biomass_CHP_y_EU = biomass_CHP_y_i.groupby('date').sum()  
    biomass_CHP_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        biomass_CHP_y_c = biomass_CHP_y_i.loc[(c)]
        biomass_CHP_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    battery_discharge_y_EU = battery_discharge_y_i.groupby('date').sum() + homebattery_discharge_y_i.groupby('date').sum()   
    battery_discharge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        battery_discharge_y_c = battery_discharge_y_i.loc[(c)] + homebattery_discharge_y_i.loc[(c)]
        battery_discharge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    
    battery_charge_y_EU = battery_charge_y_i.groupby('date').sum() + homebattery_charge_y_i.groupby('date').sum() 
    battery_charge_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        battery_charge_y_c = battery_charge_y_i.loc[(c)] + homebattery_charge_y_i.loc[(c)]
        battery_charge_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

    import_dc_y_EU = import_dc_y_i.groupby('date').sum() # should be zero
    import_dc_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        import_dc_y_c = import_dc_y_i.loc[(c)]
        import_dc_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    
    import_ac_y_EU = import_ac_y_i.groupby('date').sum() # should be zero
    import_ac_y_EU.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]
    if c != 'EU':
        import_ac_y_c = import_ac_y_i.loc[(c)]
        import_ac_y_c.index = pd.date_range('1/1/' + year,enddate + str(int(year) + delta),freq=tres)[:-1]

        export_dc_y_c = import_dc_y_c.copy()
        export_dc_y_c[export_dc_y_c > 0] = 0
        import_dc_y_c[import_dc_y_c < 0] = 0
    
    export_dc_y_EU = import_dc_y_EU.copy()
    export_dc_y_EU[export_dc_y_EU > 0] = 0
    import_dc_y_EU[import_dc_y_EU < 0] = 0
    
    if c != 'EU':
        export_ac_y_c = import_ac_y_c.copy()
        export_ac_y_c[export_ac_y_c > 0] = 0
        import_ac_y_c[import_ac_y_c < 0] = 0
    
    export_ac_y_EU = import_ac_y_EU.copy()
    export_ac_y_EU[export_ac_y_EU > 0] = 0
    import_ac_y_EU[import_ac_y_EU < 0] = 0
    
    if c != 'EU':
        output = [load_y_c, onwind_y_c, offwind_y_c, solar_y_c, 
                  phs_charge_y_c,phs_discharge_y_c, hydro_y_c, charge_y_c, discharge_y_c, H2_charge_y_c,
                  gas_OCGT_y_c, gas_CCGT_y_c, coal_y_c, nuclear_y_c, gas_CHP_CC_y_c,gas_CHP_y_c,biomass_CHP_CC_y_c,biomass_CHP_y_c,battery_discharge_y_c, battery_charge_y_c, 
                  import_dc_y_c, import_ac_y_c,  export_dc_y_c, export_ac_y_c, h2_y_c, h2_e_caps_y_i.loc[c],e_battery_y_c,battery_e_caps_y_i.loc[c], ror_y_c]
    else:
        output = [load_y_EU, onwind_y_EU, offwind_y_EU, solar_y_EU, 
                  phs_charge_y_EU,phs_discharge_y_EU, hydro_y_EU, charge_y_EU, discharge_y_EU, H2_charge_y_EU,
                  gas_OCGT_y_EU, gas_CCGT_y_EU, coal_y_EU, nuclear_y_EU, gas_CHP_CC_y_EU,gas_CHP_y_EU, biomass_CHP_CC_y_EU,biomass_CHP_y_EU,battery_discharge_y_EU, battery_charge_y_EU, 
                  import_dc_y_EU, import_ac_y_EU,  export_dc_y_EU, export_ac_y_EU, h2_y_EU, h2_e_caps_y_i.sum(), e_battery_y_EU, battery_e_caps_y_i.sum(), ror_y_EU]
    
    return output
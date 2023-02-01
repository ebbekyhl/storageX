# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:13:52 2022

@author: au485969
"""
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
import numpy as np

def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]
            
def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"

def rename_techs(label):

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [
        #"CHP CC",
        #"gas boiler",
        #"biogas",
        "solar thermal",
        #"air heat pump",
        #"ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]

    rename_if_contains_dict = {
        "CHP CC":"heat-power CC",
        "CHP":"CHP",
        #"nuclear":"nuclear",
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "H2 Electrolysis": "H2 charging",
        "H2 Fuel Cell": "H2",
        "H2 pipeline": "H2",
        "X Discharge": "storage-X",
        "X":"storage-X",
        "Li ion":'EV battery',
        "X Charge": "storage-X charging",
        # "battery": "battery storage",
        "home battery":"battery",
        "biogas": "biomass",
        "biomass": "biomass",
        "air heat pump": "heat pump",
        "ground heat pump": "heat pump",
        "gas": "gas",
        "process emissions CC": "CO2 capture",
        "DAC":"CO2 capture",
    }

    rename = {
        "heat-power CC":"CHP CC",
        "solar rooftop": "solar PV",
        "solar": "solar PV",
        "Sabatier": "gas", #methanation
        "offwind": "wind",
        "offwind-ac": "wind",
        "offwind-dc": "wind",
        "onwind": "wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "CO2 capture",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
        "battery":"battery storage",
        "home battery":"battery storage",
        "battery charger":"battery storage",
        "battery discharger":"battery storage",
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old,new in rename.items():
        if old == label:
            label = new
    return label

def rename_techs_tyndp(label):
    label = rename_techs(label)
    rename_if_contains_dict = {"H2 charging":'H2'}
    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new
            
    return label


def worst_best_week(network,country,case):
    n = network.copy()

    renewables_p = pd.DataFrame(index=pd.date_range('1/1/2013','1/1/2014',freq='3h')[0:-1])
    generators = ['offwind-ac','offwind-dc','onwind','solar','solar rooftop']
    for generator in generators:
        renewables_index = n.generators.query('carrier == @generator').index
        if country != 'EU':
            renewables_index = renewables_index[renewables_index.str.contains(country)]
        renewables_p[generator] = n.generators_t.p[renewables_index].sum(axis=1).values

    stores_p = pd.DataFrame(index=pd.date_range('1/1/2013','1/1/2014',freq='3h')[0:-1])
    stores = ['X Discharge','X Charge']
    for store in stores:
        stores_index = n.links.query('carrier == @store').index
        if country != 'EU':
            stores_index = stores_index[stores_index.str.contains(country)]
        
        if store == 'X Discharge':
            stores_p[store] = -n.links_t.p1[stores_index].sum(axis=1).values
        else:
            stores_p[store] = n.links_t.p0[stores_index].sum(axis=1).values

    # var_t = stores_p
    var_1 = stores_p['X Charge']
    var_2 = stores_p['X Discharge']

    idx_worst_energy = var_1.groupby(pd.Grouper(freq='7d')).sum().iloc[0:-1].idxmax()
    idx_best_energy = var_2.groupby(pd.Grouper(freq='7d')).sum().iloc[0:-1].idxmax()

    # idx_worst_energy = renewables_p.sum(axis=1).groupby(pd.Grouper(freq='7d')).sum().iloc[0:-1].idxmin()
    # idx_best_energy = renewables_p.sum(axis=1).groupby(pd.Grouper(freq='7d')).sum().iloc[0:-1].idxmax()


    worst_indices = pd.date_range(idx_worst_energy-pd.Timedelta('4d'),idx_worst_energy+pd.Timedelta('5d'),freq='3h')
    best_indices = pd.date_range(idx_best_energy-pd.Timedelta('4d'),idx_best_energy+pd.Timedelta('5d'),freq='3h')

    worst_indices = worst_indices[worst_indices < pd.to_datetime('31/12/2013 21:00:00')]
    worst_indices = worst_indices[worst_indices > pd.to_datetime('1/1/2013 00:00:00')]
    best_indices = best_indices[best_indices < pd.to_datetime('31/12/2013 21:00:00')]
    best_indices = best_indices[best_indices > pd.to_datetime('1/1/2013 00:00:00')]

    if case == 'worst':
        dstart = worst_indices[0]
        dend = worst_indices[-1]

    if case == 'best':
        dstart = best_indices[0]
        dend = best_indices[-1]

    return dstart,dend


def rename_low_voltage_techs(label):
    rename_if_contains = ['home battery',
                          'BEV',
                          'V2G',
                          'heat pump',
                          'resistive heater',
                          ]
    
    for rif in rename_if_contains:
        if rif in label:
            label = rif
            
    rename_if_contains_dict = {'electricity distribution grid':'High voltage electricity provision'}
    
    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new
    
    return label

def split_el_distribution_grid(supply,country,network):
    n = network.copy()
    
    low_voltage_consumers = n.links.bus0[n.links.bus0.str.endswith('low voltage')].index
    low_voltage_providers = n.links.bus1[n.links.bus1.str.endswith('low voltage')].index
    domestic_consumers = n.loads.query('carrier == "electricity"').index
    industry_consumers = n.loads.query('carrier == "industry electricity"').index
        
    if country != 'EU':
        low_voltage_consumers = low_voltage_consumers[low_voltage_consumers.str.contains(country)]
        low_voltage_providers = low_voltage_providers[low_voltage_providers.str.contains(country)]
        domestic_consumers = domestic_consumers[domestic_consumers.str.contains(country)]
        industry_consumers = industry_consumers[industry_consumers.str.contains(country)]
    
    # Consumption (negative):
    lv_consumption = -n.links_t.p0[low_voltage_consumers].groupby(rename_low_voltage_techs,axis=1).sum() # From low voltage grid to battery
    domestic_consumption = -n.loads_t.p[domestic_consumers].sum(axis=1)
    industry_consumption = -n.loads_t.p[industry_consumers].sum(axis=1)
    
    # Provision (positive)
    lv_provision = n.links_t.p0[low_voltage_providers]
    lv_provision = lv_provision[lv_provision.columns[~lv_provision.columns.str.endswith('grid')]].groupby(rename_low_voltage_techs,axis=1).sum() # From appliance to low voltage grid
    solar_rooftop = n.generators_t.p[n.generators.query('carrier == "solar rooftop"').index]
    
    if country != 'EU':
        solar_rooftop = solar_rooftop[solar_rooftop.columns[solar_rooftop.columns.str.contains(country)]]
    solar_rooftop = solar_rooftop.sum(axis=1)

    try:
        supply.drop(columns='electricity distribution grid',inplace=True)
    except:
        supply = supply
        
    supply['domestic demand'] = domestic_consumption
    supply['industry demand'] = industry_consumption
    supply['solar rooftop'] = solar_rooftop
    
    for i in lv_consumption.columns:
        supply[i] = lv_consumption[i]
        
    for i in lv_provision.columns:
        supply[i] = lv_provision[i]
    
    return supply

def plot_series(network, country, dstart, dend, tech_colors, moving_average=1, carrier="AC"):

    n = network.copy()
    assign_location(n)
    assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]
    if country != 'EU':
        buses = buses[buses.str.contains(country)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)
                                                                                                                     
                                                                                                                     
            # print(supply)     

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        if country != 'EU':
            comps = comps[comps.str.contains(country)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs, axis=1).sum()

    supply = split_el_distribution_grid(supply,country,n)

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = dstart #pd.Timestamp('2013-01-01')
    stop = dend #pd.Timestamp('2013-12-31')
    
    # start = pd.Timestamp('2013-01-01')
    # stop = pd.Timestamp('2013-12-31')

    threshold = 1e3

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand",
                           "home battery":"battery"},
                  inplace=True)
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["domestic demand",
                                "industry demand",
                                "heat pump",
                                "resistive heater",
                                "BEV",
                                "H2 charging",
                                "nuclear",
                                "hydroelectricity",
                                "wind",
                                "solar PV",
                                "solar rooftop",
                                "CHP",
                                "CHP CC",
                                "biomass",
                                "gas",
                                "home battery",
                                "battery",
                                "V2G",
                                "H2"
                                "solar thermal",
                                "Fischer-Tropsch",
                                "CO2 capture",
                                "CO2 sequestration",
                            ])

    supply =  supply.groupby(supply.columns, axis=1).sum()
    
    supply.index = pd.date_range('1/1/2013','1/1/2014',freq='3h')[0:-1]

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 5))

    supply_temp = supply.rename(columns={'battery storage charging':'battery charging',
                                         'battery storage':'battery',
                                         # 'H2 charging':'H2',
                                         #'hydroelectricity charging':'hydroelectricity',
                                         })
    supply = supply_temp.groupby(by=supply_temp.columns,axis=1).sum()

    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))
    
    supply_plot = supply.loc[start:stop,new_columns].rolling(moving_average).mean()
    supply_plot['hydroelectricity charging'][supply_plot['hydroelectricity charging'] > 0] = 0
    try:
        supply_plot['storage-X charging'][supply_plot['storage-X charging'] > 0] = 0
    except:
        print('No storage-X')
        
    (supply_plot
     .plot(ax=ax, kind="area", stacked=True, linewidth=0.,legend=False,
           color=[tech_colors[i.replace(suffix,"")] for i in new_columns]))

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if item == 'H2 charging' and 'H2' not in labels:
            new_handles.append(handles[i])
            new_labels.append('H2')
            
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    # fig.legend(new_handles, new_labels,loc='lower center', bbox_to_anchor=(0.65, -0.5), prop={'size':15},ncol=3)
    
    ax.set_xlim([start+pd.Timedelta(3*moving_average, "h"), stop])
    ax.set_ylim([-1.1*supply_plot[supply_plot > 0].sum(axis=1).max(), 1.1*supply_plot[supply_plot > 0].sum(axis=1).max()])
    # ax.set_ylim([-1600,1600])
    # xax = ax.get_xaxis()
    # xax.set_tick_params(which='major', pad=-25)
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    
    ydispl = -0.3 if 'BEV' in supply.columns else -0.2
    
    fig.legend(new_handles, new_labels, ncol=3, 
               bbox_to_anchor=(0.15, ydispl), loc=3, frameon=True,prop={'size': 15},borderaxespad=0)
    
    fig.tight_layout()

    return fig,supply

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[
            0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]

def plot_map(network, tech_colors, threshold=10,components=["links", "generators", "storage_units"],
             bus_size_factor=15e4, transmission=False):
   
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    n = network.copy()
    
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    # costs = pd.DataFrame(index=n.buses.index)
    capacity = pd.DataFrame(index=n.buses.index)
    for comp in components:
        df_c = getattr(n, comp)
        if len(df_c) == 0:
            continue # Some countries might not have e.g. storage_units
        
        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"
            
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)
        
        if comp == 'storage_units':
            df_c = df_c.drop(df_c.query('carrier == "PHS"').index)
        
        capacity_c = ((df_c[attr])
                      .groupby([df_c.location, df_c.nice_group]).sum()
                      .unstack().fillna(0.))
        
        if comp == 'generators':
            capacity_c = capacity_c[['solar PV','wind','hydroelectricity']]
            
        elif comp == 'links':
            capacity_c = capacity_c[['OCGT','CCGT','CHP','CHP CC','coal','coal CC','nuclear']]
            
        # costs_c = ((df_c.capital_cost * df_c[attr])
        #            .groupby([df_c.location, df_c.nice_group]).sum()
        #            .unstack().fillna(0.))
        # costs = pd.concat([costs, costs_c], axis=1)
        capacity = pd.concat([capacity, capacity_c], axis=1)
    plot = capacity.groupby(capacity.columns, axis=1).sum() #costs.groupby(costs.columns, axis=1).sum()
    try:
        plot.drop(index=['H2 pipeline',''],inplace=True)
    except:
        print('No H2 pipeline to drop')
    # plot.drop(columns=['electricity distribution grid'],inplace=True) # 'transmission lines'
    plot.drop(columns=plot.sum().loc[plot.sum() < threshold].index,inplace=True)
    technologies = plot.columns
    plot.drop(list(plot.columns[(plot == 0.).all()]), axis=1, inplace=True)
    
    preferred_order = pd.Index(["domestic demand",
                            "industry demand",
                            "heat pump",
                            "resistive heater",
                            "BEV",
                            "H2 charging",
                            "nuclear",
                            "hydroelectricity",
                            "wind",
                            "solar PV",
                            "solar rooftop",
                            "CHP",
                            "CHP CC",
                            "biomass",
                            "gas",
                            "home battery",
                            "battery",
                            "V2G",
                            "H2"
                            "solar thermal",
                            "Fischer-Tropsch",
                            "CO2 capture",
                            "CO2 sequestration",
                        ])
    
    new_columns = ((preferred_order & plot.columns)
                   .append(plot.columns.difference(preferred_order)))
    plot = plot[new_columns]
    for item in new_columns:
        if item not in tech_colors:
            print("Warning!",item,"not in config/plotting/tech_colors")
    plot = plot.stack()  # .sort_index()
    # hack because impossible to drop buses...
    if 'stores' in components:
        n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]
    to_drop = plot.index.levels[0] ^ n.buses.index
    if len(to_drop) != 0:
        print("dropping non-buses", to_drop)
        plot.drop(to_drop, level=0, inplace=True, axis=0)
    # make sure they are removed from index
    plot.index = pd.MultiIndex.from_tuples(plot.index.values)
    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.
    line_upper_threshold = 2e4
    linewidth_factor = 2e3
    ac_color = "gray"
    dc_color = "m"
    links = n.links #[n.links.carrier == 'DC']
    lines = n.lines
    line_widths = lines.s_nom_opt - lines.s_nom_min
    link_widths = links.p_nom_opt - links.p_nom_min
    if transmission:
        line_widths = lines.s_nom_opt
        link_widths = links.p_nom_opt
        # linewidth_factor = 2e3
        line_lower_threshold = 0.
    line_widths[line_widths < line_lower_threshold] = 0.
    link_widths[link_widths < line_lower_threshold] = 0.
    line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    link_widths[link_widths > line_upper_threshold] = line_upper_threshold
    
    fig.set_size_inches(16, 12)
    n.plot(bus_sizes=plot / bus_size_factor,
           bus_colors=tech_colors,
           line_colors=ac_color,
           link_colors=dc_color,
           line_widths=line_widths / linewidth_factor,
           link_widths=link_widths / linewidth_factor,
           ax=ax,  boundaries=(-10, 30, 34, 70),
           color_geomap={'ocean': 'white', 'land': "whitesmoke"})
    for i in technologies:
        ax.plot([0,0],[1,1],label=i,color=tech_colors[i],lw=5)
    fig.legend(bbox_to_anchor=(1.01, 0.6), frameon=False,prop={'size':18})
    # fig.suptitle('Installed power capacities and transmission lines',y=0.92,fontsize=15)
    
    handles = make_legend_circles_for(
        [1e5,1e4], scale=bus_size_factor, facecolor="grey")
    labels = ["    {} GW".format(s) for s in (100,10)]
    l1 = ax.legend(handles, labels,
                   loc="upper left", bbox_to_anchor=(0.01, 0.98),
                   labelspacing=2,
                   frameon=False,
                   title='Generation capacity',
                   fontsize=15,
                   title_fontsize = 15,
                   handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l1)
    handles = []
    labels = []
    for s in (20, 10):
        handles.append(plt.Line2D([0], [0], color=ac_color,
                                  linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} GW".format(s))
    l2 = ax.legend(handles, labels,
                    loc="upper left", bbox_to_anchor=(0.2, 0.98),
                    frameon=False,
                    fontsize=15,
                    title_fontsize = 15,
                    labelspacing=2, handletextpad=1.5,
                    title='    Transmission reinforcement')
    ax.add_artist(l2)

    return fig

def plot_investment_map(network, tech_colors, threshold=10,components=["links", "generators", "storage_units"],
             bus_size_factor=4.3e10, transmission=False):
   
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    n = network.copy()
    
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    costs = pd.DataFrame(index=n.buses.index)
    for comp in components:
        df_c = getattr(n, comp)
        if len(df_c) == 0:
            continue # Some countries might not have e.g. storage_units
        
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)
        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        
        # if comp == 'storage_units':
            # df_c = df_c.drop(df_c.query('carrier == "PHS"').index)
        
        # capacity_c = ((df_c[attr])
        #               .groupby([df_c.location, df_c.nice_group]).sum()
        #               .unstack().fillna(0.))
        
        # if comp == 'generators':
            # capacity_c = capacity_c[['solar PV','wind','hydroelectricity']]
            
        # elif comp == 'links':
            # capacity_c = capacity_c[['OCGT','CCGT','CHP','CHP CC','coal','coal CC','nuclear']]
            
        costs_c = ((df_c.capital_cost * df_c[attr])
                    .groupby([df_c.location, df_c.nice_group]).sum()
                    .unstack().fillna(0.))
        costs = pd.concat([costs, costs_c], axis=1)
    plot = costs.groupby(costs.columns, axis=1).sum()
    try:
        plot.drop(index=['H2 pipeline',''],inplace=True)
    except:
        print('No H2 pipeline to drop')
    # plot.drop(columns=['electricity distribution grid'],inplace=True) # 'transmission lines'
    plot.drop(columns=plot.sum().loc[plot.sum() < threshold].index,inplace=True)
    technologies = plot.columns
    plot.drop(list(plot.columns[(plot == 0.).all()]), axis=1, inplace=True)
    
    preferred_order = pd.Index(["domestic demand",
                            "industry demand",
                            "heat pump",
                            "resistive heater",
                            "BEV",
                            "H2 charging",
                            "nuclear",
                            "hydroelectricity",
                            "wind",
                            "solar PV",
                            "solar rooftop",
                            "CHP",
                            "CHP CC",
                            "biomass",
                            "gas",
                            "home battery",
                            "battery",
                            "V2G",
                            "H2"
                            "solar thermal",
                            "Fischer-Tropsch",
                            "CO2 capture",
                            "CO2 sequestration",
                        ])
    
    new_columns = ((preferred_order & plot.columns)
                   .append(plot.columns.difference(preferred_order)))
    plot = plot[new_columns]
    for item in new_columns:
        if item not in tech_colors:
            print("Warning!",item,"not in config/plotting/tech_colors")
    plot = plot.stack()  # .sort_index()
    # hack because impossible to drop buses...
    if 'stores' in components:
        n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]
    to_drop = plot.index.levels[0] ^ n.buses.index
    if len(to_drop) != 0:
        print("dropping non-buses", to_drop)
        plot.drop(to_drop, level=0, inplace=True, axis=0)
    # make sure they are removed from index
    plot.index = pd.MultiIndex.from_tuples(plot.index.values)
    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.
    line_upper_threshold = 2e4
    # linewidth_factor = 2e3
    ac_color = "gray"
    dc_color = "m"
    links = n.links #[n.links.carrier == 'DC']
    lines = n.lines
    line_widths = lines.s_nom_opt - lines.s_nom_min
    link_widths = links.p_nom_opt - links.p_nom_min
    if transmission:
        line_widths = lines.s_nom_opt
        link_widths = links.p_nom_opt
        # linewidth_factor = 2e3
        line_lower_threshold = 0.
    line_widths[line_widths < line_lower_threshold] = 0.
    link_widths[link_widths < line_lower_threshold] = 0.
    line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    link_widths[link_widths > line_upper_threshold] = line_upper_threshold
    
    fig.set_size_inches(16, 12)
    n.plot(bus_sizes=plot / bus_size_factor,
           bus_colors=tech_colors,
           line_colors=ac_color,
           link_colors=dc_color,
           line_widths=0, #line_widths / linewidth_factor,
           link_widths=0, #link_widths / linewidth_factor,
           ax=ax,  boundaries=(-10, 30, 34, 70),
           color_geomap={'ocean': 'white', 'land': "whitesmoke"})
    for i in technologies:
        ax.plot([0,0],[1,1],label=i,color=tech_colors[i],lw=5)
    fig.legend(bbox_to_anchor=(1.01, 0.9), frameon=False,prop={'size':18})
    
    handles = make_legend_circles_for(
        [40e9,20e9], scale=bus_size_factor, facecolor="grey")
    labels = ["    {} bn Euro".format(s) for s in (40,20)]
    l1 = ax.legend(handles, labels,
                   loc="upper left", bbox_to_anchor=(0.01, 0.98),
                   labelspacing=2,
                   frameon=False,
                   title='Investments',
                   fontsize=15,
                   title_fontsize = 15,
                   handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l1)

    return fig

def plot_storage_map(network, tech_colors, threshold=10,
             bus_size_factor=1e4, transmission=False):
   
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    n = network.copy()
    
    assign_location(n)
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    capacity = pd.DataFrame(index=n.buses.index)

    components = ['stores','storage_units']
    for comp in components:
        df_c = getattr(n, comp)
        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp).replace({'hydroelectricity':'pumped hydro'})
        if comp == 'stores':
            capacity_c = ((df_c[attr])
                          .groupby([df_c.location, df_c.nice_group]).sum()
                          .unstack().fillna(0.))
            try:
                capacity_c = capacity_c.drop(columns=['coal','gas','uranium','oil','biomass'])
            except:
                capacity_c = capacity_c.drop(columns=['coal','gas','uranium'])
            
        else:
            df_c = df_c.query('carrier == "PHS"')
            capacity_c = ((df_c[attr]*df_c['max_hours'])
                          .groupby([df_c.location, df_c.nice_group]).sum()
                          .unstack().fillna(0.))
            
        capacity = pd.concat([capacity, capacity_c], axis=1)
    plot = capacity.groupby(capacity.columns, axis=1).sum()

    plot.drop(columns=plot.sum().loc[plot.sum() < threshold].index,inplace=True)
    technologies = plot.columns
    plot.drop(list(plot.columns[(plot == 0.).all()]), axis=1, inplace=True)
    
    preferred_order = pd.Index(["storage-X",
                                "battery storage",
                                "EV battery",
                                "H2",
                                "hot water storage"])
    
    new_columns = ((preferred_order & plot.columns)
                   .append(plot.columns.difference(preferred_order)))
    plot = plot[new_columns]
    for item in new_columns:
        if item not in tech_colors:
            print("Warning!",item,"not in config/plotting/tech_colors")
    plot = plot.stack()  # .sort_index()
    # hack because impossible to drop buses...
    #if 'stores' in components:
    n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]
    to_drop = plot.index.levels[0] ^ n.buses.index
    if len(to_drop) != 0:
        print("dropping non-buses", to_drop)
        plot.drop(to_drop, level=0, inplace=True, axis=0)
    # make sure they are removed from index
    plot.index = pd.MultiIndex.from_tuples(plot.index.values)
    ac_color = "gray"
    dc_color = "m"
    
    fig.set_size_inches(16, 12)
    n.plot(bus_sizes=plot / bus_size_factor,
           bus_colors=tech_colors,
           line_colors=ac_color,
           link_colors=dc_color,
           line_widths=0, #line_widths / linewidth_factor,
           link_widths=0, #link_widths / linewidth_factor,
           ax=ax,  boundaries=(-10, 30, 34, 70),
           color_geomap={'ocean': 'white', 'land': "whitesmoke"})
    for i in technologies:
        ax.plot([0,0],[1,1],label=i,color=tech_colors[i],lw=5)
    fig.legend(bbox_to_anchor=(1.015, 0.6), frameon=False,prop={'size':18})
    # fig.suptitle('Installed power capacities and transmission lines',y=0.92,fontsize=15)
    
    handles = make_legend_circles_for(
        [1e6,1e5], scale=bus_size_factor, facecolor="grey")
    labels = ["    {} GWh".format(s) for s in (1000,100)]
    l1 = ax.legend(handles, labels,
                   loc="upper left", bbox_to_anchor=(0.01, 0.98),
                   labelspacing=2,
                   frameon=False,
                   title='Energy capacity',
                   fontsize=15,
                   title_fontsize = 15,
                   handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l1)

    return fig
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:54:36 2023

@author: au485969
"""

import pypsa
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# n = pypsa.Network('../networks/high_efficiency/elec_s_y2013_n37_lv1.0__Co2L0.05-3H-T-H-I-B-solar+p3-dist1-X Charge+e1.0-X Charge+c1.0-X Discharge+e1.9-X Discharge+c1.0-X Store+c0.1.nc')
n = pypsa.Network('../networks/high_efficiency/elec_s_y2013_n37_lv1.0__Co2L0.05-3H-T-H-I-B-solar+p3-dist1-X Charge+e1.9-X Charge+c0.1-X Discharge+e1.9-X Discharge+c1.0-X Store+c0.25.nc')

df = pd.DataFrame(index=n.links_t.p0.index) #np.arange(2920)) #

# biomass_links = n.links.query('bus0 == "EU solid biomass"').carrier.unique()

biomass_links = ['solid biomass for industry',
                 'solid biomass for industry CC',
                 'urban central solid biomass CHP',
                 'urban central solid biomass CHP CC', 
                 ]

for biomass_links_0 in biomass_links:
    biomass_links_0_t = n.links_t.p0[n.links.query('carrier == @biomass_links_0').index].sum(axis=1)

    df[biomass_links_0] = biomass_links_0_t/1e6*3 #.sort_values().values/1e6*3
    
df.cumsum().plot.area(stacked=True)
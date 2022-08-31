# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:02:17 2021

@author: au485969
"""

def tech_colors(tech):
    tech_colors = {"onshore wind" : "#2e8b57",
                   "offshore wind" : "#556b2f",
                   'Wind' : "#2e8b57",
                   'wind' : "#2e8b57",
                    "offwind-ac" : "#556b2f",
                    "offshore wind (AC)" : "#556b2f",
                    "offwind-dc" : "#556b2f",
                    "offshore wind (DC)" : "#556b2f",
                    "solar PV" : "#adff2f",
                    "onwind" : "#2e8b57",
                    "offwind" : "#556b2f",
                    "solar" : "#adff2f",
                    "Solar" : "#adff2f",
                    "hydro" : "#1e90ff",
                    "Hydro" : "#1e90ff",
                    "hydro reservoir" : "#3B5323",
                    "ror" : "#78AB46",
                    "run of river" : "#78AB46",
                    "hydroelectricity" : "#006400",
                    "wave" : "#004444",
                    "biomass" : "#cd853f",
                    "biomass CHP" : "#402810",
                    "biomass CHP CC" : "#6e441c",
                    "nuclear" : "deeppink",
                    "gas" : "#a9a9a9",
                    "Gas" : "#a9a9a9",
                    "gas turbine" : "#696969"
                    ,"gas boiler" : "#a9a9a9"
                    ,"oil" : "#696969"
                    ,"coal" : "saddlebrown"
                    ,"lignite" : "#000000"
                    ,"resistive heater" : "#ff8c00"
                    ,"heat pump" : "#ffc34d"
                    ,"hydrogen storage" : "#800080"
                    ,"storage X" : "#800080"
                    ,"battery storage" : "#ffc0cb"
                    ,"hot water storage" : "#ffe0b3"
                    ,"methanation" : "#da70d6"
                    ,"transmission" : "darkorange"
                    ,"transmission-dc":"orangered"
                    ,"transmission-ac":"brown"
                    ,"wind and solar" : "#2e8b57"
                    ,"conventional" : "#a9a9a9"
                    ,"balancing" : "#b300b3"
                    ,"power-to-heat" : "#ff8c00"
                    ,"electricity" : "#0000ff"
                    ,"rural heat" : "#ff0000"
                    ,"urban heat" : "#ffa500"
                    ,"cooling" : "#00aecc"
                    ,"waste" : "#804000"
                    ,"gas CHP" : "#f08080"
                    ,"gas CHP elec" : "#f08080"
                    ,"gas CHP CC" : "#904d4d"
                    ,"gas CHP heat" : "#f08080"
                    ,"central water tank" : "#daa520"
                    ,"electric vehicle" : "#ff4500"
                    ,"geothermal" : "#804000"
                    ,"district heating" : "#ffe0b3"
                    ,"distribution" : "#cc0066"
                    ,"central gas CHP electric" : "#f08080"
                    ,'solar thermal' : 'coral'
                    ,'solar rooftop' : '#e6b800'
                    ,"OCGT marginal" : "sandybrown"
                    ,"OCGT-heat" : "orange"
                    ,"gas boilers" : "orange"
                    ,"gas boiler marginal" : "orange"
                    ,"gas-to-power/heat" : "orange"
                    ,"natural gas" : "brown"
                    ,"SMR" : "#4F4F2F"
                    ,"oil" : "#B5A642"
                    ,"oil boiler" : "#B5A677"
                    ,"lines" : "k"
                    ,"transmission lines" : "#4d004d"
                    ,"H2" : "palegreen"
                    ,"battery" : "#ffc0cb" #"slategray"
                    ,"home battery" : "#614700"
                    ,"home battery storage" : "#614700"
                    ,"Nuclear" : "#b22222"
                    ,"Nuclear marginal" : "#b22222"
                    ,"uranium" : "#b22222"
                    ,"Coal" : "k"
                    ,"Coal marginal" : "k"
                    ,"Lignite" : "grey"
                    ,"lignite" : "grey"
                    ,"Lignite marginal" : "grey"
                    ,"OCGT" : "darkgray"
                    ,"CCGT" : "dimgray"
                    ,"CCGT marginal" : "dimgray"
                    ,"heat pumps" : "#76EE00"
                    ,"heat pump" : "#76EE00"
                    ,"air heat pump" : "#76EE00"
                    ,"ground heat pump" : "#40AA00"
                    ,"resistive heater" : "pink"
                    ,"Sabatier" : "#FF1493"
                    ,"methanation" : "#FF1493"
                    ,"power-to-gas" : "#FF1493"
                    ,"power-to-liquid" : "#FFAAE9"
                    ,"helmeth" : "#7D0552"
                    ,"DAC" : "#E74C3C"
                    ,"co2 stored" : "#123456"
                    ,"CO2 sequestration" : "#123456"
                    ,"CC" : "k"
                    ,"co2" : "#123456"
                    ,"co2 vent" : "#654321"
                    ,"solid biomass for industry co2 from atmosphere" : "#654321"
                    ,"solid biomass for industry co2 to stored": "#654321"
                    ,"gas for industry co2 to atmosphere": "#654321"
                    ,"gas for industry co2 to stored": "#654321"
                    ,"Fischer-Tropsch" : "#44DD33"
                    ,"kerosene for aviation": "#44BB11"
                    ,"naphtha for industry" : "#44FF55"
                    ,"land transport oil" : "#44DD33"
                    ,"water tanks" : "#BBBBBB"
                    ,"hot water storage" : "#BBBBBB"
                    ,"hot water charging" : "#BBBBBB"
                    ,"hot water discharging" : "#999999"
                    ,"CHP" : "r"
                    ,"CHP heat" : "r"
                    ,"CHP electric" : "r"
                    ,"PHS" : "62b1ff"
                    ,"Ambient" : "k"
                    ,"Electric load" : "b"
                    ,"Heat load" : "r"
                    ,"heat" : "darkred"
                    ,"rural heat" : "#880000"
                    ,"central heat" : "#b22222"
                    ,"decentral heat" : "#800000"
                    ,"low-temperature heat for industry" : "#991111"
                    ,"process heat" : "#FF3333"
                    ,"heat demand" : "darkred"
                    ,"electric demand" : "k"
                    ,"Li ion" : "grey"
                    ,"retrofitting" : "purple"
                    ,"building retrofitting" : "purple"
                    ,"BEV charger" : "grey"
                    ,"V2G" : "grey"
                    ,"land transport EV" : "grey"
                    ,"electricity" : "k"
                    ,"gas for industry" : "#333333"
                    ,"solid biomass for industry" : "#555555"
                    ,"industry electricity" : "#222222"
                    ,"industry new electricity" : "#222222"
                    ,"process emissions to stored" : "#444444"
                    ,"process emissions to atmosphere" : "#888888"
                    ,"process emissions" : "#222222"
                    ,"oil emissions" : "#666666"
                    ,"land transport fuel cell" : "#AAAAAA"
                    ,"biogas" : "#800000"
                    ,"solid biomass" : "#DAA520"
                    ,"today" : "#D2691E"
                    ,"shipping" : "#6495ED"
                    ,"electricity distribution grid" : "#333333"}
    color = tech_colors[tech]
    return color
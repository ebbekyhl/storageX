# Storage-X - Cost and efficiency requirements of electricity storage in a highly renewable European energy system

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository includes the scripts and main results covered in the manuscript "Electricity storage cost and efficiency requirements
in a highly renewable European energy system". 

Download network files from Zenodo: XXXX and locate them in the networks folder. The networks folder in this repository contains:

- config.yaml: file used in PyPSA-Eur-Sec to calculate the scenarios with the storage-X sample space as input.
- snakefile: snakemake rules to prepare, solve, and postprocess the networks.
- make_sspace.py: postprocess all solve networks to obtain storage-X metrics. 
- prepare_sector_network.py: scripts to attach sectors. In here, storage-X is attached to the network.
- solve_network.py: script for solving the network.
- costs_2030.csv: csv file obtained with 'PyPSA/technology_data' for the year 2030.

The Jupyter Notebook "Storage-X" directs to the functions creating the main figures of the manuscript. Before creating these visualizations, create the folder "figures" in which figures are saved.

The results folder contains the results (.csv files) obtained with the "make_sspace.py" script. These include:
- sspace_3888.csv: results obtained for the Electricity system with 3888 samples
- sspace_w_sector_coupling_merged.csv: results obtained for the Electricity system, the Electricity + Heating + Land Transport system, and the Fully sector-coupled system based on 758 storage samples (+ additional 324 when allowing technologies with charge energy efficiency above unity)
- simple_sweep/: folder with single-parametric sweeps for 2013 weather year
- weather_sensitivity/: folder with single-parametric sweeps at variable weather years

Clarificaiton of parameter names in the .csv files:
- "eta1 [-]": charge energy efficiency 
- "eta2 [-]": discharge energy efficiency
- "c1": charge power capacity cost (EUR/kW_el)
- "c2": discharge power capacity cost (EUR/kW_el)
- "c_hat [EUR/kWh]": energy capacity cost
- "tau [n_days]": self-discharge time


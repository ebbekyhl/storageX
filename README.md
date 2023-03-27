# Storage-X - Cost and efficiency requirements of electricity storage in a highly renewable European energy system

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository includes the scripts and main results covered in the manuscript ["_Cost and efficiency requirements for a successful electricity storage
in a highly renewable European energy system_"](https://doi.org/10.48550/arXiv.2208.09169). 

The study aims to investigate what a generic long-duration storage technology need to be attributed with in terms of capacity costs and efficiencies. Existing storage options show different characteristics:

![panel_fig1](existing_sketch_all.png)

We use the energy system model __PyPSA-Eur-Sec__ to derive the unique combinations that entail a substantial optimal storage deployment:

![panel_fig2](Spiderweb_result_0_w_TES_[2]_E_cor_Nperf6_w_legend.png)

Subsequently, we list design strategies that storage tech developers can use as a crosshair for further development:

![panel_fig3](figure_panel_requirement.png)

To recreate the figures, the network files containing all data from the energy system model can be acquired from Zenodo: XXXX (will be available upon publication). The The `networks/` folder in this repository contains:

- `config.yaml`: file used in PyPSA-Eur-Sec to calculate the scenarios with the storage-X sample space as input.
- `snakefile`: snakemake rules to prepare, solve, and postprocess the networks.
- `make_sspace.py`: postprocess all solve networks to obtain storage-X metrics. 
- `prepare_sector_network.py`: scripts to attach sectors. In here, storage-X is attached to the network.
- `solve_network.py`: script for solving the network.
- `costs_2030.csv`: csv file obtained with 'PyPSA/technology_data' for the year 2030.

The Jupyter Notebook "Storage-X" directs to the functions creating the main figures of the manuscript. Before creating these visualizations, create the folder "figures" in which figures are saved.

The results folder contains the results (.csv files) obtained with the "make_sspace.py" script. These include:
- `sspace_3888.csv`: results obtained for the Electricity system with 3888 samples
- `sspace_w_sector_coupling_merged.csv`: results obtained for the Electricity system, the Electricity + Heating + Land Transport system, and the Fully sector-coupled system based on 758 storage samples (+ additional 324 when allowing technologies with charge energy efficiency above unity)
- `simple_sweep/`: folder with single-parametric sweeps for 2013 weather year
- `weather_sensitivity/`: folder with single-parametric sweeps at variable weather years

Clarification of parameter names in the .csv files:
- "_eta1 [-]_": charge energy efficiency 
- "_eta2 [-]_": discharge energy efficiency
- "_c1_": charge power capacity cost (EUR/kW_el)
- "_c2_": discharge power capacity cost (EUR/kW_el)
- "_c_hat [EUR/kWh]_": energy capacity cost
- "_tau [n_days]_": self-discharge time


# storageX

Scripts used to create visualizations in the manuscript "Requirements and impacts of energy storage characteristics
in a highly renewable European energy system". 

Download network files from Zenodo: XXXX and locate them in the networks folder. The networks folder in this repository contains:

- config.yaml: file used in PyPSA-Eur-Sec to calculate the scenarios with the storage-X sample space as input.
- snakefile: snakemake rules to prepare, solve, and postprocess the networks.
- make_sspace.py: postprocess all solve networks to obtain storage-X metrics. 
- prepare_sector_network.py: scripts to attach sectors. In here, storage-X is attached to the network.
- solve_network.py: script for solving the network.
- costs_2030.csv: csv file obtained with 'PyPSA/technology_data' for the year 2030.

The Jupyter Notebook "Storage-X" directs to the functions creating the main figures of the manuscript.

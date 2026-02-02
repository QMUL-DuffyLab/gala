# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""
import os
import json
import numpy as np

'''
General stuff
'''
output_dir = os.path.join("out")
spectrum_prefix = 'spectra'
pigment_data_file = os.path.join("pigments", "pigment_data.json")
T=300.0 # temperature (Kelvin)
population_size = 500
max_gen = 500
n_runs = 3
n_rc = 2
n_t_max = 10
# multiplier for cyclic electron flow
alpha = 0.0
# long time to use for diagonalisation
# note that choosing too long a time here (~10^6 s) will blow up
# the diagonalisation because of compounding float rounding errors
# and the explosion in e^{lambda t}
tinf = 100
entropy = 0x87351080e25cb0fad77a44a3be03b491 # from numpy

''' Genetic algorithm stuff '''
selection_strategy = 'ranked'  # options: 'ranked', 'fittest', 'tournament'
reproduction_strategy = 'nads' # options: 'nads', 'steady'
conv_gen = 50 # number of generations without improvement for convergence
conv_per = 0.01 # convergence if max(fitness) within conv_per for conv_gen
fitness_cutoff = 0.2 # fraction of individuals kept if strategy = 'steady'
d_recomb = 0.05 # random perturbation of values during crossover
mu_width = 0.05 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05 # proportion of genomes to mutate after each generation
shift_inc = 1.0 # increment (nm) to shift lineshapes by
crossover_rate = 0.75 # probability that crossover is performed
arith_cross_p = 0.5 # relative weighting of per-RC or per-parameter crossover
# weighting of arithmetic crossover to intermediate crossover
arith_inter_crossover_p = 0.75
mu_type_p = 0.5 # relative weight of gaussian vs. incremental mutations
mu_all_p = 0.5 # percentage of total mutations vs. single parameter mutations

# various base rates for different processes
rates = {
"hop"  : 1.0 / 10.0E-12,
"trap" : 1.0 / 10.0E-12,
"ox"   : 1.0 / 1.0E-3,
"lin"  : 1.0 / 10.0E-3,
"cyc"  : 1.0 / 10.0E-3,
"red"  : 1.0 / 10.0E-3,
"diss" : 1.0 / 1.0E-9,
}

e_donor = -5.10 # eV
e_acceptor = -3.85 # CO2 reduction
l_tilde = 100 # wavenumbers! conversion functions in utils.py

# cross section per pigment - kinda just have to fix this
sig_chl = 9E-20

# stats stuff
hist_snapshot = 5 # generate histograms every hist_snapshot generations

x_lim = [300.0, 2000.0]
dx = 1.0
nx = int((x_lim[1] - x_lim[0]) / dx)

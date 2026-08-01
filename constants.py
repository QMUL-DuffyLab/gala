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
# multiplier for cyclic electron flow
alpha = 0.0
# no longer used - low light factor for quantum efficiency calculation
gamma_fac = 1e-4
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
d_recomb = 0.4 # random perturbation of values during crossover
mu_width = 0.10 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05 # proportion of genomes to mutate after each generation
tourney_k = 5 # selection tournament size
shift_inc = 10.0 # increment (nm) to shift lineshapes by

''' some rates that I think are going to be fixed (all in s^{-1}) '''
# RC specific rates are now in rc.py
tau_prime = 150.0E-12 # transfer from PBS to RCs
tau_hop   = 10.0E-12 # transfer from PBS to RCs
k_hop     = 1.0 / tau_hop # change to tau_prime for PBS simulations
k_diss    = 1.0 / 1.0E-9 # Chl excited state decay rate

# cross section per pigment - kinda just have to fix this
sig_chl = 9E-20

# stats stuff
hist_sub_max = 6 # number of subunits to make histograms for
hist_snapshot = 5 # generate histograms every hist_snapshot generations

'''
Gaussian fits to pigment data, done by me.
details in pigments directory.
taken variously from Nancy Kiang's database
(https://vplapps.astro.washington.edu/pigments)
and PhotochemCAD
(https://www.photochemcad.com)
'''
with open(pigment_data_file, "r") as f:
    pigment_data = json.load(f)

x_lim = [300.0, 2000.0]
dx = 1.0
nx = int((x_lim[1] - x_lim[0]) / dx)

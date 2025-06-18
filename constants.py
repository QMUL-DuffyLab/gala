# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""

import os
import json
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, astuple

'''
General stuff
'''
output_dir = os.path.join("out")
spectrum_prefix = 'spectra'
pigment_data_file = os.path.join("pigments", "pigment_data.json")
T=300.0 # temperature (Kelvin)
gamma_fac = 1e-4 # what to normalise sum(gamma) to for low light calc
population_size = 500
max_gen = 500
n_runs = 3
alpha = 0.0 # multiplier for cyclic electron flow
selection_strategy = 'ranked'  # options: 'ranked', 'fittest', 'tournament'
reproduction_strategy = 'nads' # options: 'nads', 'steady'
conv_gen = 50 # number of generations without improvement for convergence
conv_per = 0.01 # convergence if max(fitness) within conv_per for conv_gen
fitness_cutoff = 0.2 # fraction of individuals kept if strategy = 'steady'
d_recomb = 0.4 # random perturbation of values during crossover
mu_width = 0.10 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05 # proportion of genomes to mutate after each generation
tourney_k = 5 # selection tournament size
hist_snapshot = 5 # generate histograms every hist_snapshot generations
shift_inc = 10.0 # increment to shift lineshapes by

'''
some rates that I think are going to be fixed
all given in s^{-1}
'''

tau_prime = 150.0E-12 # transfer from PBS to RCs
tau_hop   = 10.0E-12 # transfer from PBS to RCs
k_hop     = 1.0 / tau_hop # change to tau_prime for PBS simulations
k_diss    = 1.0 / 1.0E-9 # Chl excited state decay rate
# RC specific rates are now in rc.py

'''
Spectral parameters
'''
sig_chl = 9E-20 #  cross-section of one pigment
np_rc = 10 # number of pigments in reaction centre

# number of subunits to make histograms for
hist_sub_max = 6

binwidths = {'n_b': 1,
        'n_s': 1,
        'n_p': 1,
        'shift': 5,
        'rho': 0.01,
        'aff': 0.01,
        'alpha': 0.01,
        'nu_e': 1.0,
        'fitness': 1.0,
        }

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

x_lim = [400.0, 2000.0]
dx = 1.0
nx = int((x_lim[1] - x_lim[0]) / dx)

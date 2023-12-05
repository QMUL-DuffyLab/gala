# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, fields

'''
General stuff
'''
spectrum_prefix = 'PHOENIX/Scaled_Spectrum_PHOENIX_'
spectrum_suffix = '.dat'
T=300.0 # temperature (Kelvin)
population_size = 1000
fitness_cutoff = 0.2 # fraction of individuals kept
mu_width = 0.25 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05
max_gen = 1000
d_recomb = 0.25 # random perturbation of values during crossover
bounds = {'n_b': np.array([1, 12], dtype=np.int),
          'n_s': np.array([1, 100], dtype=np.int),
          'n_p': np.array([1, 100], dtype=np.int),
          'lp1': np.array([200.0, 1400.0]),
          'lp2': np.array([200.0, 1400.0]),
          'w1': np.array([5.0, 30.0]),
          'w2': np.array([5.0, 30.0]),
          'a12': np.array([0.2, 0.5])}
# need to figure out how to deal with a12 bounds/choices
# if there's only two, could give them here in bounds, then
# use rng.choices to pick one of the two. add condition in
# crossover and mutate functions to select for this

'''
some rates that I think are going to be fixed
all given in s^{-1}
'''

k_diss   = 1.0 / 4.0E-9 # Chl excited state decay rate
k_trap   = 1.0 / 5.0E-12 # PSII trapping rate
k_con    = 1.0 / 10.0E-3 # PSII RC turnover rate
k_hop    = 1.0 / 10.0E-12 # just one hopping rate between all subunits
k_lhc_rc = 1.0 / 10.0E-12
k_params  = (k_diss, k_trap, k_con, k_hop, k_lhc_rc)

'''
Spectral parameters - I think these will change
'''
sig_chl = 1E-20 # (approximate!) cross-section of one chlorophyll
lp_rc = 680.0 # reaction centre
w_rc  = 9.0 # reaction centre peak width
lp2_rc = 640.0 # reaction centre
w2_rc  = 15.0 # reaction centre peak width
a12_rc = 0.2

rc_params = (1, lp_rc, w_rc, lp2_rc, w2_rc, a12_rc)

@dataclass()
class genome:
    n_b: int = 0
    n_s: int = 0
    n_p: npt.NDArray[np.int] = np.empty([], dtype=np.int)
    lp1: npt.NDArray[np.float64] = np.empty([], dtype=np.float64)
    lp2: npt.NDArray[np.float64] = np.empty([], dtype=np.float64)
    w1: npt.NDArray[np.float64] = np.empty([], dtype=np.float64)
    w2: npt.NDArray[np.float64] = np.empty([], dtype=np.float64)
    a12: npt.NDArray[np.float64] = np.empty([], dtype=np.float64)
    nu_e: float = np.nan
    phi_f: float = np.nan

# radiative genome
rg = genome(1, 1, np.array([1]), np.array([680.0]), np.array([10.0]))

# dict to put in dataframe
c_dict = {'T': T, 'population_size': population_size,
          'fitness_cutoff': fitness_cutoff,
          'mu_width': mu_width, 'mu_rate': mu_rate,
          'max_gen': max_gen, 'd_recomb': d_recomb,
          'bounds': bounds, 'k_params': k_params,
          'rc_params': rc_params}

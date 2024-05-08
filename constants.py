# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""

import json
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, astuple

'''
General stuff
'''
output_dir = "out/"
spectrum_prefix = 'spectra/'
phoenix_prefix = 'PHOENIX/Scaled_Spectrum_PHOENIX_'
spectrum_suffix = '.dat'
pigment_data_file = "pigments/pigment_data.json"
T=300.0 # temperature (Kelvin)
gamma_fac = 1e-4 # what to normalise sum(gamma) to for low light calc
population_size = 1000
max_gen = 1000
n_runs = 5
selection_strategy = 'ranked'  # options: 'ranked', 'fittest', 'tournament'
reproduction_strategy = 'nads' # options: 'nads', 'steady'
conv_gen = 50
conv_per = 0.01
fitness_cutoff = 0.2 # fraction of individuals kept
d_recomb = 0.25 # random perturbation of values during crossover
mu_width = 0.10 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05
tourney_k = 5 # selection tournament size
hist_snapshot = 25 # generate histograms every hist_snapshot generations
hist_sub_max = 10 # number of subunits to make histograms for
max_lp_offset = 0.1
# boundaries on the genome parameters. used during generation;
# mutation uses a truncated Gaussian with these as bounds as well.
bounds = {'n_b': np.array([1, 12], dtype=np.int32),
          'n_s': np.array([1, 100], dtype=np.int32),
          'n_p': np.array([1, 100], dtype=np.int32),
          'lp': np.array([-max_lp_offset, max_lp_offset]),
          # note that any pigment in this array can be picked
          # for any subunit; RCs shouldn't be included here.
          # names must match what's in pigment_data_file!
          'pigment': np.array(["bchl_a", "chl_a", "chl_b",
                            "chl_d", "chl_f", "apc",
                            "pc", "r-pe", "c-pe", "b-pe"])}

# list of parameters defined per subunit rather than per genome
# the strings here *must match* the names in genome definition below
# for the generation, crossover and mutation algorithms to work
subunit_params = ['n_p', 'lp', 'pigment']

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

rc_type = 'psii'

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
sig_chl = 9E-20 # (approximate!) cross-section of one chlorophyll
np_rc = 10 # number of pigments in reaction centre

@dataclass(eq=False)
class Genome:
    n_b: int = 0
    n_s: int = 0
    n_p: int = field(default_factory=lambda: np.empty([], dtype=np.int64))
    lp: float = field(default_factory=lambda: np.empty([], dtype=np.float64))
    pigment: str = field(default_factory=lambda: np.empty([], dtype='U10'))
    connected: bool = False
    nu_e: float = np.nan
    phi_e_g: float = np.nan
    phi_e: float = np.nan
    fitness: float = np.nan


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
output_dir = os.path.join("out", "tests")
spectrum_prefix = 'spectra'
pigment_data_file = os.path.join("pigments", "pigment_data.json")
T=300.0 # temperature (Kelvin)
gamma_fac = 1e-4 # what to normalise sum(gamma) to for low light calc
population_size = 1000
max_gen = 1000
n_runs = 5
selection_strategy = 'ranked'  # options: 'ranked', 'fittest', 'tournament'
reproduction_strategy = 'nads' # options: 'nads', 'steady'
conv_gen = 50 # number of generations without improvement for convergence
conv_per = 0.01 # convergence if max(fitness) within conv_per for conv_gen
fitness_cutoff = 0.2 # fraction of individuals kept
d_recomb = 0.25 # random perturbation of values during crossover
mu_width = 0.10 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05 # proportion of genomes to mutate after each generation
tourney_k = 5 # selection tournament size
hist_snapshot = 25 # generate histograms every hist_snapshot generations
hist_sub_max = 10 # number of subunits to make histograms for
max_shift = 0 # maximum shift (nm) of absorption peaks
shift_inc = 0.0 # increment to shift lineshapes by
peak_binwidth = 10.0 # binwidth for histograms of peak absorption per subunit

'''
boundaries on the genome parameters. used during generation;
mutation uses a truncated Gaussian with these as bounds as well.
note that specifying dtype for non-string variables is important;
the type of the numpy arrays here is used to determine what random
function to use and hence how generation/crossover/mutation will work.
'''
bounds = {'n_b': np.array([1, 12], dtype=np.int32),
          'n_s': np.array([1, 20], dtype=np.int32),
          'n_p': np.array([1, 100], dtype=np.int32),
          'shift': np.array([-20, 120], dtype=np.int32),
          # names must match what's in pigment_data_file!
          # 'rc': np.array(["rc_ox", "rc_E", "fr_rc", "ano_rc",
          #                   "hydro_rc"]),
          'rc': np.array(["rc_ox"], dtype='U10'),
          'alpha': np.array([0.0, 10.0], dtype=np.float64),
          'phi': np.array([0.0, 10.0], dtype=np.float64),
          'eta': np.array([0.0, 10.0], dtype=np.float64),
          # any pigment in this array can be picked
          # 'pigment': np.array(["averaged"])
          'pigment': np.array(["r-pe", "pe", "pc", "apc", "chl_b", "chl_a",
              "chl_d", "chl_f", "bchl_a", "bchl_b"],
                              dtype='U10')
          }

# list of parameters defined per subunit rather than per genome
# the strings here *must match* the names in genome definition below
# for the generation, crossover and mutation algorithms to work
# leave 'pigment' in here even if there's only one type, because
# otherwise the pigment arrays don't get updated after mutation etc.
subunit_params = ['n_p', 'pigment', 'shift']

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

# i think these should be sensible
# x_lim = [
# np.min([pigment_data[rc]['abs']['mu'][-1] +
#     (bounds['shift'][0] * shift_inc) for rc in bounds['rc']]),
# np.max([pigment_data[rc]['ems']['mu'][-1] +
#     (bounds['shift'][1] * shift_inc) for rc in bounds['rc']])
# ]
x_lim = [400.0, 800.0]

'''
some rates that I think are going to be fixed
all given in s^{-1}
'''

k_hop    = 1.0 / 10.0E-12 # just one hopping rate between all subunits
k_diss   = 1.0 / 1.0E-9 # Chl excited state decay rate
k_trap   = 1.0 / 10.0E-12 # PSII trapping rate
k_o2     = 1.0 / 400.0e-6
k_lin    = 1.0 / 10.0E-3 # PSII RC turnover rate
k_out    = 1.0 / 10.0E-3 # PSII RC turnover rate
k_params  = (k_diss, k_trap, k_o2, k_lin, k_out, k_hop)

'''
Spectral parameters
'''
sig_chl = 9E-20 #  cross-section of one pigment
np_rc = 10 # number of pigments in reaction centre

@dataclass(eq=False)
class Genome:
    n_b: int = 0
    n_s: int = 0
    n_p: int = field(default_factory=lambda: np.empty([], dtype=np.int64))
    shift: float = field(default_factory=lambda: np.empty([], dtype=np.float64))
    pigment: str = field(default_factory=lambda: np.empty([], dtype='U10'))
    rc: str = field(default_factory=lambda: np.empty([], dtype='U10'))
    alpha: float = 0.0
    phi: float = 0.0
    eta: float = 0.0
    connected: bool = False
    nu_e: float = np.nan
    phi_e_g: float = np.nan
    phi_e: float = np.nan
    fitness: float = np.nan

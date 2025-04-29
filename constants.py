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
output_dir = os.path.join("out", "tests", "stats")
spectrum_prefix = 'spectra'
pigment_data_file = os.path.join("pigments", "pigment_data.json")
T=300.0 # temperature (Kelvin)
gamma_fac = 1e-4 # what to normalise sum(gamma) to for low light calc
population_size = 1000
max_gen = 1000
n_runs = 5
alpha = 0.0 # multiplier for cyclic electron flow
selection_strategy = 'ranked'  # options: 'ranked', 'fittest', 'tournament'
reproduction_strategy = 'nads' # options: 'nads', 'steady'
conv_gen = 50 # number of generations without improvement for convergence
conv_per = 0.01 # convergence if max(fitness) within conv_per for conv_gen
fitness_cutoff = 0.2 # fraction of individuals kept if strategy = 'steady'
d_recomb = 0.25 # random perturbation of values during crossover
mu_width = 0.10 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05 # proportion of genomes to mutate after each generation
tourney_k = 5 # selection tournament size
hist_snapshot = 25 # generate histograms every hist_snapshot generations
max_shift = 0 # maximum shift (nm) of absorption peaks
shift_inc = 0.0 # increment to shift lineshapes by
peak_binwidth = 10.0 # binwidth for histograms of peak absorption per subunit

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

'''
boundaries on the genome parameters. used during generation;
mutation uses a truncated Gaussian with these as bounds as well.
specifying the type via dtype= is important;
the type of the numpy arrays here is used to determine what random
function to use and hence how generation/crossover/mutation will work.
The types should match those given in the dataclass definition below!
'''
n_p_total = 200 # total number of "pigments" in the PBS
bounds = {'n_b': np.array([1, 12], dtype=np.int32),
          'n_s': np.array([1, 20], dtype=np.int32),
          'n_p': np.array([1, 100], dtype=np.int32),
          'shift': np.array([-1, 1], dtype=np.int32),
          # any pigment in this array can be picked
          'pigment': np.array(["r-pe", "pe", "pc", "apc", "chl_b", "chl_a",
              "chl_d", "chl_f", "bchl_a", "bchl_b"],
                              dtype='U10'),
          'rc': np.array(["ox", "frl", "anox", "exo"], dtype='U10'),
          'rho': np.array([0.0, 1.0], dtype=np.float64),
          'aff': np.array([0.0, 100.0], dtype=np.float64),
          'alpha': np.array([0.0, 1.0], dtype=np.float64),
          'nu_e': np.array([0.0, 100.0], dtype=np.float64),
          'phi_e_g': np.array([0.0, 1.0], dtype=np.float64),
          'phi_e': np.array([0.0, 1.0], dtype=np.float64),
          'fitness': np.array([0.0, 100.0], dtype=np.float64),
          }

# number of subunits to make histograms for
hist_sub_max = bounds['n_s'][1] if bounds['n_s'][1] < 10 else 10

'''
these are for the stats functions. every hist_snapshot generations
we take distributions of the Genome parameters. `bounds` above is
used to determine the range over which the histogram is binned; the
binwidth is (obviously) the binwidth for those histograms. if you don't
give one, it'll use the default 50 bins for the whole range and print
a warning, so it'll still do the histogram, just might not be as useful
'''
binwidths = {'n_b': 1,
        'n_s': 1,
        'n_p': 1,
        'shift': 5,
        'rho': 0.01,
        'aff': 0.01,
        'alpha': 0.01,
        'nu_e': 1.0,
        'phi_e': 0.01,
        'phi_e_g': 0.01,
        'fitness': 1.0,
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

x_lim = [400.0, 2000.0]
dx = 1.0
nx = int((x_lim[1] - x_lim[0]) / dx)

'''
finally the dataclass definition. the names of the members here
should match the names in bounds and binwidths above, and the
types should match those given in bounds as well. if all is well,
the genetic algorithm will automatically do the right things for
generation, crossover and mutation, and the stats functions will
automatically do the stats correctly.
'''
@dataclass(eq=False)
class Genome:
    n_b: int = 0
    n_s: int = 0
    n_p: np.ndarray     = field(default_factory=lambda:
                        np.empty([], dtype=np.int64))
    shift: np.ndarray   = field(default_factory=lambda:
                        np.empty([], dtype=np.float64))
    pigment: np.ndarray = field(default_factory=lambda:
                        np.empty([], dtype='U10'))
    rc: str = ""
    rho: np.ndarray     = field(default_factory=lambda:
                        np.empty([], dtype=np.int64))
    aff: np.ndarray     = field(default_factory=lambda:
                        np.empty([], dtype=np.int64))
    alpha: float = 0.0
    connected: bool = False
    nu_e: float = np.nan
    phi_e_g: float = np.nan
    phi_e: float = np.nan
    fitness: float = np.nan

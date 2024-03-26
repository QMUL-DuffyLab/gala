# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""

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
T=300.0 # temperature (Kelvin)
gamma_fac = 1e-4 # what to normalise sum(gamma) to for low light calc
population_size = 1000
max_gen = 1000
n_runs = 5
selection_strategy = 'ranked'  # options: 'ranked', 'fittest', 'tournament'
reproduction_strategy = 'nads' # options: 'nads', 'steady'
conv_gen = 50
conv_per = 0.01
cost_per_pigment = 0.01
fitness_cutoff = 0.2 # fraction of individuals kept
d_recomb = 0.25 # random perturbation of values during crossover
mu_width = 0.10 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05
tourney_k = 5 # selection tournament size
hist_snapshot = 50 # generate histograms every hist_snapshot generations
hist_sub_max = 10 # number of subunits to make histograms for
max_lp_offset = 10.0
bounds = {'n_b': np.array([1, 12], dtype=np.int32),
          'n_s': np.array([1, 100], dtype=np.int32),
          'n_p': np.array([1, 100], dtype=np.int32),
          'lp': np.array([-max_lp_offset, max_lp_offset]),
          'pigment': np.array(["bchl_a", "chl_a", "chl_b",
                            "chl_d", "chl_f", "r_apc",
                            "r_pc", "r_pe"])}

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
# lp_rc = 680.0 # reaction centre
# w_rc  = 9.0 # reaction centre peak width
# lp2_rc = 640.0 # reaction centre
# w2_rc  = 15.0 # reaction centre peak width
# a12_rc = 0.2

# rc_params = (1, lp_rc, w_rc, lp2_rc, w2_rc, a12_rc)

def array_safe_eq(a, b) -> bool:
    """Check equality of a and b"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    try:
        return a == b
    except TypeError:
        return NotImplemented

def dc_eq(dc1, dc2) -> bool:
    """check equality of dataclasses"""
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented
    t1 = astuple(dc1)
    t2 = astuple(dc2)
    return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))

@dataclass(eq=False)
class Genome:
    n_b: int = 0
    n_s: int = 0
    n_p: int = field(default_factory=lambda: np.empty([], dtype=np.int64))
    lp: float = field(default_factory=lambda: np.empty([], dtype=np.float64))
    pigment: str = field(default_factory=lambda: np.empty([], dtype='U10'))
    nu_e: float = np.nan
    phi_f_g: float = np.nan
    phi_f: float = np.nan
    fitness: float = np.nan
    def __eq__(self, other):
        return dc_eq(self, other)

# list of parameters defined per subunit rather than per genome
# the strings here *must match* the names in genome definition above
subunit_params = ['n_p', 'lp', 'pigment']

'''
Gaussian fits to pigment data, done by me.
details in pigments directory.
taken variously from Nancy Kiang's database
(https://vplapps.astro.washington.edu/pigments)
and PhotochemCAD
(https://www.photochemcad.com)
'''
pigment_data = {
'rc': {
    'n_gauss': 2,
    'amp': [1.00e+00, 2.00e-01],
    'lp': [6.80e+02, 6.40e+02],
    'w': [9.00e+00, 1.5e+01],
    'text': r'$ \text{RC} $',
    },
'bchl_a': {
    'n_gauss': 3,
    'amp': [9.327708e-01, 1.868076e-01, 2.200819e-01],
    'lp': [7.801765e+02, 7.357920e+02, 6.110172e+02],
    'w': [1.351709e+01, 3.463805e+01, 1.840540e+01],
    'text': r'$ \text{BChl}_{a} $',
   },
'chl_a': {
    'n_gauss': 2,
    'amp': [8.724538e-01, 2.245409e-01],
    'lp': [6.661618e+02, 6.250687e+02],
    # 'w': [8.878352e+00, 3.685004e+01],
    'w': [10.0, 3.685004e+01],
    'text': r'$ \text{Chl}_{a} $',
    },
'chl_b': {
    'n_gauss': 3,
    'amp': [9.138061e-01, 1.695460e-01, 1.633274e-01],
    'lp': [6.514902e+02, 6.052384e+02, 5.821592e+02],
    'w': [1.164620e+01, 1.445649e+01, 5.696023e+01],
    'text': r'$ \text{Chl}_{b} $',
    },
'chl_d': {
    'n_gauss': 2,
    'amp': [8.543801e-01, 2.434290e-01],
    'lp': [6.988473e+02, 6.570787e+02],
    'w': [1.189424e+01, 4.001474e+01],
    'text': r'$ \text{Chl}_{d} $',
    },
'chl_f': {
    'n_gauss': 2,
    'amp': [8.745562e-01, 1.972934e-01],
    'lp': [7.094259e+02, 6.666164e+02],
    'w': [1.332439e+01, 4.319979e+01],
    'text': r'$ \text{Chl}_{f} $',
    },
'r_apc': {
    'n_gauss': 2,
    'amp': [7.133819e-01, 6.393681e-01],
    'lp': [6.512198e+02, 6.130897e+02],
    'w': [8.502225e+00, 3.140512e+01],
    'text': r'$ \text{APC} $',
    },
'r_pc': {
    'n_gauss': 2,
    'amp': [9.007754e-01, 5.814556e-01],
    'lp': [6.167174e+02, 5.558603e+02],
    'w': [1.616777e+01, 3.300788e+01],
    'text': r'$ \text{PC} $',
    },
'r_pe': {
    'n_gauss': 3,
    'amp': [6.199999e-01, 9.10000e-01, 6.35000e-01],
    'lp': [5.67000e+02, 5.37000e+02, 4.95000e+02],
    'w': [9.000000e+00, 2.200000e+01, 9.199999e+00],
    'text': r'$ \text{PE} $',
    },
}

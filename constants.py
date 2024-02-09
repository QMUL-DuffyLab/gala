# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

'''
General stuff
'''
spectrum_prefix = 'PHOENIX/Scaled_Spectrum_PHOENIX_'
spectrum_suffix = '.dat'
T=300.0 # temperature (Kelvin)
gamma_fac = 1e-4 # what to normalise sum(gamma) to for low light calc
population_size = 1000
max_gen = 1000
n_runs = 3
fitness_cutoff = 0.2 # fraction of individuals kept
d_recomb = 0.25 # random perturbation of values during crossover
mu_width = 0.25 # width of Gaussian/Poisson we draw from for mutation
mu_rate = 0.05
hist_snapshot = 50 # generate histograms every hist_snapshot generations
hist_sub_max = 10 # number of subunits to make histograms for
bounds = {'n_b': np.array([1, 12], dtype=np.int32),
          'n_s': np.array([1, 100], dtype=np.int32),
          'n_p': np.array([1, 100], dtype=np.int32),
          'lp': np.array([200.0, 1400.0]),
          'pigment': np.array(["bchl_a\x00", "chl_a\x00", "chl_b\x00",
                            "chl_d\x00", "chl_f\x00", "r_apc\x00",
                            "r_pc\x00", "r_pe\x00"])}

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
np_rc = 1 # number of pigments in reaction centre
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
    n_p: int = field(default_factory=lambda: np.empty([], dtype=np.int64))
    lp: float = field(default_factory=lambda: np.empty([], dtype=np.float64))
    pigment: str = field(default_factory=lambda: np.empty([], dtype='U10'))
    nu_e: float = np.nan
    phi_f: float = np.nan

# radiative genome
rg = genome(1, 1, np.array([1]), np.array([680.0]), np.array(['chl_a\x00']))

pigment_data = {
'rc': {
    'n_gauss': 2,
    'amp': [1.00e+00, 2.00e-01],
    'lp': [6.80e+02, 6.40e+02],
    'w': [9.00e+00, 1.5e+01]
    },
'bchl_a': {
    'n_gauss': 3,
    'amp': [9.327708e-01, 1.868076e-01, 2.200819e-01],
    'lp': [7.801765e+02, 7.357920e+02, 6.110172e+02],
    'w': [1.351709e+01, 3.463805e+01, 1.840540e+01]
   },
'chl_a': {
    'n_gauss': 2,
    'amp': [8.724538e-01, 2.245409e-01],
    'lp': [6.661618e+02, 6.250687e+02],
    'w': [8.878352e+00, 3.685004e+01],
    },
'chl_b': {
    'n_gauss': 3,
    'amp': [9.138061e-01, 1.695460e-01, 1.633274e-01],
    'lp': [6.514902e+02, 6.052384e+02, 5.821592e+02],
    'w': [1.164620e+01, 1.445649e+01, 5.696023e+01],
    },
'chl_d': {
    'n_gauss': 2,
    'amp': [8.543801e-01, 2.434290e-01],
    'lp': [6.988473e+02, 6.570787e+02],
    'w': [1.189424e+01, 4.001474e+01],
    },
'chl_f': {
    'n_gauss': 2,
    'amp': [8.745562e-01, 1.972934e-01],
    'lp': [7.094259e+02, 6.666164e+02],
    'w': [1.332439e+01, 4.319979e+01],
    },
'r_apc': {
    'n_gauss': 2,
    'amp': [7.133819e-01, 6.393681e-01],
    'lp': [6.512198e+02, 6.130897e+02],
    'w': [8.502225e+00, 3.140512e+01],
    },
'r_pc': {
    'n_gauss': 2,
    'amp': [9.007754e-01, 5.814556e-01],
    'lp': [6.167174e+02, 5.558603e+02],
    'w': [1.616777e+01, 3.300788e+01],
    },
'r_pe': {
    'n_gauss': 3,
    'amp': [6.199999e-01, 9.10000e-01, 6.35000e-01],
    'lp': [5.67000e+02, 5.37000e+02, 4.95000e+02],
    'w': [9.000000e+00, 2.200000e+01, 9.199999e+00],
    },
}

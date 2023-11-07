# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:49:39 2023

@author: callum
"""

import numpy as np
import scipy as sp
from scipy.constants import h as h, c as c, Boltzmann as kb
import Lattice_antenna as lattice
import constants

ts = 2600

spectrum_file = constants.spectrum_prefix + \
                '{:4d}K'.format(ts) + \
                constants.spectrum_suffix
l, ip_y = np.loadtxt(spectrum_file, unpack=True)

'''
reaction centre and rates - I think most of this is fixed within the
genetic algorithm, so load from the constants file.
NB: rc_params[0] is N_RC, the number of reaction centres, which I think
we assume is always 1?
'''
rc_params = (1, constants.sig, constants.lp_rc, constants.w_rc)
k_params  = (constants.k_diss,constants.k_trap,constants.k_con,
             constants.k_hop,constants.k_lhc_rc)

'''
now branch_params setup
I think the majority of the parameter space we're exploring will be
within branch_params - changing number of branches, size of block etc
'''
branch_params=[] # will be a list of how long each branch is, I think?

# now we'll set up one individual with something like
# individual = lattice.Antenna_branched_funnel(l, ip_y, branch_params,
                                             rc_params, k_params, constants.T)

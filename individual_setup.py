# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:49:39 2023

@author: callum
"""

from scipy.constants import h as h, c as c, Boltzmann as kb
import numpy as np
import scipy as sp
import Lattice_antenna as lattice
import constants

rng = np.random.default_rng()

# these two will be args eventually i guess
ts = 2600
init_type = 'radiative' # can be radiative or random

spectrum_file = constants.spectrum_prefix \
                + '{:4d}K'.format(ts) \
                + constants.spectrum_suffix
l, ip_y = np.loadtxt(spectrum_file, unpack=True)

def generate_random_subunit():
    '''
    Generates a completely random subunit, with a random number
    of pigments, random cross-section, random absorption peak and
    random width.
    '''
    n_pigments  = rng.integers(1, constants.max_size)
    sigma       = constants.sig_chl
    lambda_peak = rng.uniform(constants.lambda_min, constants.lambda_max)
    width       = rng.uniform(constants.width_min, constants.width_max)
    return (n_pigments, sigma, lambda_peak, width)

def initialise_individual(init_type):
    '''
    Initialise one individual from our population.
    There are two ways to do this - either assume they all
    have identical prototypical antennae, or they're all
    completely random. Option controlled by changing init_type.
    '''
    if init_type == 'radiative':
        '''
        Assumes every individual at the start is an
        identical kind of prototypical antenna with
        one branch, one subunit, one Chl-like pigment.
        '''
        return [1, constants.radiative_subunit]
    else: # random initialisation
        '''
        Each branch is (currently) assumed to be identical!
        First randomise n_branches, then n_subunits.
        Then generate n_subunits using generate_random_subunit.
        '''
        nb = rng.integers(1, constants.max_size)
        branch_params = [nb]
        ns = rng.integers(1, constants.max_size)
        for i in range(ns):
            branch_params.append(generate_random_subunit())
        return branch_params

for i in range(constants.n_individuals):
    bp = initialise_individual('random')
    # now we'll set up one individual with something like
    # individual = lattice.Antenna_branched_funnel(l, ip_y, branch_params,
    #              constants.rc_params, constants.k_params, constants.T)

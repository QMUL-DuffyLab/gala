# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:49:39 2023

@author: callum
"""

from scipy.constants import h as h, c as c, Boltzmann as kb
from operator import itemgetter
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
    n_pigments  = rng.integers(1, constants.n_p_max)
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
        NB: if we do this we can just calculate nu_e for this 
        setup once - they'll all have the same nu_e so we'll
        need to think of an alternative fitness strategy here
        '''
        return [1, constants.radiative_subunit]
    else: # random initialisation
        '''
        Each branch is (currently) assumed to be identical!
        First randomise n_branches, then n_subunits.
        Then generate n_subunits using generate_random_subunit.
        '''
        nb = rng.integers(1, constants.n_b_max)
        branch_params = [nb]
        ns = rng.integers(1, constants.n_s_max)
        for i in range(ns):
            branch_params.append(generate_random_subunit())
        return branch_params

def selection(population, results):
    n_parents = int(constants.fitness_cutoff * constants.n_individuals)
    '''
    pull out the nu_e values and their indices in the results,
    then sort them in descending order (reverse=True) by nu_e.
    then the first n_parents of nu_es_sorted are the highest nu_e values
    and we can pull them from the population using the corresponding indices.
    '''
    nu_es_sorted = sorted([(i, r['nu_e'] * r['phi_F'])
                          for i, r in enumerate(results)],
                          key=itemgetter(1), reverse=True)
    print(nu_es_sorted)
    best_ind = nu_es_sorted[0][0]
    best = (population[best_ind],
           (results[best_ind]['nu_e'], results[best_ind]['phi_F']))
    parents = []
    for i in range(n_parents):
       parents.append(population[nu_es_sorted[i][0]])
    return parents, best

population = []
results = []
running_best = []
for i in range(constants.n_individuals):
    bp = initialise_individual('random')
    population.append(bp)
    print(i, bp[0], len(bp) - 1, pow(bp[0] * len(bp) - 1, 2))
    results.append(lattice.Antenna_branched_overlap(l, ip_y, bp,
                                                   constants.rc_params,
                                                   constants.k_params,
                                                   constants.T))

'''
NB: in antenna branched i think we want
N_eq = torch.linalg.solve(K_mat, gamma_vec)
'''
print([(r['nu_e'], r['phi_F']) for r in results])
parents, best = selection(population, results)
running_best.append(best)
print(parents)
print(best)

# -*- coding: utf-8 -*-
"""
06/11/2023
@author: callum
"""

from operator import itemgetter
from scipy.constants import h as h, c as c, Boltzmann as kb
import ctypes
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import Lattice_antenna as lattice
import constants
import antenna_torch as at
import timeit

rng = np.random.default_rng()

def generate_random_subunit():
    '''
    Generates a completely random subunit, with a random number
    of pigments, random cross-section, random absorption peak and
    random width.
    '''
    n_pigments  = rng.integers(constants.n_p_bounds[0],
                               constants.n_p_bounds[1])
    sigma       = constants.sig_chl
    lambda_peak = rng.uniform(constants.lambda_bounds[0],
                              constants.lambda_bounds[1])
    width       = rng.uniform(constants.width_bounds[0],
                              constants.width_bounds[1])
    return [n_pigments, sigma, lambda_peak, width]

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
        nb = rng.integers(constants.n_b_bounds[0],
                          constants.n_b_bounds[1])
        branch_params = [nb]
        ns = rng.integers(constants.n_s_bounds[0],
                          constants.n_s_bounds[1])
        for i in range(ns):
            branch_params.append(generate_random_subunit())
        return branch_params

def selection(population, results):
    '''
    given a population and the calculated results (nu_e, quantum efficiency,
    etc.), pick out the top fraction based on a cutoff given in constants.py.
    Return these along with the very best individual and its score.
    Note - I feel like normally they're picked probabilistically based on
    a fitness criterion, rather than just deterministically taking the best?
    '''
    n_survivors = int(constants.fitness_cutoff * constants.n_individuals)
    '''
    pull out nu_e and efficiency values and their indices in the results,
    then sort them in descending order (reverse=True) by the product of these.
    then the first n_survivors of nu_es_sorted are the highest values,
    and we can pull them from the population using the corresponding indices.
    '''
    nu_es_sorted = sorted([(i, r['nu_e'] * r['phi_F'])
                          for i, r in enumerate(results)],
                          key=itemgetter(1), reverse=True)
    best_ind = nu_es_sorted[0][0]
    best = (population[best_ind],
           (results[best_ind]['nu_e'], results[best_ind]['phi_F']))
    survivors = []
    for i in range(n_survivors):
       survivors.append(population[nu_es_sorted[i][0]])
    return survivors, best

def reproduction(survivors, population):
    '''
    Given the survivors of a previous iteration, generate the necessary
    number of children and return the new population.
    Characteristics are taken randomly from each parent as much as possible.
    '''
    n_children = constants.n_individuals - len(survivors)
    for i in range(len(survivors)):
        population[i] = survivors[i]

    for i in range(n_children):
        child   = []
        # print("CHILD ", i)
        # pick two different parents from the survivors
        p_i = rng.choice(len(survivors), 2, replace=False)
        # print("parent indices = ", p_i)
        parents = [survivors[p_i[0]], survivors[p_i[1]]]
        # NB: this doesn't work very well since everything's a weird
        # mixed list of scalars and tuples etc. can probably be tidied later
        # parents = np.array(survivors)[rng.integers(0, len(survivors), size=2)]
        nb_c = parents[rng.integers(2)][0] # individual[0] = nb
        ns_p = [len(p) - 1 for p in parents]
        ns_c = ns_p[rng.integers(2)] # = ns of random parent
        # print(ns_p, ns_c, np.min(ns_p), np.argmin(ns_p))
        child.append(nb_c)
        for j in range(1, ns_c + 1):
            if j > np.min(ns_p):
                choice = np.argmax(ns_p)
                sub = parents[choice][j]
                # print(j, choice, sub)
                child.append(sub)
            else:
                choice = rng.integers(2)
                sub = parents[choice][j]
                # print(j, choice, sub)
                child.append(sub)
        # print("child ", i, ": ", child)
        population[i + len(survivors)] = child
    return population

def mutation(individual):
    '''
    Perform the mutation step of the genetic algorithm.
    We do this by looping over each mutable parameter and selecting from
    a truncated normal distribution, rounding to int if necessary.
    The truncation bounds for each parameter are given mostly by chemical
    argument (e.g. no physical pigment has an absorption peak 1000nm wide),
    and the width of the distribution is a hyperparameter of our choice
    derived from the current value of the parameter - currently 0.1 * value.
    I think it's acceptable to just loop over all parameters since they
    are all independent quantities?
    '''
    # NB: if I set up a class for an individual, this is easier to do;
    # then i can just use the name of each parameter and then get the type
    # to decide whether or not we need to round the result
    current = individual[0]
    scale = current * constants.mutation_width
    b = (constants.n_b_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    individual[0] = new
    print("Branch mutation - ", current, b, new)
    # n_s
    current = len(individual) - 1
    scale = current * constants.mutation_width
    b = (constants.n_s_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    print("Subunit mutation - ", current, b, new)
    if current > new:
        # add subunits as necessary
        # assume new subunits are also random? this is a meaningful choice
        for i in range(current - new):
            individual.append(generate_random_subunit())
    elif current < new:
        # delete the last (new - current) elements
        # this would fail if current = new, hence the inequality
        del individual[-(new - current):]
    # now it gets more involved - we have to pick a random subunit
    # to apply these last three mutations to.
    # note that this is also a choice about how the algorithm works,
    # and it's not the only possible way to apply a mutation!
    # n_pigments
    s = rng.integers(1, len(individual))
    current = individual[s][0]
    scale = current * constants.mutation_width
    b = (constants.n_p_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    print("n_p mutation - ", individual[s], current, b, new)
    individual[s][0] = new
    # lambda_peak
    s = rng.integers(1, len(individual))
    current = individual[s][2]
    scale = current * constants.mutation_width
    b = (constants.lambda_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    print("l_p mutation - ", individual[s], current, b, new)
    individual[s][2] = new
    # width
    s = rng.integers(1, len(individual))
    current = individual[s][3]
    scale = current * constants.mutation_width
    b = (constants.width_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    print("width mutation - ", individual[s], current, b, new)
    individual[s][2] = new

    return individual

if __name__ == "__main__":
    c_antenna = ctypes.cdll.LoadLibrary("./libantenna.so")
    c_antenna.antenna.argtypes = [ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]

    # these two will be args eventually i guess
    ts = 2600
    init_type = 'radiative' # can be radiative or random

    spectrum_file = constants.spectrum_prefix \
                    + '{:4d}K'.format(ts) \
                    + constants.spectrum_suffix
    # instead of unpacking and pulling both arrays, pull once
    # and split by column. this stops the arrays being strided,
    # so we can use them in the C code below without messing around
    phoenix_data = np.loadtxt(spectrum_file)
    l    = phoenix_data[:, 0]
    ip_y = phoenix_data[:, 1]
    l_c = np.ctypeslib.as_ctypes(np.ascontiguousarray(l))
    ip_y_c = np.ctypeslib.as_ctypes(np.ascontiguousarray(ip_y))
    l_t = torch.from_numpy(l)
    ip_y_t = torch.from_numpy(ip_y)

    population = []
    chris_results = []
    torch_results = []
    c_results = []
    running_best = []
    chris_time = 0.0
    torch_time = 0.0
    c_time = 0.0
    for i in range(constants.n_individuals):
        bp = initialise_individual('random')
        population.append(bp)
        print(i, bp[0], len(bp) - 1, pow(bp[0] * len(bp) - 1, 2))
        print("Calling Chris's code")
        chris_start = timeit.default_timer()
        chris_result = lattice.Antenna_branched_overlap(l, ip_y, bp,
                                                    constants.rc_params,
                                                    constants.k_params,
                                                    constants.T)
        chris_time += timeit.default_timer() - chris_start
        np.savetxt("out/twa_mat_chris.dat", chris_result['TW_Adj_mat'])
        np.savetxt("out/k_mat_chris.dat", chris_result['K_mat'])

        print("Calling torch code")
        torch_start = timeit.default_timer()
        torch_result = at.antenna(l_t, ip_y_t, bp)
        torch_time += timeit.default_timer() - torch_start
        np.savetxt("out/twa_mat_torch.dat", torch_result['TW_Adj_mat'].numpy())
        np.savetxt("out/k_mat_torch.dat", torch_result['K_mat'].numpy())
        # check the matrices and steady state solution are the same!
        try:
            assert np.allclose(chris_result['TW_Adj_mat'], torch_result['TW_Adj_mat'].numpy())
        except AssertionError:
            print("Chris:")
            print(chris_result['TW_Adj_mat'])
            print("Torch:")
            print(torch_result['TW_Adj_mat'].numpy())
            print("Max difference: ", np.max(chris_result['TW_Adj_mat'] - torch_result['TW_Adj_mat'].numpy()))
        
        assert np.allclose(chris_result['K_mat'], torch_result['K_mat'].numpy())
        assert np.allclose(chris_result['N_eq'], torch_result['N_eq'])

        '''
        setup for calling C version
        '''
        n_b = bp[0]
        subunits = bp[1:]
        n_s = len(subunits)
        side = (n_b * n_s) + 2
        n_p   = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.uint32))
        lp    = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.float64))
        width = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.float64))
        n_p[0]   = constants.rc_params[0]
        lp[0]    = constants.rc_params[2]
        width[0] = constants.rc_params[3]
        for i in range(n_s):
            n_p[i + 1]   = subunits[i][0]
            lp[i + 1]    = subunits[i][2]
            width[i + 1] = subunits[i][3]
        # twa    = np.ctypeslib.as_ctypes(np.zeros((side, side), dtype=np.float64))
        # twa = ((ctypes.c_double * side) * side)()
        # n_eq   = np.ctypeslib.as_ctypes(np.zeros(side, dtype=np.float64))
        n_eq   = (ctypes.c_double * side)()
        nu_phi = np.ctypeslib.as_ctypes(np.zeros(2, dtype=np.float64))
        kp = (ctypes.c_double * len(constants.k_params))(*constants.k_params)

        # start timer here to time the actual function only lol
        print("Calling C code")
        c_start = timeit.default_timer()

        c_antenna.antenna(l_c, ip_y_c,
                ctypes.c_double(constants.sig_chl),
                ctypes.c_double(constants.rc_params[1]), kp,
                ctypes.c_double(constants.T),
                n_p, lp, width,
                ctypes.c_uint(n_b), ctypes.c_uint(n_s),
                ctypes.c_uint(len(l)), n_eq, nu_phi)
        print("Done")
        print(chris_result['N_eq'])
        print("n_eq from C")
        print(np.ctypeslib.as_array(n_eq))
        print("nu_e, phi_f")
        print(np.ctypeslib.as_array(nu_phi))
        c_time += timeit.default_timer() - c_start
        # assert np.allclose(chris_result['N_eq'], n_eq)
        chris_results.append(chris_result)
        torch_results.append(torch_result)

    print("Chris's code time: ", chris_time)
    print("Torch code time: ", torch_time)
    survivors, best = selection(population, torch_results)
    # print("---------\nSURVIVORS\n---------")
    # print(survivors)
    # running_best.append(best)
    # new_pop = reproduction(survivors, population)
    # print("---------\n NEW POP \n---------")
    # print(population)
    # for i in range(constants.n_individuals):
    #     population[i] = mutation(population[i])
    # print("---------\nMUTATIONS\n---------")
    # print(population)

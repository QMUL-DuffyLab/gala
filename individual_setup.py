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
    How is this done? Is there are a random chance to do each?
    i feel like that's probably the obvious way to do it. in
    that case, what are the chances of each? weighting them is
    just introducing more hyperparameters.
    we do the mutations by centreing a Gaussian/Poisson distribution
    on the current value for each parameter that can mutate, and then
    selecting from that distribution. The width of this distribution
    is set to a fraction of the current value and the fraction is
    a hyperparameter; I'm gonna set it (preliminary) to 0.1.
    This should stop the number of branches from varying wildly while
    also allowing the subunit parameters to change relatively freely
    '''
    # NB: if I set up a class for an individual, this is easier to do;
    # then i can just check dtype of the relevent class member to figure
    # out which kind of distribution to use
    possible_mutations = ['branch', 'subunit', 'n_pigments',
                          'lambda_peak', 'width']
    # we need a way to get the current value of whatever trait we're mutating
    # - this is kind of annoying to do with the code structured as it is
    choice = rng.integers(len(possible_mutations))
    if possible_mutations[choice] == 'branch':
        current = individual[0]
        # NB: need to figure out what distribution we're going for here!
        # Poisson doesn't work - need decoupled mean and width
        new = rng.poisson(current, int(current * constants.mutation_width))
        individual[0] = new
        print("Branch mutation - ", current, new)
    elif possible_mutations[choice] == 'subunit':
        current = len(individual) - 1
        new = rng.poisson(current, int(current * constants.mutation_width))
        print("Subunit mutation - ", current, new)
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
    elif possible_mutations[choice] == 'n_pigments':
        s = rng.integers(1, len(individual))
        current = individual[s][0]
        new = rng.poisson(current, int(current * constants.mutation_width))
        print("n_p mutation - ", individual[s], current, new)
        individual[s][0] = new
    elif possible_mutations[choice] == 'lambda_peak':
        s = rng.integers(1, len(individual))
        current = individual[s][2]
        new = rng.normal(current, current * constants.mutation_width)
        print("l_p mutation - ", individual[s], current, new)
        individual[s][2] = new
    elif possible_mutations[choice] == 'width':
        s = rng.integers(1, len(individual))
        current = individual[s][3]
        new = rng.normal(current, current * constants.mutation_width)
        print("width mutation - ", individual[s], current, new)
        individual[s][2] = new
    return individual

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

survivors, best = selection(population, results)
print("---------\nSURVIVORS\n---------")
print(survivors)
running_best.append(best)
new_pop = reproduction(survivors, population)
print("---------\n NEW POP \n---------")
print(population)
for i in range(constants.n_individuals):
    population[i] = mutation(population[i])
print("---------\nMUTATIONS\n---------")
print(population)

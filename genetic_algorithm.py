# -*- coding: utf-8 -*-
"""
22/11/2023
@author: callum
"""

from operator import itemgetter
import numpy as np
import scipy.stats as ss
import constants

def generate_random_subunit(rng):
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

def initialise_individual(rng, init_type):
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
            branch_params.append(generate_random_subunit(rng))
        return branch_params

def selection(rng, population):
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
    nu_es_sorted = sorted([(i, r['nu_e'] * r['phi_f'])
                          for i, r in enumerate(population)],
                          key=itemgetter(1), reverse=True)
    best_ind = nu_es_sorted[0][0]
    best = (population[best_ind])
    survivors = []
    for i in range(n_survivors):
       survivors.append(population[nu_es_sorted[i][0]])
    return survivors, best

def reproduction(rng, survivors, population):
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
        # pick two different parents from the survivors
        p_i = rng.choice(len(survivors), 2, replace=False)
        parents = [survivors[p_i[0]], survivors[p_i[1]]]
        # NB: this doesn't work very well since everything's a weird
        # mixed list of scalars and tuples etc. can probably be tidied later
        nb_c = parents[rng.integers(2)]['params'][0] # individual[0] = nb
        ns_p = [len(p['params']) - 1 for p in parents]
        ns_c = ns_p[rng.integers(2)] # = ns of random parent
        # print("repro - n_sp = ", ns_p, "n_sc = ", ns_c)
        child.append(nb_c)
        for j in range(1, ns_c + 1):
            if j > np.min(ns_p):
                choice = np.argmax(ns_p)
                sub = parents[choice]['params'][j]
                child.append(sub)
            else:
                choice = rng.integers(2)
                sub = parents[choice]['params'][j]
                child.append(sub)
        population[i + len(survivors)] = {'params': child,
                                          'nu_e': np.nan,
                                          'phi_f': np.nan}
    return population

def mutation(rng, individual, n_s_changes):
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
    c = individual['params']
    p = individual['params'].copy()
    current = p[0]
    scale = current * constants.mutation_width
    b = (constants.n_b_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    p[0] = new
    # print("Branch mutation - ", current, b, new)
    # n_s
    current = len(p) - 1
    scale = current * constants.mutation_width
    b = (constants.n_s_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    # print("Subunit mutation - ", current, new)
    if current < new:
        # add subunits as necessary
        # assume new subunits are also random? this is a meaningful choice
        n_s_changes[0] += new - current
        for i in range(new - current):
            p.append(generate_random_subunit(rng))
        # print("add: old = ", oldlen, " new = ", len(p) - 1, "c, n = ", current, new)
    elif current > new:
        # delete the last (new - current) elements
        n_s_changes[1] += current - new
        # this would fail if current = new, hence the inequality
        del p[-(current - new):]
        # print("del: old = ", oldlen, " new = ", len(p) - 1, "c, n = ", current, new)
    # now it gets more involved - we have to pick a random subunit
    # to apply these last three mutations to.
    # note that this is also a choice about how the algorithm works,
    # and it's not the only possible way to apply a mutation!
    # n_pigments
    s = rng.integers(1, len(p))
    current = p[s][0]
    scale = current * constants.mutation_width
    b = (constants.n_p_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    # print("n_p mutation - ", p[s], current, b, new)
    p[s][0] = new
    # lambda_peak
    s = rng.integers(1, len(p))
    current = p[s][2]
    scale = current * constants.mutation_width
    b = (constants.lambda_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    # print("l_p mutation - ", p[s], current, b, new)
    p[s][2] = new
    # width
    s = rng.integers(1, len(p))
    current = p[s][3]
    scale = current * constants.mutation_width
    b = (constants.width_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    # print("width mutation - ", p[s], current, b, new)
    p[s][3] = new

    individual['params'] = p
    return individual

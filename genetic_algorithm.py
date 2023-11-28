# -*- coding: utf-8 -*-
"""
22/11/2023
@author: callum
"""

from operator import itemgetter
from dataclasses import fields
import numpy as np
import scipy.stats as ss
import constants

def generate_random_subunit(rng):
    '''
    Generates a completely random subunit, with a random number
    of pigments, random cross-section, random absorption peak and
    random width.
    '''
    n_p = rng.integers(*constants.bounds['n_p'])
    lp  = rng.uniform(*constants.bounds['lp'])
    w   = rng.uniform(*constants.bounds['w'])
    # sigma       = constants.sig_chl
    return [n_p, lp, w]

def fill_arrays(rng, l, res_length, name, t):
    if (t == "int"):
        dt = np.int32
        fn = rng.integers
    else:
        dt = np.float64
        fn = rng.uniform
    result = np.zeros((2, res_length), dtype=dt)
    for i in range(res_length):
        for j in range(2):
            if i < len(l[j]):
                result[j][i] = l[j][i]
            else:
                result[j][i] = fn(*constants.bounds[name])
    return result

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
        return constants.rg
    else: # random initialisation
        '''
        Each branch is (currently) assumed to be identical!
        First randomise n_branches, then n_subunits.
        Then generate n_s random subunits
        '''
        nb = rng.integers(*constants.bounds['n_b'])
        ns = rng.integers(*constants.bounds['n_s'])
        n_p = np.zeros(ns, dtype=np.int)
        lp = np.zeros(ns, dtype=np.float64)
        w  = np.zeros(ns, dtype=np.float64)
        for i in range(ns):
            n_p[i] = rng.integers(*constants.bounds['n_p'])
            lp[i]  = rng.uniform(*constants.bounds['lp'])
            w[i]   = rng.uniform(*constants.bounds['w'])
        return constants.genome(nb, ns, n_p, lp, w)

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
    nu_es_sorted = sorted([(i, r.nu_e * r.phi_f)
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
        # NB: this still needs tidying up, horrible mix of dicts etc
        # pick two different parents from the survivors
        p_i = rng.choice(len(survivors), 2, replace=False)
        parents = [survivors[p_i[i]] for i in range(2)]
        # n_b
        bounds = constants.bounds['n_b']
        vals = [p.n_b for p in parents]
        b = rng.uniform(-constants.d_re, 1 + constants.d_re)
        new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        while new < bounds[0] or new > bounds[1]:
            b = rng.uniform(-constants.d_re, 1 + constants.d_re)
            new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        n_b = new
        # n_s
        bounds = constants.bounds['n_s']
        vals = [p.n_s for p in parents]
        b = rng.uniform(-constants.d_re, 1 + constants.d_re)
        new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        while new < bounds[0] or new > bounds[1]:
            b = rng.uniform(-constants.d_re, 1 + constants.d_re)
            new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        n_s = new
        '''
        next we have to pick n_p, lp and w for each subunit on the child
        we first have to check n_s of the child against its parents
        then if one or both of the parents is shorter, we extend its
        arrays (n_p, lp, w) until they match the length of the child,
        and finally perform the same intermediate recombination on
        the result to get values for the child.
        '''
        # n_p, lp, w
        lps_temp  = [p.lp for p in parents]
        ws_temp   = [p.w for p in parents]
        n_ps_temp = [p.n_p for p in parents]
        n_ps = fill_arrays(rng, n_ps_temp, n_s, 'n_p', 'int')
        lps = fill_arrays(rng, lps_temp, n_s, 'lp', 'float')
        ws = fill_arrays(rng, ws_temp, n_s, 'w', 'float')
        n_p = np.zeros(n_s, dtype=np.int32)
        lp = np.zeros(n_s, dtype=np.float64)
        w = np.zeros(n_s, dtype=np.float64)
        # n_ps = [n_ps_temp[k][j] if j < len(n_ps_temp[k])
        #         else rng.integers(*constants.bounds['n_p'])
        #         for k in range(2) for j in range(c.n_s)]
        # arr_temp = [n_ps, lps, ws]
        # keys = ['n_p', 'lp', 'w']
        # arr = [arr_temp[m][k][j] if j < len(arr_temp[m][k])
        #         else rng.integers(*constants.bounds[keys[m]])
        #         for k in range(2) for m in range(3)]
        # print("n_s = ", n_s, "shape(n_s array) = ", n_ps.shape)
        for j in range(n_s):
            bounds = constants.bounds['n_p']
            vals = np.array([n_ps[k][j] for k in range(2)])
            b = rng.uniform(-constants.d_re, 1 + constants.d_re)
            new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
            while new < bounds[0] or new > bounds[1]:
                b = rng.uniform(-constants.d_re, 1 + constants.d_re)
                new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
            n_p[j] = new

            bounds = constants.bounds['lp']
            vals = np.array([lps[k][j] for k in range(2)])
            b = rng.uniform(-constants.d_re, 1 + constants.d_re)
            new = vals[0] * b + vals[1] * (1 - b)
            while new < bounds[0] or new > bounds[1]:
                b = rng.uniform(-constants.d_re, 1 + constants.d_re)
                new = vals[0] * b + vals[1] * (1 - b)
            lp[j] = new

            bounds = constants.bounds['w']
            vals = np.array([ws[k][j] for k in range(2)])
            b = rng.uniform(-constants.d_re, 1 + constants.d_re)
            new = vals[0] * b + vals[1] * (1 - b)
            while new < bounds[0] or new > bounds[1]:
                b = rng.uniform(-constants.d_re, 1 + constants.d_re)
                new = vals[0] * b + vals[1] * (1 - b)
            w[j] = new

        # for field in constants.bounds.keys():
        # '''
        # intermediate recombination. for each field we can
        # crossover, we take a mix of the two parents' values
        # including a factor b which expands the range of the
        # value a little, and helps to offset the tendency for
        # values to reduce between generations. we also have to
        # remember to check the bounds for each field!
        # '''
        #     c = constants.genome()
        #     vals = [getattr(p, field) for p in parents]
        #     if type(vals[0]) in (float, int):
        #         shape = None
        #     else:
        #         shape = vals[0].shape
        #     bounds = constants.bounds[field]
        #     b = rng.uniform(-constants.d_re, 1 + constants.d_re,
        #                     size=shape)
        #     new = vals[0] * b + vals[1] * (1 - b)
        #     while new < bounds[0] or new > bounds[1]:
        #         b = rng.uniform(-constants.d_re, 1 + constants.d_re,
        #                         size=shape)
        #         new = vals[0] * b + vals[1] * (1 - b)
        #     setattr(c, field, new.astype(vals[0].dtype))
        #     print("field: ", field, "vals: ", vals, "b: ", b,
        #           "c val: ", getattr(c, field))
        # population[i + len(survivors)] = c
        population[i + len(survivors)] = constants.genome(n_b, n_s, n_p, lp, w)
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
    current = individual.n_b
    scale = current * constants.mutation_width
    b = (constants.n_b_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    individual.n_b = new
    # print("Branch mutation - ", current, b, new)
    # n_s
    current = individual.n_s
    scale = current * constants.mutation_width
    b = (constants.n_s_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    # print("Subunit mutation - ", current, new)
    if current < new:
        # add subunits as necessary
        # assume new subunits are also random? this is a meaningful choice
        n_s_changes[0] += new - current
        # print("len(n_p) before addition = ", individual.n_p.shape)
        individual.n_p.resize(new)
        individual.lp.resize(new)
        individual.w.resize(new)
        for i in range(new - current):
            individual.n_s += 1
            individual.n_p[-(i + 1)] = rng.integers(*constants.bounds['n_p'])
            individual.lp[-(i + 1)] = rng.uniform(*constants.bounds['lp'])
            individual.w[-(i + 1)] = rng.uniform(*constants.bounds['w'])
        # print("add: old = ", current, " new = ", individual.n_s,
        #       "c, n = ", current, new, "len(n_p) = ", individual.n_p.shape)
    elif current > new:
        # delete the last (new - current) elements
        n_s_changes[1] += current - new
        # this would fail if current = new, hence the inequality
        # this can probably be replaced by np.delete(arr, new-current)
        # print("len(n_p) before subtraction = ", individual.n_p.shape)
        # individual.n_p.resize(new)
        # individual.lp.resize(new)
        # individual.w.resize(new)
        for i in range(current - new):
            individual.n_s -= 1
            individual.n_p = np.delete(individual.n_p, -1)
            individual.lp = np.delete(individual.lp, -1)
            individual.w = np.delete(individual.w, -1)
        # print("add: old = ", current, " new = ", individual.n_s,
        #       "c, n = ", current, new, "len(n_p) = ", individual.n_p.shape)
    # now it gets more involved - we have to pick a random subunit
    # to apply these last three mutations to.
    # note that this is also a choice about how the algorithm works,
    # and it's not the only possible way to apply a mutation!
    # n_pigments
    s = rng.integers(1, individual.n_s) if individual.n_s > 1 else 0
    current = individual.n_p[s]
    scale = current * constants.mutation_width
    b = (constants.n_p_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    new = new.round().astype(int)
    # print("n_p mutation - ", p[s], current, b, new)
    individual.n_p[s] = new
    # lambda_peak
    s = rng.integers(1, individual.n_s) if individual.n_s > 1 else 0
    current = individual.lp[s]
    scale = current * constants.mutation_width
    b = (constants.lambda_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    # print("l_p mutation - ", p[s], current, b, new)
    individual.lp[s] = new
    # width
    s = rng.integers(1, individual.n_s) if individual.n_s > 1 else 0
    current = individual.w[s]
    scale = current * constants.mutation_width
    b = (constants.width_bounds - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current, scale=scale, random_state=rng)
    # print("width mutation - ", p[s], current, b, new)
    individual.w[s] = new

    return individual

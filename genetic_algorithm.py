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

def fill_arrays(rng, l, res_length, name):
    if (isinstance(constants.bounds[name][0], (int, np.integer))):
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
    n_survivors = int(constants.fitness_cutoff * constants.population_size)
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

def crossover(child, parents, parameter, rng, subunit):
    '''
    '''
    d = constants.d_recomb
    bounds = constants.bounds[parameter]
    parent_vals = [getattr(p, parameter) for p in parents]

    if isinstance(bounds[0], (int, np.integer)):
        var_type = np.int
    else:
        var_type = np.float

    if subunit:
        s = child.n_s
        # print(parameter, s, [p.n_s for p in parents])
        vals = fill_arrays(rng, parent_vals, s, parameter)
    else:
        s = 1
        vals = parent_vals
    if s == 1:
        b = rng.uniform(-d, 1 + d)
        new = vals[0] * b + vals[1] * (1 - b)
        while new < bounds[0] or new > bounds[1]:
            b = rng.uniform(-d, 1 + d)
            new = vals[0] * b + vals[1] * (1 - b)
        if isinstance(bounds[0], (int, np.integer)):
            new = np.round(new).astype(int)
    else:
        new = np.zeros(s, dtype=var_type)
        for i in range(s):
            # we want to loop here since each value of b should be
            # different; otherwise we have to find a value of b where
            # every element of new is within bounds, which would
            # significantly reduce the amount of variation we can have
            v = [vals[j][i] for j in range(2)]
            b = rng.uniform(-d, 1 + d)
            n = v[0] * b + v[1] * (1 - b)
            while n < bounds[0] or n > bounds[1]:
                b = rng.uniform(-d, 1 + d)
                n = v[0] * b + v[1] * (1 - b)
            if isinstance(bounds[0], (int, np.integer)):
                n = np.round(n).astype(int)
            new[i] = n
    setattr(child, parameter, new)

def reproduction(rng, survivors, population):
    '''
    Given the survivors of a previous iteration, generate the necessary
    number of children and return the new population.
    Characteristics are taken randomly from each parent as much as possible.
    '''
    d = constants.d_recomb
    n_children = constants.population_size - len(survivors)
    for i in range(len(survivors)):
        population[i] = survivors[i]

    for i in range(n_children):
        # old_child = population[i + len(survivors)]
        child = population[i + len(survivors)]
        # NB: this still needs tidying up, horrible mix of dicts etc
        # pick two different parents from the survivors
        p_i = rng.choice(len(survivors), 2, replace=False)
        parents = [survivors[p_i[i]] for i in range(2)]
        # n_b
        crossover(child, parents, 'n_b', rng, False)
        # bounds = constants.bounds['n_b']
        # vals = [p.n_b for p in parents]
        # b = rng.uniform(-d, 1 + d)
        # new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        # while new < bounds[0] or new > bounds[1]:
        #     b = rng.uniform(-d, 1 + d)
        #     new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        # n_b = new
        # n_s
        # print("before n_s crossover", old_child.n_s, child.n_s)
        crossover(child, parents, 'n_s', rng, False)
        # print("after n_s crossover", old_child.n_s, child.n_s)
        # print(old_child.n_s)
        # bounds = constants.bounds['n_s']
        # vals = [p.n_s for p in parents]
        # b = rng.uniform(-d, 1 + d)
        # new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        # while new < bounds[0] or new > bounds[1]:
        #     b = rng.uniform(-d, 1 + d)
        #     new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        # n_s = new
        '''
        next we have to pick n_p, lp and w for each subunit on the child
        we first have to check n_s of the child against its parents
        then if one or both of the parents is shorter, we extend its
        arrays (n_p, lp, w) until they match the length of the child,
        and finally perform the same intermediate recombination on
        the result to get values for the child.
        '''
        # n_p, lp, w
        # print("before n_p crossover", old_child.n_p, child.n_p)
        crossover(child, parents, 'n_p', rng, True)
        # print("after n_p crossover", old_child.n_p, child.n_p)
        crossover(child, parents, 'lp', rng, True)
        crossover(child, parents, 'w', rng, True)
        # lps_temp  = [p.lp for p in parents]
        # ws_temp   = [p.w for p in parents]
        # n_ps_temp = [p.n_p for p in parents]
        # n_ps = fill_arrays(rng, n_ps_temp, n_s, 'n_p', 'int')
        # lps = fill_arrays(rng, lps_temp, n_s, 'lp', 'float')
        # ws = fill_arrays(rng, ws_temp, n_s, 'w', 'float')
        # n_p = np.zeros(n_s, dtype=np.int32)
        # lp = np.zeros(n_s, dtype=np.float64)
        # w = np.zeros(n_s, dtype=np.float64)
        # for j in range(n_s):
        #     bounds = constants.bounds['n_p']
        #     vals = np.array([n_ps[k][j] for k in range(2)])
        #     b = rng.uniform(-d, 1 + d)
        #     new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        #     while new < bounds[0] or new > bounds[1]:
        #         b = rng.uniform(-d, 1 + d)
        #         new = np.round(vals[0] * b + vals[1] * (1 - b)).astype(int)
        #     n_p[j] = new

        #     bounds = constants.bounds['lp']
        #     vals = np.array([lps[k][j] for k in range(2)])
        #     b = rng.uniform(-d, 1 + d)
        #     new = vals[0] * b + vals[1] * (1 - b)
        #     while new < bounds[0] or new > bounds[1]:
        #         b = rng.uniform(-d, 1 + d)
        #         new = vals[0] * b + vals[1] * (1 - b)
        #     lp[j] = new

        #     bounds = constants.bounds['w']
        #     vals = np.array([ws[k][j] for k in range(2)])
        #     b = rng.uniform(-d, 1 + d)
        #     new = vals[0] * b + vals[1] * (1 - b)
        #     while new < bounds[0] or new > bounds[1]:
        #         b = rng.uniform(-d, 1 + d)
        #         new = vals[0] * b + vals[1] * (1 - b)
        #     w[j] = new
        # population[i + len(survivors)] = constants.genome(n_b, n_s, n_p, lp, w)
        assert child == population[i + len(survivors)]
        population[i + len(survivors)] = child
    return population

def mutate(genome, parameter, rng, subunit=None):
    '''
    mutate a parameter with a given name.
    getattr/setattr can be used to get the right dataclass fields.
    if the parameter we're mutating is a per-subunit one, index into
    getattr to get the right element. we also need to check type,
    since if we're mutating n_s we need to have integer extents and
    indices into the resulting arrays.
    '''
    if subunit is not None:
       current = getattr(genome, parameter)[subunit]
    else:
       current = getattr(genome, parameter)
    scale = current * constants.mu_width
    b = (constants.bounds[parameter] - current) / (scale)
    new = ss.truncnorm.rvs(b[0], b[1], loc=current,
                           scale=scale, random_state=rng)
    if isinstance(current, (int, np.integer)):
        new = new.round().astype(int)
    if subunit is not None:
        getattr(genome, parameter)[subunit] = new
    else:
        setattr(genome, parameter, new)

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
    mutate(individual, 'n_b', rng, None)
    # n_s - we also have to update arrays here
    current = individual.n_s
    mutate(individual, 'n_s', rng, None)
    new = individual.n_s
    if current < new:
        # add subunits as necessary
        # assume new subunits are also random? this is a meaningful choice
        n_s_changes[0] += new - current
        individual.n_p.resize(new)
        individual.lp.resize(new)
        individual.w.resize(new)
        for i in range(new - current):
            # individual.n_s += 1
            individual.n_p[-(i + 1)] = rng.integers(*constants.bounds['n_p'])
            individual.lp[-(i + 1)] = rng.uniform(*constants.bounds['lp'])
            individual.w[-(i + 1)] = rng.uniform(*constants.bounds['w'])
    elif current > new:
        # delete the last (new - current) elements
        n_s_changes[1] += current - new
        # this can probably be replaced by np.delete(arr, new-current)
        for i in range(current - new):
            # individual.n_s -= 1
            individual.n_p = np.delete(individual.n_p, -1)
            individual.lp = np.delete(individual.lp, -1)
            individual.w = np.delete(individual.w, -1)
    # now pick a random subunit to apply these mutations to.
    # note that this is also a choice about how the algorithm works,
    # and it's not the only possible way to apply a mutation!
    # n_pigments, lambda_peak, width
    for p in ['n_p', 'lp', 'w']:
        s = rng.integers(1, individual.n_s) if individual.n_s > 1 else 0
        mutate(individual, p, rng, s)
    return individual

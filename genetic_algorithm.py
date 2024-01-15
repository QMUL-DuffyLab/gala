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

def get_type(parameter):
    ''' get parameter type to declare numpy array correctly '''
    test = constants.bounds[parameter][0]
    if (isinstance(test, (int, np.integer))):
        dt = np.int32
    elif (isinstance(test, (float, np.float))):
        dt = np.float64
    elif (isinstance(test, (str, np.str_))):
        dt = 'U10'
    else:
        raise TypeError("get_type cannot determine parameter type.")
    return dt

def get_rand(parameter, rng):
    ''' get a single random number or choice depending on parameter '''
    bounds = constants.bounds[parameter]
    if (isinstance(bounds[0], (int, np.integer))):
        r = rng.integers(*bounds)
    elif (isinstance(bounds[0], (float, np.float))):
        r = rng.uniform(*bounds)
    elif (isinstance(bounds[0], (str, np.str_))):
        r = rng.choice(bounds)
    else:
        raise TypeError("get_rand cannot determine parameter type.")
    return r

def fill_arrays(rng, parent_values, res_length, parameter):
    '''
    the number of subunits a child has might be smaller
    or larger than one or both of its parents. we make sure
    we have two arrays of the relevant parameter that are
    the right length for the child and then perform recombination
    elementwise on those. fill_arrays takes the values from
    the parents where they exist, else it generates parameter
    values randomly. NB: if we relax the assumption that every
    branch is identical this will stop working. but then so
    will literally everything else in the code, come to think of it
    '''
    dt = get_type(parameter)
    result = np.zeros((2, res_length), dtype=dt)
    for i in range(res_length):
        for j in range(2):
            if i < len(parent_values[j]):
                result[j][i] = parent_values[j][i]
            else:
                # new subunits are just copies of tail subunits
                result[j][i] = parent_values[j][len(parent_values[j]) - 1]
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
        nb = get_rand('n_b', rng)
        ns = get_rand('n_s', rng)
        n_p = np.zeros(ns, dtype=np.int)
        lp = np.zeros(ns, dtype=np.float64)
        pigment = np.empty(ns, dtype='U10')
        for i in range(ns):
            n_p[i] = get_rand('n_p', rng)
            lp[i]  = get_rand('lp', rng)
            pigment[i]  = get_rand('pigment', rng)
        return constants.genome(nb, ns, n_p, lp, pigment)

def fitness(individual):
    nu_e_weight = 1.0
    phi_f_weight = 1.0
    return (nu_e_weight * individual.nu_e) * (phi_f_weight * individual.phi_f)

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
    apply fitness function to the entire population (defined above),
    then sort them in descending order (reverse=True) by the product of these.
    then the first n_survivors of nu_es_sorted are the highest values,
    and we can pull them from the population using the corresponding indices.
    '''
    nu_es_sorted = sorted([(i, fitness(r))
                          for i, r in enumerate(population)],
                          key=itemgetter(1), reverse=True)
    best_ind = nu_es_sorted[0][0]
    best = (population[best_ind])
    survivors = []
    for i in range(n_survivors):
       survivors.append(population[nu_es_sorted[i][0]])
    return survivors, best

def recombine(vals, parameter, rng):
    '''
    perform intermediate recombination of parameters
    where possible, and a straight choice in the case of
    pigment type, for a single subunit.
    '''
    d = constants.d_recomb
    bounds = constants.bounds[parameter]
    if parameter == 'pigment': # binary choice
        new = rng.choice(vals)
    else:
        b = rng.uniform(-d, 1 + d)
        new = vals[0] * b + vals[1] * (1 - b)
        while new < bounds[0] or new > bounds[1]:
            b = rng.uniform(-d, 1 + d)
            new = vals[0] * b + vals[1] * (1 - b)
        if isinstance(bounds[0], (int, np.integer)):
            new = np.round(new).astype(int)
    return new

def crossover(child, parents, parameter, rng, subunit):
    '''
    '''
    parent_vals = [getattr(p, parameter) for p in parents]
    dt = get_type(parameter)
    if subunit:
        '''
        if this is a per-subunit parameter, the number of subunits on
        the child might be larger or smaller than one or both parents,
        so we generate two arrays of the correct size to recombine from.
        see fill_arrays for more explanation
        '''
        s = child.n_s
        vals = fill_arrays(rng, parent_vals, s, parameter)
    else:
        s = 1
        vals = parent_vals
    if s == 1:
        new = recombine(vals, parameter, rng)
    else:
        new = np.zeros(s, dtype=dt)
        for i in range(s):
            '''
            we need to loop here since each value of b in the recombine
            function should be different; otherwise we have to find
            a value of b where every element of new is within bounds,
            which significantly reduces the possible variation
            '''
            v = [vals[j][i] for j in range(2)]
            new[i] = recombine(v, parameter, rng)
    setattr(child, parameter, new)

def reproduction(rng, survivors, population):
    '''
    Given the survivors of a previous iteration, generate the necessary
    number of children and return the new population.
    Characteristics are taken randomly from each parent as much as possible.
    '''
    n_children = constants.population_size - len(survivors)
    for i in range(len(survivors)):
        population[i] = survivors[i]

    for i in range(n_children):
        child = population[i + len(survivors)]
        # pick two different parents from the survivors
        p_i = rng.choice(len(survivors), 2, replace=False)
        parents = [survivors[p_i[i]] for i in range(2)]
        crossover(child, parents, 'n_b', rng, False)
        crossover(child, parents, 'n_s', rng, False)
        for p in ['n_p', 'lp', 'pigment']:
            crossover(child, parents, p, rng, True)
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
    if parameter == 'pigment':
        new = rng.choice(constants.bounds[parameter])
    else:
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
    # bit ugly but i haven't thought of a neater way yet
    current = individual.n_s
    mutate(individual, 'n_s', rng, None)
    new = individual.n_s
    per_sub_params = ['n_p', 'lp', 'pigment']
    if current < new:
        # add subunits as necessary
        # NB: we assume new subunits are copies of the tail subunit
        n_s_changes[0] += new - current
        for p in per_sub_params:
            c = getattr(individual, p)
            # i think setting refcheck to false is fine?
            # it seems to work, and i only reference it here
            c.resize(new, refcheck=False)
            for i in range(new - current):
                c[-(i + 1)] = c[current - 1]
    elif current > new:
        # delete the last (new - current) elements
        n_s_changes[1] += current - new
        for p in per_sub_params:
            np.delete(getattr(individual, p), new - current)
    # now pick a random subunit to apply these mutations to.
    # note that this is also a choice about how the algorithm works,
    # and it's not the only possible way to apply a mutation!
    for p in per_sub_params:
        s = rng.integers(1, individual.n_s) if individual.n_s > 1 else 0
        mutate(individual, p, rng, s)
    return individual

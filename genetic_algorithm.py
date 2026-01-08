# -*- coding: utf-8 -*-
"""
22/11/2023
@author: callum
"""
import dataclasses
import hashlib
import numpy as np
import scipy.stats as ss
import constants

it = np.int64
ft = np.float64
gt = np.dtype([
        ('dE0',  ft, (constants.n_rc)),
        ('i',    ft, (constants.n_rc)),
        ('k_cs', ft, (constants.n_rc)),
        ('n_t',  it, (constants.n_rc)),
        ('k',    ft, (constants.n_rc, constants.n_t_max)),
        ('e',    ft, (constants.n_rc, constants.n_t_max)),
        ])
# these are bounds on each element for array variables
# they have to match the dict in make_arrays
# NB: [0.4, 4.0]eV ~= [3000.0, 300.0]nm wavelength light
bounds = {
        'dE0':  np.array([0.4, 4.0], dtype=ft),
        'i':    np.array([0.0, 10.0], dtype=ft),
        'k_cs': np.array([1.0, 1.0E12], dtype=ft),
        'n_t':  np.array([1, 10], dtype=it),
        'k':    np.array([1.0, 1.0E12], dtype=ft),
        'e':    np.array([-4.0, -0.4], dtype=ft),
        }

def fix_matrices(rng, p):
    '''
    k and e are a matrix (n_rc, n_t_max) for each individual,
    but the actual number of energies and rates for each row
    should be n_t[i]. when reproducing or mutating, the elements
    of n_t can change - set the rest to np.nan to make sure they
    don't get used by accident anywhere
    '''
    nt = p['n_t']
    for i, nti in enumerate(nt):
        krow = p['k'][i]
        erow = p['e'][i]
        for j in range(nti):
            # if n_t has mutated, it might've increased, in which
            # case there will be too many nans in k and e. fix this
            if np.isnan(krow[j]):
                krow[j] = get_rand(rng, 'k')
                erow[j] = get_rand(rng, 'e')
        krow[nti:] = np.nan
        erow[nti:] = np.nan

def get_rand(rng, parameter, size=None):
    '''
    get random value(s) based on parameter type
    '''
    b = bounds[parameter]
    t = b.dtype
    if t == it:
        r = rng.integers(*b, endpoint=True, size=size)
    elif t == ft:
        r = rng.uniform(*b, size=size)
    else:
        raise TypeError(f"get_rand failed on {parameter}, type {t}")
    return r

def new(rng, **kwargs):
    '''
    create a new instance of the dataclass defined above.
    there are a few ways of doing this: firstly, you can
    supply a kwarg "template", which should be an instance
    of the *same* dataclass defined above, and it'll just
    be copied in. to go with that, you can also add a
    kwarg "variability" as a float between 0 and 1,
    which controls the likelihood of mutation.
    
    otherwise, you can pass kwargs with values to "clamp"
    those parameters. for example, to force every instance
    to have one branch and one subunit in the antenna,
    add `**{"n_t": [3,4], ...}`. the code will set those
    and then loop over the remaining parameters and randomise
    '''
    population = np.zeros(constants.population_size, dtype=gt)
    if 'template' in kwargs:
        # if you gave the template using lists instead of
        # numpy arrays, for some reason the dataclass will
        # accept that and not bother converting them, which
        # will break the GA down the line. so check the types
        # of the parameters and convert them as necessary
        template = kwargs['template']
        if template.keys() != set(gt.names):
            print("function new in genetic_algorithm.py: template error")
            print("keys in template should match those in population:")
            print(f"template keys: {template.keys()}")
            print(f"population names: {gt.names}")
            raise KeyError
        else:
            for k in population.keys():
                for i in range(constants.population_size):
                    population[k][i] = np.array(template[k], dtype=gt[k])
                    if 'variability' in kwargs:
                        if rng.random() < kwargs['variability']:
                            mutate(population, i)
        return population
    
    for k, v in kwargs.items():
        if k in gt.names:
            for i in range(constants.population_size):
                population[i][k] = v
        else:
            print("function new in genetic_algorithm.py: kwarg warning")
            print(f"kwargs passed: {kwargs}")
            print(f"population names: {gt.names}")
    other_params = set(gt.names) - kwargs.keys()
    for i in range(constants.population_size):
        for k in other_params:
            shape = population[i][k].shape
            if len(shape) > 0:
                size = shape
            else:
                size = None
            population[i][k] = get_rand(rng, k, size=size)
        fix_matrices(rng, population[i])
    return population

def fitness(g, nu_e, cost, rc_nu_e):
    '''
    hmm.
    '''
    f = ((nu_e - rc_nu_e) - (cost * (g.n_b * np.sum(g.n_p))))
    return f if f >= 0.0 else 0.0

def selection(rng, population, fitnesses, cost):
    '''
    build up a 'mating pool' for recombination. i've implemented
    a few different strategies here - the highest (too high) selection
    pressure is to keep the n_survivors fittest individuals; after that
    there's ranked (where the probability of each individual being chosen
    is proportional to its fitness) or tournament selection, where we
    choose tourney_k members at random and choose the fittest of those.
    i found tournament selection didn't seem to work very well
    '''
    if np.max(fitnesses) == 0.0:
        # if the cost is too high or the algorithm can't find
        # any workable antennae, return an error to catch in main.py
        raise ValueError("No antennae with positive fitness values.")

    n_survivors = int(constants.fitness_cutoff * constants.population_size)
    strategy = constants.selection_strategy
    fidx = np.argsort(fitnesses)
    print(type(fitnesses), fidx.dtype)
    fsort = fitnesses[fidx]
    if constants.selection_strategy == 'fittest':
        survivors = [fidx[i] for i in range(n_survivors)]
    elif constants.selection_strategy == 'ranked':
        survivors = []
        ps = np.array([1.0 - np.exp(-f / np.max(fitnesses)) for f in fsort])
        pc = np.cumsum(ps / np.sum(ps))
        # stochastic universal sampling to pick pool
        i = 0
        c = 0
        r = rng.uniform(low=0.0, high=(1.0 / n_survivors))
        while c < n_survivors and i < constants.population_size:
            while r <= pc[i]:
                survivors.append(fidx[i])
                r += 1.0 / n_survivors
                c += 1
            i += 1
    else:
        raise ValueError("Invalid selection strategy")
    return survivors

def recombine(rng, vals, bb):
    '''
    perform intermediate recombination for a single array element,
    where the vals are those for the two parents and bb are the
    bounds [lb, ub] of the parameter we're recombining. return
    an appropriate recombined value for the child within [lb, ub]
    '''
    d = constants.d_recomb
    b = rng.uniform(-d, 1 + d)
    new = vals[0] * b + vals[1] * (1 - b)
    while new < bb[0] or new > bb[1]:
        b = rng.uniform(-d, 1 + d)
        new = vals[0] * b + vals[1] * (1 - b)
    if bb.dtype == it:
        new = np.round(new).astype(int)
    return new

def crossover(rng, parameter, parents, child):
    '''
    perform crossover for a given parameter element-by-element
    '''
    p_iters = [parents[i][parameter].flat for i in range(2)]
    c_iter = child[parameter].flat
    b = bounds[parameter]
    # all arrays are the same size in every row, so this is fine
    # need a more sophisticated nan checking function
    for i, parent_vals in enumerate(zip(*p_iters)):
        for pv in parent_vals:
            if np.isnan(pv):
                pv = get_rand(rng, parameter)
        c_iter[i] = recombine(rng, parent_vals, b)

def reproduction(rng, survivors, population):
    '''
    Given the survivors of a previous iteration, generate the necessary
    number of children and return the new population.
    Characteristics are taken randomly from each parent as much as possible.
    Generational strategy replaces the whole population, otherwise
    use a steady-state model where the survivors are carried forward
    '''
    if constants.reproduction_strategy == 'nads':
        n_carried = 0
    elif constants.reproduction_strategy == 'steady':
        n_carried = len(survivors)
    elif constants.reproduction_strategy == 'tournament':
        n_carried = len(survivors)
    else:
        raise ValueError("Invalid reproduction strategy")

    n_children = constants.population_size - n_carried
    for i in range(n_carried):
        population[i] = population[survivors[i]]

    for i in range(n_children):
        # pick two different parents from the survivors
        # these are indices from the population
        p_i = rng.choice(len(survivors), 2, replace=False)
        # the population index of the child
        child = population[i + n_carried]
        parents = [population[p_i[i]] for i in range(2)]
        indices = (*p_i, child)
        for k in gt.names:
            crossover(rng, k, parents, child)
        fix_matrices(rng, child)

def mutate(rng, current, bb):
    '''
    mutate a value with current value current and bounds bb [lb, ub].
    '''
    width = np.round(constants.mu_width *
            (bb[1] - bb[0])).astype(int)
    increment = ss.norm.rvs(loc=0, scale=width, random_state=rng)
    if bb[0].dtype == it:
        increment = increment.round().astype(int)
    new = current + increment
    if new < bb[0]:
        new = bb[0]
    if new > bb[1]:
        new = bb[1]
    return new

def mutation(rng, individual):
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
    for name in gt.names:
        bb = bounds[name]
        arr = individual[name]
        it = np.nditer(arr, flags=['multi_index'])
        with np.nditer(arr, op_flags=['readwrite']) as it:
            for elem in it:
                new = mutate(rng, elem, bb)
                elem[...] = new
    fix_matrices(rng, individual)

def evolve(rng, population, fitnesses, cost):
    '''
    apply the whole GA and return the next generation
    '''
    survivors = selection(rng, population, fitnesses, cost)
    reproduction(rng, survivors, population)
    n_mutations = 0
    for j in range(constants.population_size):
        p = rng.random()
        if p < constants.mu_rate:
            mutation(rng, population[j])
            n_mutations += 1

def assertions_bounds_and_nans(population):
    '''
    check that everything is as it should be
    add any other assertions we can make here
    '''
    for i in range(constants.population_size):
        individual = population[i]
        for name in gt.names:
            arr = individual[name]
            b = bounds[name]
            if len(arr.shape) > 1:
                # this is e or k
                for j, ntj in enumerate(individual['n_t']):
                    assert np.all(np.isnan(arr[j, ntj:])),\
f'''Matrix assertion failed (not enough nans):
{i}, {individual['n_t']}, {arr}
{j}, {ntj}, {arr[j]}
'''
                    assert not np.any(np.isnan(arr[j, :ntj])),\
f'''Matrix assertion failed (too many nans):
{i}, {individual['n_t']}, {arr}
{j}, {ntj}, {arr[j]}
'''
            for j, elem in enumerate(arr.flat):
                if not np.isnan(elem):
                    assert elem >= b[0] and elem <= b[1],\
            f"Bounds assertion failed: {name}, bounds = {b}, {elem}, {j}"

def assertions_kwarg(population, name, arr):
    for p in population:
        if len(arr.shape) > 1:
            # this is e or k
            for j, ntj in enumerate(p['n_t']):
                assert np.all(p[name][j, :ntj] == arr[j, :ntj]),\
f'''
kwarg matrix assertion failed.
{p}
{name}
{arr}
'''
        else:
            assert np.all(p[name] == arr),\
f'''
kwarg assertion failed.
{p}
{name}
{arr}
'''

def test_new_kwargs(n_tests):
    cost = 0.01
    rng = np.random.default_rng()
    for i in range(n_tests):
        print(f"kwarg assertions, test {i}:")
        guh = np.zeros(1, dtype=gt)
        for k in gt.names:
            arr = get_rand(rng, k, size=gt[k].shape)
            kwargs = {k: arr}
            print(f"parameter {k}, arr {arr}")
            population = new(rng, **kwargs)
            assertions_kwarg(population, k, arr)
            print("assertions passed")

def test_bounds_and_nans(n_gens):
    cost = 0.01
    rng = np.random.default_rng()
    print("Initialising population")
    population = new(rng)
    print("Running assertions")
    assertions_bounds_and_nans(population)
    print("Assertions passed!")
    print("Averaging")
    for i in range(n_gens):
        for name in gt.names:
            arr = population[name]
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            print(f"Parameter: {name}. Mean: {mean}. sigma: {std}")
        fitnesses = np.array([100.0 * rng.uniform()
                     for _ in range(constants.population_size)])
        print("Evolving")
        evolve(rng, population, fitnesses, cost)
        print("Running assertions")
        assertions_bounds_and_nans(population)
        print("Assertions passed!")

if __name__ == "__main__":
    n = 10
    try:
        test_bounds_and_nans(n)
    except AssertionError:
        print("BOUNDS AND NANS ASSERTION FAILED")
        raise
    try:
        test_new_kwargs(n)
    except AssertionError:
        print("NEW KWARGS ASSERTIONS FAILED")
        raise

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
        ('i_p',    ft, (constants.n_rc)),
        ('k_cs', ft, (constants.n_rc)),
        ('n_t',  it, (constants.n_rc)),
        ('k_t',    ft, (constants.n_rc, constants.n_t_max)),
        ('e_t',    ft, (constants.n_rc, constants.n_t_max)),
        ('output',    ft),
        ('redox',    ft, (constants.n_rc, 2)),
        ])
# these are bounds on each element for array variables
# they have to match the dict in make_arrays
# NB: [0.4, 4.0]eV ~= [3000.0, 300.0]nm wavelength light
bounds = {
        'dE0':  np.array([0.4, 4.0], dtype=ft),
        'i_p':  np.array([3.0, 8.0], dtype=ft),
        'k_cs': np.array([1.0E3, 1.0E12], dtype=ft),
        'n_t':  np.array([1, constants.n_t_max], dtype=it),
        'k_t':  np.array([1.0E3, 1.0E12], dtype=ft),
        'e_t':  np.array([-7.0, -1.0], dtype=ft),
        }

increments = {
        'dE0':  0.1,
        'i_p':  0.1,
        'k_cs': 1.0E6,
        'n_t':  1,
        'k_t':  1.0E6,
        'e_t':  0.1,
        }

def fix_matrices(genome, rng):
    '''
    k and e are a matrix (n_rc, n_t_max) for each individual,
    but the actual number of energies and rates for each row
    should be n_t[i]. when reproducing or mutating, the elements
    of n_t can change - set the rest to np.nan to make sure they
    don't get used by accident anywhere
    '''
    for kk, ntk in enumerate(genome['n_t']):
        krow = genome['k_t'][kk]
        erow = genome['e_t'][kk]
        for jj in range(ntk):
            # if n_t has mutated, it might've increased, in which
            # case there will be too many nans in k and e. fix this
            if np.isnan(krow[jj]):
                krow[jj] = get_rand(rng, 'k_t')
                erow[jj] = get_rand(rng, 'e_t')
        krow[ntk:] = np.nan
        erow[ntk:] = np.nan

def fix_bounds(genome):
    '''
    fix any values that have gone out-of-bounds back in bounds
    '''
    for name in gt.names:
        with np.nditer(genome[name], op_flags=['readwrite']) as it:
            for elem in it:
                if elem < bounds[name][0]:
                    new = bounds[name][0]
                elif elem > bounds[name][1]:
                    new = bounds[name][1]
                else:
                    new = elem
                elem[...] = new

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

def create_from_dict(params):
    '''
    create one instance of the numpy dtype defined above for the genome,
    from a dict of parameters. just makes it easier to test
    '''
    nn = np.zeros(1, dtype=gt)
    for key, val in params.items():
        if key not in set(gt.names):
            raise KeyError(
            f'''
            invalid key provided to ga.create_from_dict().
            valid keys are: {gt.names}, provided keys are {params.keys}
            ''')
        else:
            if key in ['k_t', 'e_t']:
                nn[0][key].fill(np.nan)
                for ii, nti in enumerate(params['n_t']):
                    nn[0][key][ii, :nti] = params[key][ii]
            else:
                nn[0][key] = np.array(params[key])
    return nn

def new(rng, **kwargs):
    '''
    create a new population of the dataclass defined above.
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
    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = constants.population_size
    population = np.zeros(size, dtype=gt)
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
            tt = create_from_dict(template)
            for ii in range(size):
                for name in gt.names:
                    population[ii][name] = tt[name]
                    if 'variability' in kwargs:
                        if rng.random() < kwargs['variability']:
                            mutate(population, i)
        return population
    
    for k, v in kwargs.items():
        if k in gt.names:
            for i in range(size):
                population[i][k] = v
        else:
            print("function new in genetic_algorithm.py: kwarg warning")
            print(f"kwargs passed: {kwargs}")
            print(f"population names: {gt.names}")
    other_params = set(gt.names) - kwargs.keys()
    for i in range(size):
        for k in other_params:
            shape = population[i][k].shape
            if len(shape) > 0:
                size = shape
            else:
                size = None
            population[i][k] = get_rand(rng, k, size=size)
        fix_matrices(population[i], rng)
    return population

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

    n_survivors = int(constants.fitness_cutoff * len(population))
    fidx = np.argsort(fitnesses)
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
        while c < n_survivors and i < len(population):
            while r <= pc[i]:
                survivors.append(fidx[i])
                r += 1.0 / n_survivors
                c += 1
            i += 1
    else:
        raise ValueError("Invalid selection strategy")
    return survivors

def per_rc_cross(rng, parents, children):
    '''
    different kind of crossover procedure: for each RC pick a
    parent at random and copy the parent's RC in this position
    to one child, the other's RC in this position to the other.
    This allows us to mix one photosystem at a time, which (hopefully)
    will allow well-optimised photosystems to stick around more?
    '''
    children = np.zeros(2, dtype=gt)
    for ii in range(constants.n_rc):
        pp = np.round(rng.uniform()).astype(int)
        for name in gt.names:
            children[0][name] = parents[pp][name]
            children[1][name] = parents[1 - pp][name]

def per_param_cross(rng, parents, children):
    '''
    different kind of crossover procedure: for each parameter, pick one
    parent: one child gets this parent's values for that parameter, the
    other child gets the other parent's parameters.
    This allows us to mix one photosystem at a time, which (hopefully)
    will allow well-optimised photosystems to stick around more?
    '''
    for name in gt.names:
        for ii in range(constants.n_rc):
            pp = np.round(rng.uniform()).astype(int)
            children[0][name] = parents[pp][name]
            children[1][name] = parents[1 - pp][name]

def intrec(rng, vals):
    ''' return a pair of new values based on parameter type '''
    new = np.zeros_like(vals)
    d = constants.d_recomb
    b = rng.uniform(-d, 1 + d)
    new = np.matmul(vals, [[b, 1 - b], [1 - b, b]])
    if vals[0].dtype == it:
        new = np.round(new).astype(int)
    return new

def intermediate_cross(rng, parents, children):
    '''
    perform crossover using intermediate recombination for all parameters
    '''
    for name in gt.names:
        p_iters = [parents[i][name].flat for i in range(2)]
        c_iters = [children[i][name].flat for i in range(2)]
        # all arrays are the same size in every row, so this is fine
        # need a more sophisticated nan checking function
        vv = zip(zip(*p_iters), zip(*c_iters))
        for parent_vals, child_vals in vv:
            # print(name, parent_vals, child_vals)
            for pv in parent_vals:
                if np.isnan(pv):
                    pv = get_rand(rng, name)
            child_vals = intrec(rng, parent_vals)

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

    n_children = len(population) - n_carried
    for i in range(n_carried):
        population[i] = population[survivors[i]]

    for ii in range(0, n_children, 2):
        children = population[ii + n_carried:ii + n_carried + 2]
        # pick two different parents from the survivors
        # these are indices from the population
        # p_i = rng.choice(len(survivors), 2, replace=False)
        p_i = rng.choice(survivors, 2)
        # the population index of the child
        parents = [population[p_i[i]] for i in range(2)]
        if rng.random() > constants.arith_inter_crossover_p:
            intermediate_cross(rng, parents, children)
        else:
            if rng.random() > constants.arith_cross_p:
                per_rc_cross(rng, parents, children)
            else:
                per_param_cross(rng, parents, children)
        for child in children:
            fix_bounds(child)
            fix_matrices(child, rng)

def gauss_mutate(rng, name, current):
    '''
    mutate a value of parameter `name` and current value `current`.
    uses a truncated Gaussian centred on the current value of the parameter
    '''
    bb = bounds[name]
    width = np.round(constants.mu_width *
            (bb[1] - bb[0])).astype(int)
    increment = ss.norm.rvs(loc=0, scale=width, random_state=rng)
    if bb[0].dtype == it:
        increment = increment.round().astype(int)
    return current + increment

def inc_mutate(rng, name, current):
    '''
    mutate one increment at a time, as defined by increments above
    '''
    sign = 1 if rng.uniform() > 0.5 else -1
    return current + sign * increments[name]

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
    if rng.uniform() > constants.mu_all_p:
        to_mutate = list(gt.names)
    else:
        to_mutate = [rng.choice(gt.names)]
    for name in to_mutate:
        arr = individual[name]
        with np.nditer(arr, op_flags=['readwrite']) as it:
            for elem in it:
                if rng.uniform() > constants.mu_type_p:
                    new = gauss_mutate(rng, name, elem)
                else:
                    new = inc_mutate(rng, name, elem)
                elem[...] = new
    fix_bounds(individual)
    fix_matrices(individual, rng)

def evolve(rng, population, fitnesses, cost):
    '''
    apply the whole GA and return the next generation
    '''
    survivors = selection(rng, population, fitnesses, cost)
    reproduction(rng, survivors, population)
    n_mutations = 0
    for gg in population:
        if rng.random() < constants.mu_rate:
            mutation(rng, gg)
            n_mutations += 1

def assertions_bounds_and_nans(population):
    '''
    check that everything is as it should be
    add any other assertions we can make here
    '''
    for gg in population:
        for name in gt.names:
            arr = gg[name]
            b = bounds[name]
            if len(arr.shape) > 1:
                # this is e or k
                for ii, nti in enumerate(gg['n_t']):
                    assert np.all(np.isnan(arr[ii, nti:])),\
f'''Matrix assertion failed (not enough nans):
{gg}, {gg['n_t']}, {arr}
{ii}, {nti}, {arr[ii]}
'''
                    assert not np.any(np.isnan(arr[ii, :nti])),\
f'''Matrix assertion failed (too many nans):
{gg['n_t']}, {arr}
{ii}, {nti}, {arr[ii]}
'''
            for kk, elem in enumerate(arr.flat):
                if not np.isnan(elem):
                    assert elem >= b[0] and elem <= b[1],\
            f"Bounds assertion failed: {name}, bounds = {b}, {elem}, {kk}"

def assertions_kwarg(population, name, arr):
    for gg in population:
        if len(arr.shape) > 1:
            # this is e or k
            for ii, nti in enumerate(gg['n_t']):
                assert np.all(gg[name][ii, :nti] == arr[ii, :nti]),\
f'''
kwarg matrix assertion failed.
{gg}
{name}
{arr}
'''
        else:
            assert np.all(gg[name] == arr),\
f'''
kwarg assertion failed.
{gg}
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

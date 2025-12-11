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
# these are bounds on each element for array variables
# they have to match the dict in make_arrays
bounds = {
        'dE0': np.array([300, 3000], dtype=it),
        'k_cs': np.array([1.0, 1.0E12], dtype=ft),
        'n_t': np.array([1, 10], dtype=it),
        'e': np.array([300, 3000], dtype=it),
        'k': np.array([1.0, 1.0E12], dtype=ft),
        }

def make_arrays(n_rc, n_t_max):
    '''
    '''
    dt = [('dE0', ft, (n_rc)),
          ('k_cs', ft, (n_rc)),
          ('n_t', it, (n_rc)),
          ('k', ft, (n_rc, n_t_max)),
          ('e', ft, (n_rc, n_t_max)),
          ]
    population = np.zeros(constants.population_size, dtype=dt)
    return population

def get_index(population, index):
    return {k: v[index] for k, v in population.items()}

def set_index(population, index, d):
    for k, v in d.items():
        population[k][index] = v

def fix_matrices(population, index):
    '''
    k and e are a matrix (n_rc, n_t_max) for each individual,
    but actually only the first n_t[i] elements of row i
    are meaningful. set the rest to np.nan to make sure they
    don't get used by accident anywhere
    '''
    nt = population[index]['n_t']
    for i, nti in enumerate(nt):
        population[index]['k'][i, nti:] = np.nan
        population[index]['e'][i, nti:] = np.nan

def get_rand(rng, parameter, size=None):
    '''
    get a single random value based on parameter type
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

def new(rng, n_rc, n_t_max, **kwargs):
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
    add `**{"n_b": 1, "n_s": 1}`. the code will set those
    and then loop over the remaining parameters and randomise
    '''
    population = make_arrays(n_rc, n_t_max)
    if 'template' in kwargs:
        # if you gave the template using lists instead of
        # numpy arrays, for some reason the dataclass will
        # accept that and not bother converting them, which
        # will break the GA down the line. so check the types
        # of the parameters and convert them as necessary
        template = kwargs['template']
        if template.keys()) != population.keys()):
            print("function new in genetic_algorithm.py: template error")
            print("keys in template should match those in population:")
            print(f"template keys: {template.keys()}")
            print(f"population keys: {population.keys()}")
            raise KeyError
        else:
            for k in population.keys():
                for i in range(constants.population_size):
                    population[k][i] = template[k]
                    if 'variability' in kwargs:
                        if rng.random() < kwargs['variability']:
                            mutate(population, i)
        return population
    
    for k, v in kwargs.items():
        if k in population.keys():
            for i in range(constants.population_size):
                population[k][i] = v
        else:
            print("function new in genetic_algorithm.py: kwarg warning")
            print(f"kwargs passed: {kwargs}")
            print(f"population keys: {population.keys()}")
    other_params = population.keys() - kwargs.keys()
    for k in other_params:
        for i in range(constants.population_size):
            shape = population[k][i].shape
            if len(shape) > 0:
                size = shape
            else:
                size = None
            population[k][i] = get_rand(rng, k, size=size)
        fix_matrices(population, i)
    return population

def fitness(g, nu_e, cost, rc_nu_e):
    '''
    hmm.
    '''
    # return (nu_e - (cost * (g.n_b * np.sum(g.n_p))))
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

def recombine(rng, vals, parameter):
    '''
    perform intermediate recombination of parameters
    where possible, and a straight choice in the case of
    pigment type, for a single array element
    '''
    d = constants.d_recomb
    bounds = bounds[parameter]
    if parameter == 'pigment' or parameter == 'rc': # binary choice
        new = rng.choice(vals)
    else:
        b = rng.uniform(-d, 1 + d)
        new = vals[0] * b + vals[1] * (1 - b)
        '''
        the next line requires some explanation. basically, if
        a custom normalisation function is defined for some
        parameter, then that function can (not always, but in
        general) move elements out of the bounds given, even
        if they were within the bounds when they were chosen here.
        if that happens, the while check below will always fail.
        therefore, we only do this check if there's no norm function
        defined, and instead we do a check in crossover to make sure
        that the elements of the array are all in bounds.
        '''
        while new < bounds[0] or new > bounds[1]:
            b = rng.uniform(-d, 1 + d)
            new = vals[0] * b + vals[1] * (1 - b)
        if bounds.dtype == it:
            new = np.round(new).astype(int)
    return new

def crossover(rng, child, parents, parameter):
    '''
    '''
    parent_vals = [p[parameter] for p in parents]
    dt = bounds[parameter].dtype
    if genome_parameters[parameter]['array']:
        '''
        if this is an array parameter, the size of the array on
        the child might be larger or smaller than one or both parents,
        so we generate two arrays of the correct size to recombine from.
        see fill_arrays for more explanation.
        note that the current size of the array might be wrong if
        RC type or n_s has changed, so check what size it *should* be
        '''
        s = genome_parameters[parameter]['size'](child)
        vals = fill_arrays(rng, parent_vals, s, parameter)
        new = np.zeros(s, dtype=dt)
        all_in_bounds = False
        tries = 1
        while not all_in_bounds:
            b = genome_parameters[parameter]['bounds']
            for i in range(s):
                '''
                we need to loop here since each value of b in the recombine
                function should be different; otherwise we have to find
                a value of b where every element of new is within bounds,
                which significantly reduces the possible variation
                '''
                v = [vals[j][i] for j in range(2)]
                new[i] = recombine(rng, v, parameter)
            if genome_parameters[parameter]['norm'] is not None:
                # normalise based on the given function
                new = genome_parameters[parameter]['norm'](new)
            '''
            now we check that all elements of the array are in bounds,
            as described above. this maybe isn't the most efficient
            way of doing this, but i can't think of a better one
            NB: this is not working properly, particularly for the
            affinities - the division tends to throw the array elements
            out of bounds. maybe just snap them back to [lb, ub]?
            '''
            all_in_bounds = check_bounds(new, parameter)
            tries += 1
            if tries > 50:
                raise RuntimeError(f"Recombination is broken for {parameter}: bounds {b}, vals {vals}, new {new}")
    else:
        vals = parent_vals
        new = recombine(rng, vals, parameter)
    setattr(child, parameter, new)

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
        set_index(population, i, get_index(population, survivors[i]))

    for i in range(n_children):
        child = get_index(population, i + n_carried)
        # pick two different parents from the survivors
        # these are indices from the population
        p_i = rng.choice(len(survivors), 2, replace=False)
        parents = [get_index(population, survivors[p_i[i]] for i in range(2))]
        for k in child.keys():
            crossover(rng, child, parents, k)
    return population

def mutate(rng, genome, parameter, index=None):
    '''
    mutate a parameter with a given name.
    getattr/setattr can be used to get the right dataclass fields.
    if the parameter we're mutating is an array, index into
    getattr to get the right element. we also need to check type,
    since if we're mutating n_s we need to have integer extents and
    indices into the resulting arrays.
    '''
    if index is not None:
        current = getattr(genome, parameter)[index]
    else:
        current = getattr(genome, parameter)
    bounds = genome_parameters[parameter]['bounds']
    if parameter == 'pigment' or parameter == 'rc':
        new = rng.choice(bounds)
    else:
        width = np.round(constants.mu_width *
                (bounds[1] - bounds[0])).astype(int)
        increment = ss.norm.rvs(loc=0, scale=width, random_state=rng)
        if isinstance(current, (int, np.integer)):
            increment = increment.round().astype(int)
        new = current + increment
        if new < bounds[0]:
            new = bounds[0]
        if new > bounds[1]:
            new = bounds[1]
    if index is not None:
        getattr(genome, parameter)[index] = new
    else:
        setattr(genome, parameter, new)

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
    for name, param in genome_parameters.items():
        if param['mutable']:
            if param['array']:
                # check the size first - has it changed?
                current = getattr(individual, name)
                current_size = len(current)
                new_size = param['size'](individual)
                if current_size != new_size:
                    current.resize(new_size, refcheck=False)
                    if new_size > current_size:
                        # resize pads with zeros - fill with copies
                        # of the last element instead
                        for i in range(new_size - current_size):
                            current[-(i + 1)] = current[current_size - 1]
                # the way i've had it set up is to only mutate
                # one element of each array; this is just one
                # choice. maybe chat to chris about it, but for now
                # copy the previous behaviour
                setattr(individual, name, current)
                index = rng.integers(0, new_size)
                mutate(rng, individual, name, index)
            else:
                mutate(rng, individual, name, None)
            if param['norm'] is not None:
                setattr(individual, name,
                        param['norm'](getattr(individual, name)))
    return individual

def evolve(rng, population, fitnesses, cost):
    '''
    apply the whole GA and return the next generation
    '''
    survivors = selection(rng, population, fitnesses, cost)
    population = reproduction(rng, survivors, population)
    n_mutations = 0
    for j in range(constants.population_size):
        p = rng.random()
        if p < constants.mu_rate:
            population[j] = mutation(rng, population[j])
            n_mutations += 1
    return population

def test_parameters(individual):
    '''
    check that everything is as it should be
    add any other assertions we can make here
    '''
    for name, param in genome_parameters.items():
        val = getattr(individual, name)
        b = param['bounds']
        if param['array']:
            assert param['size'](individual) == len(val),\
            f"{name}, {len(val)}, {param['size'](individual)}"
            if param['norm'] is not None:
                assert np.all(param['norm'](val) - val < 1e-6),\
                    f"norm failed for {name}: {val}, {param['norm'](val)}"
            else:
                for i, vi in enumerate(val):
                    if isinstance(vi, (str, np.str_)):
                        assert vi in b, f"{vi} not in {b}"
                    else:
                        assert vi >= b[0] and vi <= b[1],\
                        f"bounds:{name}, {val}, {i}, {vi}, {b}" 
        else:
            if isinstance(val, (str, np.str_)):
                assert val in b, f"{val} not in {b}"
            else:
                assert val >= b[0] and val <= b[1],\
                f"bounds: {name}, {val}, {b}" 

if __name__ == "__main__":
    '''
    need to do this testing stuff properly. calculate the fitnesses
    properly, do the algorithm properly, run it for 20 generations, do
    the stats
    '''
    n = constants.population_size
    test_gens = 2
    cost = 0.02
    rng = np.random.default_rng()
    fitnesses  = np.array([100 * rng.random()
        for _ in range(n)])

    template = Genome("ox", 4, 3, [50, 60, 40],
            [0, 0, 0],
            ['chl_a', 'chl_b', 'pc'],
            [1.2, 0.8, 1.0],
            [0.5, 1.0])

    conditions = [{},
            {'n_b': 1, 'n_s': 1},
            {'template': template, 'variability': 0.5},
    ]
    names = ["random", "proto", "template"]
    for c, name in zip(conditions, names):
        pop = [new(rng, **c) for _ in range(n)]

        for i, genome in enumerate(pop):
            try:
                test_parameters(genome)
            except AssertionError:
                print("Assertion failed in check_parameters.")
                print(f"Genome {i}: {genome}")
                raise
        print(f"initial {name} population passed assertions.")
        for i in range(test_gens):
            old_pop = [copy(g) for g in pop]
            pop = evolve(rng, pop, fitnesses, cost)
            for j, genome in enumerate(pop):
                try:
                    test_parameters(genome)
                except AssertionError:
                    print("Assertion failed in check_parameters.")
                    print(f"Genome {j}: {genome}")
                    print(f"Genome {j} before mutation: {old_pop[j]}")
                    raise
            print(f"{name} population passed assertions at gen {i}.")


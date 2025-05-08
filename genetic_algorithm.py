# -*- coding: utf-8 -*-
"""
22/11/2023
@author: callum

GA version 2, 7/5/25 - dataclass itself and various other things are
just generated as needed from the dict genome_parameters.

NB: this currently depends on the fact that rc and n_s are defined
in the dict genome_parameters before the things that depend on them;
it doesn't actually check when generating a new genome (i.e. the
"affects" parameter in there currently does nothing). 
There's probably a way to do this more cleverly, 
but I haven't figured it out yet. Unsure it's necessary.
"""
import dataclasses
import numpy as np
import scipy.stats as ss
import constants
import rc as rcm

genome_parameters = {
    'rc': {
        'type'   : np.str_,
        'array'  : False,
        'default' : '',
        'affects' : ['rho', 'aff'],
        'bounds' : ['ox', 'frl', 'anox', 'exo'],
        'norm'   : None
    },
    'n_b': {
        'type'   : np.int64,
        'default': 0,
        'array'  : False,
        'bounds' : [1, 12],
        'norm'   : None,
    },
    'n_s': {
        'type'   : np.int64,
        'default': 0,
        'array'  : False,
        'affects': ['n_p', 'shift', 'pigment'],
        'bounds' : [1, 10],
        'norm'   : None,
    },
    'n_p': {
        'type'    : np.int64,
        'array'   : True,
        'size'    : lambda g: getattr(g, "n_s"),
        'depends' : 'n_s',
        'bounds'  : [1, 100],
        'norm'    : None,
    },
    'shift': {
        'type'    : np.int64,
        'array'   : True,
        'size'    : lambda g: getattr(g, "n_s"),
        'bounds'  : [-1, 1],
        'norm'    : None,
    },
    'pigment': {
        'type'    : 'U10',
        'array'   : True,
        'size'    : lambda g: getattr(g, "n_s"),
        'bounds'  : ['pe', 'pc', 'apc', 'chl_b', 'chl_a', 'chl_d', 'chl_f', 'bchl_a', 'bchl_b'],
        'norm'    : None,
    },
    'rho': {
        'type'    : np.float64,
        'array'   : True,
        'size'    : lambda g: rcm.n_rc[getattr(g, 'rc')] + 1,
        # note that this is a bit fake - upper bound must just be
        # >= the largest possible sum
        'bounds'  : [0.1, 5.0],
        'norm'    : lambda p: p * (len(p) / np.sum(p)),
    },
    'aff': {
        'type'    : np.float64,
        'array'   : True,
        'size'    : lambda g: rcm.n_rc[getattr(g, 'rc')],
        'bounds'  : [0.1, 10.0],
        'norm'    : lambda p: p / p[-1],
    },
}

# construct a new dataclass definition from the dict given above
# this (hopefully) makes it easier to see what is in the genome
# as well as to change it where necessary.
fields = []
for key in genome_parameters:
    tt = genome_parameters[key]["type"]
    if genome_parameters[key]["array"]:
        fields.append((key, np.ndarray,
            dataclasses.field(default_factory=lambda:np.empty([], dtype=tt))))
    else:
        fields.append((key, tt, dataclasses.field(default=genome_parameters[key]["default"])))
Genome = dataclasses.make_dataclass('Genome', fields)

def get_type(parameter):
    ''' get parameter type to declare numpy array correctly '''
    b = genome_parameters[parameter]['bounds']
    if (isinstance(b[0], (int, np.integer))):
        dt = np.int32
    elif (isinstance(b[0], (float, np.float64))):
        dt = np.float64
    elif (isinstance(b[0], (str, np.str_))):
        dt = 'U10'
    else:
        raise TypeError("get_type cannot determine parameter type.")
    return dt

def get_rand(rng, parameter):
    '''
    get a single random value based on the type
    of the given bounds of the parameter. note that
    even though we give a variable type in the dict,
    it's not used here, because the dtype of pigment is
    actually U10 for numpy reasons and the type check
    will fail there. so just make sure that the
    bounds are the correct type; they should be anyway.
    note - numpy must be doing some internal conversion
    somewhere but i'm not sure how it works exactly. do
    not look the gift horse in the mouth.
    '''
    b = genome_parameters[parameter]['bounds']
    if isinstance(b[0], (int, np.integer)):
        r = rng.integers(*b, endpoint=True)
    elif isinstance(b[0], (float, np.float64)):
        r = rng.uniform(*b)
    elif isinstance(b[0], (str, np.str_)):
        r = rng.choice(b)
    else:
        t = genome_parameters[parameter]['type']
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
    add `**{"n_b": 1, "n_s": 1}`. the code will set those
    and then loop over the remaining parameters and randomise
    '''
    g = Genome()
    if 'template' in kwargs:
        # if you gave the template using lists instead of
        # numpy arrays, for some reason the dataclass will
        # accept that and not bother converting them, which
        # will break the GA down the line. so check the types
        # of the parameters and convert them as necessary
        for param in genome_parameters:
            p = getattr(template, param)
            tt = genome_parameters[param]['type']
            if tt == 'U10': # pigment list
                setattr(g, param, np.array(p, dtype='U10'))
            else:
                setattr(g, param, tt(p))
        if 'variability' in kwargs:
            if rng.random() < kwargs['variability']:
                mutation(rng, g)
        return g
    
    for k, v in kwargs.items():
        setattr(g, k, v)
    # now set the other things randomly. the reason for the dict
    # comprehension being like this rather than just a set
    # subtraction is that we want to keep ordering - see the
    # module docstring comment about parameter dependency
    other_params = {k: genome_parameters[k] for k in
                    genome_parameters.keys() if k not in kwargs.keys()}
    for name, param in other_params.items():
        if param['array']:
            p = np.zeros(param['size'](g), dtype=param['type'])
            for i in range(param['size'](g)):
                p[i] = get_rand(rng, name)
            if param['norm'] is not None:
                p = param['norm'](p)
            setattr(g, name, p)
        else:
            setattr(g, name, get_rand(rng, name))
    return g

def copy(g):
    '''
    return a copy of g
    '''
    h = Genome()
    for key in genome_parameters.keys():
        setattr(h, key, getattr(g, key))
    return h

def fitness(g, nu_e, cost):
    '''
    hmm.
    '''
    return (nu_e - cost * ((g.n_b * np.sum(g.n_p)) 
                           + np.sum(rcm.params[g.rc]['n_p'])))

def fill_arrays(rng, parent_values, res_length, parameter):
    '''
    the length of a subunit/RC dependent array a child has might be
    smaller or larger than one or both of its parents. we make sure
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

def tournament(rng, population, fitnesses, k, cost):
    '''
    tournament selection with tournament size k.
    return the most fit individual after k random samples
    '''
    fit_max = 0.0
    for i in range(k):
        ind = rng.integers(constants.population_size)
        winner = ind # if they all have 0 fitness, just take the first
        p = population[ind]
        if (fitnesses[ind] > fit_max):
            fit_max = fitnesses[ind]
            winner = ind
    return population[winner]

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
    psort = [population[i] for i in fidx]
    if constants.selection_strategy == 'fittest':
        survivors = [psort[i] for i in range(n_survivors)]
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
                survivors.append(psort[i])
                r += 1.0 / n_survivors
                c += 1
            i += 1
    elif constants.selection_strategy == 'tournament':
        survivors = []
        for i in range(n_survivors):
            survivors.append(tournament(rng, population, fitnesses,
                                        constants.tourney_k, cost))
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
    bounds = genome_parameters[parameter]['bounds']
    if parameter == 'pigment' or parameter == 'rc': # binary choice
        new = rng.choice(vals)
    else:
        b = rng.uniform(-d, 1 + d)
        new = vals[0] * b + vals[1] * (1 - b)
        tries = 1
        # NB: this is a bit of a stopgap, and i'm not sure if it's
        # acceptable to do in general or not. basically my current
        # thinking is that if there's a norm defined, then applying
        # that should always produce an acceptable result, so don't
        # worry about the bounds here. but idk
        if genome_parameters[parameter]['norm'] is None:
            while new < bounds[0] or new > bounds[1]:
                b = rng.uniform(-d, 1 + d)
                new = vals[0] * b + vals[1] * (1 - b)
                tries += 1
                if tries > 100:
                    print(f"{parameter}, {tries}, {new}, {vals}, {b}, {bounds}")
        if isinstance(bounds[0], (int, np.integer)):
            new = np.round(new).astype(int)
    return new

def crossover(rng, child, parents, parameter):
    '''
    '''
    parent_vals = [getattr(p, parameter) for p in parents]
    dt = get_type(parameter)
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
    else:
        s = 1
        vals = parent_vals
    if s == 1:
        new = recombine(rng, vals, parameter)
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
            new[i] = recombine(rng, v, parameter)
    if genome_parameters[parameter]['norm'] is not None: # normalise
        new = genome_parameters[parameter]['norm'](new)
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
        population[i] = survivors[i]

    for i in range(n_children):
        child = population[i + n_carried]
        # pick two different parents from the survivors
        p_i = rng.choice(len(survivors), 2, replace=False)
        parents = [survivors[p_i[i]] for i in range(2)]
        for parameter in genome_parameters:
            crossover(rng, child, parents, parameter)
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
        scale = (bounds[1] - bounds[0]) * constants.mu_width
        b = (bounds - current) / (scale)
        new = ss.truncnorm.rvs(b[0], b[1], loc=current,
                               scale=scale, random_state=rng)
        if isinstance(current, (int, np.integer)):
            new = new.round().astype(int)
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
    NB: this doesn't work for rho and aff in particular,
    since the normalisation can pull individual elements
    out of the bounds. is there a way to fix that? need to
    think.
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
    # test random generation
    random_pop = []
    for i in range(n):
        random_pop.append(new(rng))

    for i, genome in enumerate(random_pop):
        try:
            test_parameters(genome)
        except AssertionError:
            print("Assertion failed in check_parameters.")
            print(f"Genome {i}: {genome}")
            raise
    print("initial random population passed assertions.")
    for i in range(test_gens):
        old_pop = [copy(g) for g in random_pop]
        random_pop = evolve(rng, random_pop, fitnesses, cost)
        for j, genome in enumerate(random_pop):
            try:
                test_parameters(genome)
            except AssertionError:
                print("Assertion failed in check_parameters.")
                print(f"Genome {j}: {genome}")
                print(f"Genome {j} before mutation: {old_pop[j]}")
                raise
        print(f"random population passed assertions at gen {i}.")

    # test proto generation
    random_pop = [new(rng, **{'n_b': 1, 'n_s': 1})
            for _ in range(n)]

    for i, genome in enumerate(random_pop):
        try:
            test_parameters(genome)
        except AssertionError:
            print("Assertion failed in check_parameters.")
            print(f"Genome {i}: {genome}")
            raise
    print("initial proto population passed assertions.")
    for i in range(test_gens):
        old_pop = [copy(g) for g in random_pop]
        random_pop = evolve(rng, random_pop, fitnesses, cost)
        for j, genome in enumerate(random_pop):
            try:
                test_parameters(genome)
            except AssertionError:
                print("Assertion failed in check_parameters.")
                print(f"Genome {j}: {genome}")
                print(f"Genome {j} before mutation: {old_pop[j]}")
                raise
        print(f"proto population passed assertions at gen {i}.")

    # test template generation
    template = Genome("ox", 4, 3, [50, 60, 40],
            [0, 0, 0],
            ['chl_a', 'chl_b', 'pc'],
            [1.2, 0.8, 1.0],
            [0.5, 1.0])

    random_pop = [new(rng, **{'template': template, 'variability': 0.5})
            for _ in range(n)]

    for i, genome in enumerate(random_pop):
        try:
            test_parameters(genome)
        except AssertionError:
            print("Assertion failed in check_parameters.")
            print(f"Genome {i}: {genome}")
            raise
    print("initial template population passed assertions.")
    for i in range(test_gens):
        old_pop = [copy(g) for g in random_pop]
        random_pop = evolve(rng, random_pop, fitnesses, cost)
        for j, genome in enumerate(random_pop):
            try:
                test_parameters(genome)
            except AssertionError:
                print("Assertion failed in check_parameters.")
                print(f"Genome {j}: {genome}")
                print(f"Genome {j} before mutation: {old_pop[j]}")
                raise
        print(f"template population passed assertions at gen {i}.")


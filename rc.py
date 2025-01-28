# -*- coding: utf-8 -*-
"""
27/01/2025
@author: callum

note: this is maybe a bit of a hack, but by matching
the names of the functions to the strings listed in
constants.bounds["rc"], we can do

`pigments, mat, lin, cyc = getattr(rc, p.rc)()`
and pull them straight out. then, to add a completely
new kind of RC supersystem, you just need to define the
pigments and processes for it here, name it, and add the
function name to the list of strings.

i thought this would be the best way to do this since
the processes and rates for each type don't change,
and the types of photosynthesis will stay constant, so it
didn't really make sense to write something complicated to
generate arbitrary supersystems or anything.

i think there is a way to do it programatically, by checking
`diff[2i, 2i+1]` or `diff[2i, 2i + 1, 2i + 2, 2i + 3]`; sets of
these should be unique to different processes. e.g.: ::

for i in indices:
    initial = indices[i]
    for j in indices:
        final = indices[j]
        diff = final - initial
            for k in range(n_rc):
                if diff[2*k] == 1 and diff[2*k + 1] == 0:
                    mat[i][j] = k_trap
                if diff[2*k] == -1 and diff[2*k + 1] == 0:
                    mat[i][j] = k_trap * np.exp(-1.0 * energy_gap)

and so on. the only difficulty here is that for anoxygenic,
(-1, 0) corresponds to both de-trapping and cyclic e- flow?
you can build the set of (rc) indices that correspond to
linear and cyclic in the same way: ::

lin_index = 2 * n_rc - 1
if n_rc == 1:
    cyc_index = 0
else:
    cyc_index = [2 * i for i in range(1, n_rc)]
...
if np.any(initial[cyc_index]) == 1:
    cyc.append(i)
if initial[lin_index] == 1:
    lin.append(i)

or something like that, then convert them to matrix indices as you go.
but i don't think we need that complexity, so i haven't bothered
"""
import numpy as np
import itertools

# these were in constants.py but they're only needed here
k_trap = 1.0 / 10.0E-12
k_ox = 1.0 / 1.0E-3
k_lin = 1.0 / 10.0E-3
k_cyc = 1.0 / 10.0E-3
k_red = 1.0 / 10.0E-3

def get_rc_indices(n_rc):
    '''
    generate the set of possible combined RC states
    along with an index for them, for use below.

    parameters
    ----------
    n_rc: how many reaction centres in your supersystem

    outputs
    -------
    indices: dict of index: state pairs
    '''
    one_rc = [[0, 0], [1, 0], [0, 1]]
    n_states = len(one_rc)**n_rc
    n_states_per_rc = len(one_rc[0]) * n_rc
    # itertools.product multiplies the list one_rc with itself
    # n_rc times, to get a total state of all the RCs made up of the
    # one_rc states above. don't ask. nicked it off stackexchange.
    rc_states = np.array(list(map(list,
                itertools.product(one_rc,
                repeat=n_rc)))).reshape(n_states, n_states_per_rc)
    # made slightly more complicated by the fact that numpy arrays are not
    # hashable and therefore can't be dict keys, but we want a set of
    # indices to refer to. so make them tuples here and return
    return {i: rc_states[i] for i in range(n_states)}

def ox():
    '''
    return list of pigments and dict of processes
    for oxygenic photosynthesis.

    parameters
    ----------

    outputs
    -------
    pigments: list of pigments the photosystems are composed of
    processes: dict of population changes and rates for each process
    '''
    pigments = ["ps_ox", "ps_r"] # these should be in pigments.json
    energy_gap = 17.0 # energy drop in terms of k_B T
    processes = { # dict of change in population and rate
        (1, 0, 0, 0): k_trap,
        (0, 0, 1, 0): k_trap,
        (-1, 0, 0, 0): k_trap * np.exp(-1.0 * energy_gap),
        (0, 0, -1, 0): k_trap * np.exp(-1.0 * energy_gap),
        (-1, 1, 0, 0): k_ox,
        (0, -1, -1, 1): k_lin,
        (0, 0, -1, 0): k_cyc,
        (0, 0, 0, -1): k_red
    }
    indices = get_rc_indices(len(pigments))
    mat = np.zeros((len(indices), len(indices)), dtype=np.float64)

    lin = []
    cyc = []
    for i in indices:
        initial = indices[i]
        # we need the indices that correspond to ps^r_T to get \nu(cyc)
        if initial[2] == 1:
            cyc.append(i)
        # and the indices that correspond to ps^r_R to get \nu(CH_2O)
        if initial[3] == 1:
            lin.append(i)
        for j in indices:
            final = indices[j] # is this the right way round? check
            diff = tuple(final - initial)
            if diff in processes:
                mat[i][j] = processes[diff]
    return pigments, mat, lin, cyc

def frl():
    '''
    return list of pigments and dict of processes
    for far-red adapted oxygenic photosynthesis.

    parameters
    ----------

    outputs
    -------
    pigments: list of pigments the photosystems are composed of
    processes: dict of population changes and rates for each process
    '''
    pigments = ["ps_ox_frl", "ps_r_frl"] # these should be in pigments.json
    energy_gap = 10.0 # energy drop in terms of k_B T
    processes = { # dict of change in population and rate
        (1, 0, 0, 0): k_trap,
        (0, 0, 1, 0): k_trap,
        (-1, 0, 0, 0): k_trap * np.exp(-1.0 * energy_gap),
        (0, 0, -1, 0): k_trap * np.exp(-1.0 * energy_gap),
        (-1, 1, 0, 0): k_ox,
        (0, -1, -1, 1): k_lin,
        (0, 0, -1, 0): k_cyc,
        (0, 0, 0, -1): k_red
    }    
    indices = get_rc_indices(len(pigments))
    mat = np.zeros((len(indices), len(indices)), dtype=np.float64)

    lin = []
    cyc = []
    for i in indices:
        initial = indices[i]
        if initial[2] == 1:
            cyc.append(i)
        if initial[3] == 1:
            lin.append(i)
        for j in indices:
            final = indices[j] # is this the right way round? check
            diff = tuple(final - initial)
            if diff in processes:
                mat[i][j] = processes[diff]
    return pigments, mat, lin, cyc

def anox():
    '''
    return list of pigments and dict of processes
    for anoxygenic photosynthesis.

    parameters
    ----------

    outputs
    -------
    pigments: list of pigments the photosystems are composed of
    processes: dict of population changes and rates for each process
    '''
    pigments = ["ps_anox"] # these should be in pigments.json
    energy_gap = 14.0 # energy drop in terms of k_B T
    processes = { # dict of change in population and rate
        (1, 0): k_trap,
        (-1, 0): k_trap * np.exp(-1.0 * energy_gap),
        (-1, 1): k_ox, # or k_lin? i think they are equal
        (-1, 0): k_cyc,
        (0, -1): k_red
    }
    indices = get_rc_indices(len(pigments))
    mat = np.zeros((len(indices), len(indices)), dtype=np.float64)

    lin = []
    cyc = []
    for i in indices:
        initial = indices[i]
        if initial[0] == 1:
            cyc.append(i)
        if initial[1] == 1:
            lin.append(i)
        for j in indices:
            final = indices[j] # is this the right way round? check
            diff = tuple(final - initial)
            if diff in processes:
                mat[i][j] = processes[diff]
    return pigments, mat, lin, cyc

def exo():
    '''
    return list of pigments and dict of processes
    for "exotic" photosynthesis (hypothetical, near-IR
    three-photosystem photosynthesis).

    parameters
    ----------

    outputs
    -------
    pigments: list of pigments the photosystems are composed of
    processes: dict of population changes and rates for each process
    '''
    pigments = ["ps_ox_exo", "ps_i_exo", "ps_r_exo"] # these should be in pigments.json
    energy_gap = 10.0 # energy drop in terms of k_B T
    '''
    below assumes that only the first photosystem does substrate
    oxidation, that linear electron flow proceeds in two steps
    1->2, then 2->3, that photosystems 2 and 3 can both do cyclic
    electron flow, and that reduction only happens at photosystem 3
    '''
    processes = { # dict of change in population and rate
        (1, 0, 0, 0, 0, 0): k_trap,
        (0, 0, 1, 0, 0, 0): k_trap,
        (0, 0, 0, 0, 1, 0): k_trap,
        (-1, 0, 0, 0, 0, 0): k_trap * np.exp(-1.0 * energy_gap),
        (0, 0, -1, 0, 0, 0): k_trap * np.exp(-1.0 * energy_gap),
        (0, 0, 0, 0, -1, 0): k_trap * np.exp(-1.0 * energy_gap),
        (-1, 1, 0, 0, 0, 0): k_ox,
        (0, -1, -1, 1, 0, 0): k_lin,
        (0, 0, 0, -1, -1, 1): k_lin,
        (0, 0, -1, 0, 0, 0): k_cyc,
        (0, 0, 0, 0, -1, 0): k_cyc,
        (0, 0, 0, 0, 0, -1): k_red
    }
    indices = get_rc_indices(len(pigments))
    mat = np.zeros((len(indices), len(indices)), dtype=np.float64)

    lin = []
    cyc = []
    for i in indices:
        initial = indices[i]
        if initial[2] == 1 or initial[4] == 1:
            cyc.append(i)
        if initial[5] == 1:
            lin.append(i)
        for j in indices:
            final = indices[j] # is this the right way round? check
            diff = tuple(final - initial)
            if diff in processes:
                mat[i][j] = processes[diff]
    return pigments, mat, lin, cyc


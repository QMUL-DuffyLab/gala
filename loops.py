# -*- coding: utf-8 -*-
"""
10/02/2026
@author: callum
"""
import os
import itertools
import pickle
import zipfile
import numpy as np
import pandas as pd
import constants
import solvers
import stats
import light
import utils
import genetic_algorithm as ga

'''
we want to first loop over the absorption peak dE0,
ionisation potentials i_p, number of traps n_t and trap energies e_t
to find good solutions; then once we've found the best-optimised
configurations we optimise the trap rates separately, i think

this whole method of looping, only using one instance of ga.gt,
only keeping ones that have above a certain cutoff output and then
pickling them in batches is horrible and i don't like it. don't judge me pls.
it's just that the number of configurations in this product gets
absolutely enormous very quickly and the vast majority of them are useless,
and i didn't want to brick my hard drive trying to write a 4TB pandas
dataframe full of mostly shite
'''

def is_descending(x):
    return all(x[i] >= x[i + 1] for i in range(len(x) - 1))

def generate_iterators(n_t_tuple):
    meshes = {}
    nums = {}
    eiters = []
    for name in ['dE0', 'i_p', 'e_t']:
        nums[name] = np.round((ga.bounds[name][1] - ga.bounds[name][0])
                       / ga.increments[name]).astype(int)
        meshes[name] = np.linspace(*ga.bounds[name], num=nums[name])
    e0s = [meshes['dE0'] for _ in range(constants.n_rc)]
    ips = [meshes['i_p'] for _ in range(constants.n_rc)]
    e0ip = e0s + ips
    dd = itertools.product(*e0ip)
    for nti in n_t_tuple:
        eiters.append(itertools.product(meshes['e_t'], repeat=nti))
    return itertools.product(dd, *eiters)

def check_viability(iteration):
    # not the cleanest possible way of doing this probably. but it works
    for ei in range(constants.n_rc):
        if (iteration[ei + 1][0] > 
           -iteration[0][constants.n_rc + ei] + iteration[0][ei]):
            return False
        # iteration[0] is ([dE0], [i_p]), i.e. 2 vectors of length n_rc
        # iteration[1:] is ([traps on RC 0], [traps on RC 1], ...)
        # and these can be of different lengths depending on n_t.
        if not is_descending(iteration[ei + 1]):
            return False
    return True

def check_restart(iteration, restart_point):
    '''
    compare a loop iteration to ga.gt instance restart_point
    and return True if all_close
    '''
    tests = np.zeros(constants.n_rc + 2, dtype=bool)
    tests[0] = np.allclose(iteration[0][:constants.n_rc], restart_point[0])
    tests[1] = np.allclose(iteration[0][constants.n_rc:], restart_point[1])
    # restart_point[2] is k_cs and restart_point[4] is k_t
    # we are currently not iterating over rates so ignore these
    for rci, nti in enumerate(restart_point[3]):
        tests[nti + 2] = np.allclose(iteration[rci + 1][:nti],
                                     restart_point[5][:nti])
    return np.all(tests)

spectrum, output_prefix = light.stellar(Tstar=5697, Rstar=0.994,
                                        a=1.0, attenuation=0.0)
fif = light.fractional_integrated_flux(spectrum)
good_configs = []
good_cutoff = 0.1

e0arr = np.zeros(constants.n_rc, dtype=ga.ft)
iparr = np.zeros_like(e0arr)
kcsarr = np.zeros_like(e0arr)
kcsarr.fill(1e12)
etarr = np.zeros((constants.n_rc, constants.n_t_max), dtype=ga.ft)
ktarr = np.zeros_like(etarr)

parr = np.zeros(1, ga.gt)
pp = parr[0]
pp['k_cs'] = kcsarr

n_processed = 0
n_rejected = 0
n_good = 0

restart_found = False
n_skipped = 0
restart_point = (
            [4.0, 4.0], 
            [7.791666666666667, 6.75], 
            [1000000000000.0, 1000000000000.0], 
            [1, 1], 
            [[1000000000000.0, np.nan, np.nan, np.nan, np.nan], 
             [1000000000000.0, np.nan, np.nan, np.nan, np.nan]], 
            [[-5.9655172413793105, np.nan, np.nan, np.nan, np.nan], 
             [-3.896551724137931, np.nan, np.nan, np.nan, np.nan]], 
            0.0010558417248717061, 
            [[1.191796876414199e-08, 0.6194169799973075], 
             [0.04724911415249722, 4.804769303423381e-05]])

n_gf = 2361 # starting file number, update as necessary

for nts in itertools.product(range(1, constants.n_t_max),
                             repeat=constants.n_rc):
    iterator = generate_iterators(nts)
    for comb in iterator:
        while not restart_found:
            at_restart = check_restart(comb, restart_point)
            if at_restart:
                print("found the restart point!")
                print(f"iteration details: {comb}")
                print(f"number of iterations skipped: {n_skipped}")
                restart_found = True
            else:
                n_skipped += 1
                if n_skipped % 1000000 == 0:
                    print(f"n_skipped = {n_skipped}")
                    print(comb, restart_point)
                continue

        if not check_viability(comb):
            n_rejected += 1
            continue
        etarr.fill(np.nan)
        ktarr.fill(np.nan)
        e0arr = comb[0][:constants.n_rc]
        pp['dE0'] = e0arr
        iparr = comb[0][constants.n_rc:]
        pp['i_p'] = iparr
        pp['n_t'] = np.array(nts, dtype=ga.it)
        for ei in range(constants.n_rc):
            etarr[ei][:nts[ei]] = comb[ei + 1][:nts[ei]]
            ktarr[ei][:nts[ei]] = 1e12
        pp['k_t'] = ktarr
        pp['e_t'] = etarr
        oo, rr, der = solvers.solve(pp, fif, debug=False)
        pp['output'] = oo
        pp['redox'] = rr
        n_processed += 1
        if n_processed % 1000 == 0:
            print(f"Number of inputs processed: {n_processed}")
            print(f"Number of inputs rejected: {n_rejected}")
            print(f"Number of good inputs: {n_good}")
            print()
        if pp['output'] > good_cutoff:
            print(pp)
            good_configs.append(pp)
            n_good += 1
            if n_good % 1000 == 0:
                fn = os.path.join("out", f"good_configs_{n_gf}.pkl")
                with open(fn, "wb") as f:
                    pickle.dump(good_configs, f)
                n_gf += 1

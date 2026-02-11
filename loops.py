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

spectrum, output_prefix = light.stellar(Tstar=5697, Rstar=0.994,
                                        a=1.0, attenuation=0.0)
fif = light.fractional_integrated_flux(spectrum)
good_configs = []
good_cutoff = 0.001

parr = np.zeros(1, ga.gt)
pp = parr[0]
meshes = {}
nums = {}
for name in ['dE0', 'i_p', 'e_t']:
    nums[name] = np.round((ga.bounds[name][1] - ga.bounds[name][0])
                   / ga.increments[name]).astype(int)
    meshes[name] = np.linspace(*ga.bounds[name], num=nums[name])

e0s = [meshes['dE0'] for _ in range(constants.n_rc)]
ips = [meshes['i_p'] for _ in range(constants.n_rc)]

e0arr = np.zeros(constants.n_rc, dtype=ga.ft)
iparr = np.zeros_like(e0arr)
kcsarr = np.zeros_like(e0arr)
kcsarr.fill(1e12)
pp['k_cs'] = kcsarr

etarr = np.zeros((constants.n_rc, constants.n_t_max), dtype=ga.ft)
ktarr = np.zeros_like(etarr)

e0ip = e0s + ips
dd = itertools.product(*e0ip)
n_processed = 0
n_good = 0
n_gf = 0
for nts in itertools.product(range(1, constants.n_t_max),
                             repeat=constants.n_rc):
    eiters = []
    for nti in nts:
        eiters.append(itertools.product(meshes['e_t'], repeat=nti))
    for comb in itertools.product(dd, *eiters):
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
        if n_processed % 100 == 0:
            print(f"Number of inputs processed: {n_processed}")
        if pp['output'] > good_cutoff:
            print(pp)
            good_configs.append(pp)
            n_good += 1
            if n_good % 1000 == 0:
                fn = os.path.join("out", "good_configs_{n_gf}.pkl")
                with open(fn, "w") as f:
                    pickle.dump(good_configs, f)
                n_gf += 1

# -*- coding: utf-8 -*-
"""
9/4/24
@author: callum
i keep writing different tests in different files and deleting them
for various reasons. so just keep them all here instead
"""
import numpy as np
import genetic_algorithm as ga
import antenna as la
import constants
import light

def test_connected_kmat():
    '''
    generate a pair of identical antennae with one connected and one
    not. solve the equations and print out all the debug info for both
    to check that the rate matrix is reasonable etc.
    '''
    spectrum_dict = [{'type': "am15", 'kwargs': {'dataset': "tilt"}}]
    spectrum, out_name = zip(*light.build(spectrum_dict))
    # zip(*x) returns a 1-tuple rather than just the array
    l = spectrum[0][:, 0]
    ip_y = spectrum[0][:, 1]
    rng = np.random.default_rng()

    c = ga.new(rng, 'random')
    c.connected = True
    u = ga.copy(c)
    u.connected = False
    cd = la.antenna(l, ip_y, c, debug=True)
    ud = la.antenna(l, ip_y, u, debug=True)

    return [cd, ud]

def test_connected_fitness(n_trials):
    '''
    test the connected property of antennae by creating pairs of
    identical antennae with one connected and one not. keep track
    of the results and return two (n_trials, 3) arrays
    '''
    connected_res = np.zeros((n_trials, 3), dtype=np.float64)
    unconnected_res = np.zeros((n_trials, 3), dtype=np.float64)
    spectrum_dict = [{'type': "am15", 'kwargs': {'dataset': "tilt"}}]
    spectrum, out_name = zip(*light.build(spectrum_dict))
    l = spectrum[0][:, 0]
    ip_y = spectrum[0][:, 1]
    rng = np.random.default_rng()

    for i in range(n_trials):
        c = ga.new(rng, 'random')
        c.connected = True
        u = ga.copy(c)
        u.connected = False
        cr = la.antenna(l, ip_y, c)
        ur = la.antenna(l, ip_y, u)
        connected_res[i] = cr
        unconnected_res[i] = ur
        diff = np.abs(cr - ur)
        print(str.format(("Iteration {:d}: Δν_e = {:10.4e}, "
        + "Δϕ_e(γ) = {:10.6e}, Δϕ_e = {:10.6e}"), i, *diff))

    return [connected_res, unconnected_res]

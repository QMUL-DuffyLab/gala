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
        '''
        set nu_e and phi_e values
        this is done by hand in main.py; the reasoning is that if
        using a non-Python kernel to do the calculations, it might
        not be aware of the dataclass structure (doesn't need to be),
        so it can't set the class members. might be worth writing a
        wrapper to do it though, since i forgot about it until i ran
        this code and expected the values to be there.
        alternatively just change ga.fitness() to take nu_e directly
        as an argument, but this will break if we ever change the
        fitness criterion.
        '''
        c.phi_e   = cr[0]
        c.phi_e_g = cr[1]
        c.phi_e   = cr[2]
        u.phi_e   = ur[0]
        u.phi_e_g = ur[1]
        u.phi_e   = ur[2]
        connected_res[i] = cr
        unconnected_res[i] = ur
        cf = ga.fitness(c)
        uf = ga.fitness(u)
        print(i, cf, uf)

    return [connected_res, unconnected_res]

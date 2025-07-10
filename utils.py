# -*- coding: utf-8 -*-
"""
17/1/24
@author: callum
utilities - lineshape and free energy calculations
"""
import os
from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
import xarray as xr
import numpy as np
import constants
import genetic_algorithm as ga

hcnm = (h * c) / (1.0E-9)

def lookups(spectrum):
    '''
    precalculate gamma (photon absorption rates) and overlaps
    for the set of pigments and shifts available.
    for overlaps, the first pair of indices is the emitter, second
    is the absorber
    '''
    l = spectrum[:, 0]
    fp_y = (spectrum[:, 1] * l) / hcnm
    shifts = np.arange(*ga.genome_parameters['shift']['bounds'])
    pigments = ga.genome_parameters['pigment']['bounds']
    gammas = xr.DataArray(np.zeros((len(pigments), len(shifts))),
                          coords = [pigments, shifts],
                          dims = ["pigment", "shift"])
    overlaps = xr.DataArray(np.zeros((len(pigments), len(shifts),
                                      len(pigments), len(shifts))),
                          coords = [pigments, shifts, pigments, shifts],
                          dims = ["p1", "s1", "p2", "s2"])
    for p1 in pigments:
        for s1 in shifts:
            # s1 is the integer shift which is multiplied by shift_inc
            # use the integer for indexing so we don't have to worry
            # about floating point precision, but we need to do the
            # multiplication for the actual calculation
            s = s1 * constants.shift_inc
            a1 = absorption(l, p1, s)
            e1 = emission(l, p1, s)
            n1 = overlap(l, a1, e1)
            gammas.loc[p1, s1] = constants.sig_chl * overlap(l, fp_y, a1)
            for p2 in pigments:
                for s2 in shifts:
                    s = s2 * constants.shift_inc
                    a2 = absorption(l, p2, s)
                    e2 = emission(l, p2, s)
                    n2 = overlap(l, a2, e2)
                    overlaps.loc[p1, s1, p2, s2] = overlap(l, e1, a2) / n2
    return overlaps, gammas

def absorption(l, pigment, shift):
    '''
    return lineshape of pigment shifted by lp
    NB: we multiply by shift_inc outside this function: really it would make
    more sense to change this and just use the Genome and an index, i think
    '''
    try:
        params = constants.pigment_data[pigment]['abs']
    except KeyError:
        print(f"loading absorption data failed for {pigment}")
        print(constants.pigment_data[pigment])
    lp = [x + shift for x in params['mu']]
    g = gauss(l, lp, params['sigma'], params['amp'])
    return g

def emission(l, pigment, shift):
    '''
    return emission lineshape for these parameters
    NB: we multiply by shift_inc outside this function: really it would make
    more sense to change this and just use the Genome and an index, i think
    '''
    try:
        params = constants.pigment_data[pigment]['ems']
    except KeyError:
        print(f"loading emission data failed for {pigment}")
        print(constants.pigment_data[pigment])
    lp = [x + shift for x in params['mu']]
    g = gauss(l, lp, params['sigma'], params['amp'])
    return g

def gauss(l, mu, sigma, a = None):
    '''
    return a normalised gaussian lineshape. if you give it one peak
    it'll use that, if you give it a list of peaks it'll
    add them together. assumes that w and a are arrays
    of the same length as lp.
    '''
    g = np.zeros_like(l)
    if isinstance(mu, float):
        g = np.exp(-1.0 * (l - mu)**2/(2.0 * sigma**2))
    else:
        for i in range(len(mu)):
            g += a[i] * np.exp(-1.0 * (l - mu[i])**2/(2.0 * sigma[i]**2))
    n = np.trapz(g, l)
    return g/n

def overlap(l, f1, f2):
    return np.trapz(f1 * f2, l)

def dG(l1, l2, n, T):
    '''
    Gibbs free energy difference between two blocks of pigments with
    absorption peaks l1 and l2 (give in nm!) and ratio n = n1/n2, where n1
    and n2 are the number of pigments in each.
    '''
    h12 = hcnm * ((l1 - l2) / (l1 * l2))
    s12 = -kB * np.log(n)
    return h12 - (s12 * T)

def peak(shift, pigment):
    '''
    returns the 0-0 line for a given index on a given individual
    assume this is the fluorescence 0-0 line because we assume
    ultrafast equilibriation on each block.
    '''
    params = constants.pigment_data[pigment]
    return shift + params['ems']['0-0']

# -*- coding: utf-8 -*-
"""
17/1/24
@author: callum
branched antenna with saturating reaction centre
"""
import os
from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
from scipy.optimize import nnls
import numpy as np
import constants
import ctypes
import plots
import light
import matplotlib.pyplot as plt

hcnm = (h * c) / (1.0E-9)

'''
a lot of the stuff we need to build the matrices is precomputable;
this class precomputes it. note that i still need to decide once and
for all how the averaged chlorophyll instead of named pigments is gonna
work, but in the end it might be easier to just write a few functions to
do each bit and then compose them in different ways depending on whether
we have a set of named pigments, an averaged shifting one, or both
'''

def lookups(spectrum, pigment_list, return_lineshapes=False):
    l = spectrum[:, 0]
    fp_y = (spectrum[:, 1] * l) / hcnm
    # first index is emitter, second absorber
    abso = {p1: absorption(l, p1, 0.0) for p1 in pigment_list}
    emis = {p1: emission(l, p1, 0.0) for p1 in pigment_list}
    # normalise by requiring that the overlap of the absorbing
    # block *with an identical emitting block* should be 1.
    norm = {p1: overlap(l, abso[p1], emis[p1]) for p1 in pigment_list}
    overlaps = {emitter: 
                {absorber: (overlap(l, e, a) / norm[absorber])
                for absorber, a in abso.items()}
                for emitter, e in emis.items()
                }
    # for n_p = 1
    gammas = {absorber: constants.sig_chl * overlap(l, fp_y, a)
              for absorber, a in abso.items()}
    if return_lineshapes:
        return overlaps, gammas, abso, emis
    else:
        return overlaps, gammas

def precalculate_peak_locations(pigment, lmin, lmax, increment):
    # NB: this won't work, only considering absorption
    op = constants.pigment_data[pigment]['abs']['mu'][0]
    # ceil in both cases - for the min, it's -ve so rounds towards zero,
    # for the max we want to go one step outside the range because
    # np.arange(min, max, step) generates [min, max)
    smin = op + np.ceil((lmin - op) / increment) * increment
    smax = op + np.ceil((lmax - op) / increment) * increment
    # if we turn off shift, set increment to 0 or set min and max to
    # the same number, np.arange will return an empty array; fix that
    if smin == smax or increment == 0.0:
        peaks = np.array([smin])
    else:
        peaks = np.arange(smin, smax, increment)
    # return a dict so we can index with the shift to get peak location
    return {shift: peak for shift, peak in zip(peaks - op, peaks)}

def precalculate_overlap_gamma(pigment, rcs, spectrum, shift_peak):
    # NB: don't use this yet! needs to use emission lines
    n_shifts = len(shift_peak.keys())
    ntot = n_shifts + len(rcs)
    gamma = np.zeros(ntot)
    lines = np.zeros((ntot, len(spectrum[:, 0])))
    for i, s in enumerate(shift_peak.keys()):
        lines[i] = absorption(spectrum[:, 0], pigment, s)
        gamma[i] = (constants.sig_chl *
                    overlap(spectrum[:, 0], spectrum[:, 1], lines[i]))
    for i in range(len(rcs)):
        lines[n_shifts + i] = absorption(spectrum[:, 0], rcs[i], 0.0)
        gamma[n_shifts + i] = (constants.sig_chl *
                    overlap(spectrum[:, 0], spectrum[:, 1],
                            lines[n_shifts + i]))
    
    overlaps = np.zeros((len(lines), len(lines)))
    for i, l1 in enumerate(lines):
        for j, l2 in enumerate(lines):
            overlaps[i, j] = overlap(spectrum[:, 0], l1, l2)
    return gamma, overlaps

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

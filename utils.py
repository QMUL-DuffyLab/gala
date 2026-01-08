# -*- coding: utf-8 -*-
"""
17/1/24
@author: callum
utilities - lineshape and free energy calculations
"""
import os
import zipfile
import pickle
from scipy.constants import h, c, e
from scipy.constants import Boltzmann as kB
import numpy as np
import constants
import genetic_algorithm as ga

hcnm = (h * c) / (1.0E-9)
beta_cm = 1.0 / (kB * constants.T * (1. / (100.0 * c))) # in wavenumbers
beta_ev = 1.0 / (kB * constants.T / e) # in eV

@np.vectorize
def ev_nm(x):
    ''' convert eV to nm and vice-versa '''
    return (hcnm / e) * (1.0 / x)

@np.vectorize
def nm_wvn(l):
    ''' convert nm to wavenumber '''
    return (1e-7 * l)

def db(e1, e2, k12):
    '''
    apply detailed balance to a single rate k12, where the initial/final
    states 1 and 2 (process is 1 -> 2) have energies e1, e2 (in eV).
    obviously this should always be applied to pairs of rates as below
    '''
    fac = np.exp(-(e1 - e2) * beta_ev)
    return k12 * fac

def db_pair(e1, e2, k12, k21):
    '''
    apply detailed balance for a pair of states 1 and 2,
    with energies e1 e2 (in eV) and rates k12 for 1 -> 2,
    k21 for 2 -> 1. these are bare rates which are then modified by the gap
    (we assume that dS = 0 and only the energy gap is important).
    return the rates [fw, bw] where fw is 1 -> 2, bw is 2 -> 1
    '''
    return [db(e1, e2, k12), db(e2, e1, k21)]
    '''
    explaining the following commented line:
    it's for the case where we only want to penalise the uphill process.
    e1 - e2 is positive if the forward (1 -> 2) process is decreasing
    in energy; in this case, np.sign(gap) will return 1
    and the reverse (backward) process should be penalised.
    it's negative if the forward process is *increasing* in energy,
    in which case np.sign(gap) returns -1 and fw should be penalised.
    finally, if they're equal, gap is 0 and np.sign(gap) = 0 as well.
    (np.sign(gap) + 1 // 2) therefore returns 1, 0, 0 in each of these
    three cases respectively. since we consider rate[0] to be forward and
    rate[1] to be backward, 1, 0, 0 are the array indices of the rate
    which should be penalised in each case (in the third case actually
    the exponential is identically 1 anyway, so there's no penalty)
    '''
    # rates[(np.sign(gap) + 1) // 2] *= np.exp(-gap * beta)
    # return rates

def get_hash_table(prefix):
    f = os.path.join(prefix, "hash_table.zip")
    if os.path.isfile(f):
        with zipfile.ZipFile(f,
                mode="r", compression=zipfile.ZIP_BZIP2) as archive:
            pf = archive.read(f"hash_table.pkl")
            table = pickle.loads(pf)
            print(f"Hash table found for {prefix}.")
    else:
        table = {}
        print(f"No hash table found for {prefix}. Generating a new one.")
    return table

def save_hash_table(hash_table, prefix):
    htf = os.path.join(prefix, f"hash_table.pkl")
    with open(htf, "wb") as f:
        pickle.dump(hash_table, f)
    zipfilename = os.path.splitext(htf)[0] + ".zip"
    with zipfile.ZipFile(zipfilename,
            mode="w", compression=zipfile.ZIP_BZIP2) as archive:
            archive.write(htf, arcname=os.path.basename(htf))
    os.remove(htf)
    return

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

def peak(shift, pigment, which):
    '''
    returns the 0-0 line for a given index on a given individual
    assume this is the fluorescence 0-0 line because we assume
    ultrafast equilibriation on each block.
    '''
    params = constants.pigment_data[pigment]
    return shift + params[which]['mu'][0]

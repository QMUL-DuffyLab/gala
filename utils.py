# -*- coding: utf-8 -*-
"""
17/1/24
@author: callum
utilities - lineshape and free energy calculations
"""
import os
import zipfile
import pickle
from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
import numpy as np
import constants
import genetic_algorithm as ga
import rc as rcm

hcnm = (h * c) / (1.0E-9)

def generate_arrays():
    '''
    generate a dict containing a set of keyed arrays; each
    one should hold the set of possible values of a relevant Genome
    member, so that we can use np.where or similar to index into a big
    array of overlaps
    '''
    d = {}
    for k, v in ga.genome_parameters.items():
        if ga.get_type(k) == "U10":
            if k == 'pigment':
                # the RC pigments should be included here!
                antenna_pigments = ga.genome_parameters['pigment']['bounds']
                rcs = ga.genome_parameters['rc']['bounds']
                rc_pigments = [rcm.params[rc]['pigments'] for rc in rcs]
                # rc_pigments will be a list of lists; flatten it
                rc_flat = [rc for rcs in rc_pigments for rc in rcs]
                all_pigments = antenna_pigments + rc_flat
                d[k] = all_pigments
            else:
                d[k] = v['bounds']
        elif ga.get_type(k) == np.float64:
            # lookup tables for float parameters would require some
            # kind of mesh, snapping to the nearest value, etc, and
            # we do not currently need that. so set to None and move on
            d[k] = None
        elif ga.get_type(k) == np.int32:
            d[k] = np.arange(v['bounds'][0], v['bounds'][1] + 1)
        else:
            raise TypeError("utils.generate_indices() can't find type")
    return d

index_arrays = generate_arrays()

def get_index(parameter, value):
    '''
    complementary function to generate_arrays - search the
    dict of arrays for the parameter and the value given; if not
    found, raise a KeyError. i think this is the behaviour we want,
    because who knows what else could go wrong if it returns an invalid
    value for whatever reason
    '''
    if parameter in index_arrays:
        index = np.where(index_arrays[parameter] == value)
    else:
        raise KeyError("invalid key given to utils.get_index")
    return index

def lookups(spectrum):
    '''
    precalculate gamma (photon absorption rates) and overlaps
    for the set of pigments and shifts available.
    for overlaps, the first pair of indices is the emitter, second
    is the absorber
    '''
    l = spectrum[:, 0]
    fp_y = (spectrum[:, 1] * l) / hcnm
    p_length = len(index_arrays['pigment'])
    s_length = len(index_arrays['shift'])
    overlaps = np.zeros((p_length, s_length,
        p_length, s_length), dtype=np.float64)
    gammas = np.zeros((p_length, s_length), dtype=np.float64)
    for i in range(p_length):
        for j in range(s_length):
            # s1 is the integer shift which is multiplied by shift_inc
            # use the integer for indexing so we don't have to worry
            # about floating point precision, but we need to do the
            # multiplication for the actual calculation
            p1 = index_arrays['pigment'][i]
            s1 = index_arrays['shift'][j]

            s = s1 * constants.shift_inc
            a1 = absorption(l, p1, s)
            e1 = emission(l, p1, s)
            n1 = overlap(l, a1, e1)
            gammas[i, j] = constants.sig_chl * overlap(l, fp_y, a1)

            for k in range(p_length):
                for m in range(s_length):
                    p2 = index_arrays['pigment'][k]
                    s2 = index_arrays['shift'][m]

                    s = s2 * constants.shift_inc
                    a2 = absorption(l, p2, s)
                    e2 = emission(l, p2, s)
                    n2 = overlap(l, a2, e2)
                    overlaps[p1, s1, p2, s2] = overlap(l, e1, a2) / n2
    return gammas, overlaps

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

def calc_rates(p, spectrum, **kwargs):
    '''
    calculate the set of rates we need to build the matrix of
    equations to solve. this is basically a way of abstracting out all
    the dataclass definition stuff so that i can speed up the building
    of the matrices; presumably that just moves the bottleneck to here,
    but hopefully makes things faster overall.

    inputs:
    -------
    p - an instance of ga.Genome
    spectrum - a spectrum from light.py

    outputs:
    --------
    gamma - vector of input rates
    k_b - vector of branch transfer rates
    rc_mat - matrix of RC rates
    '''
    l = spectrum[:, 0]
    ip_y = spectrum[:, 1]
    fp_y = (ip_y * l) / hcnm
    rcp = rcm.params[p.rc]
    n_rc = len(rcp["pigments"])
    rc_n_p = [constants.pigment_data[rc]["n_p"] for rc in rcp["pigments"]]
    n_p = np.array([*rc_n_p, *p.n_p], dtype=np.int32)
    # 0 shift for RCs. shifts stored as integer increments, so
    # multiply by shift_inc here
    shift = np.array([*[0 for _ in range(n_rc)], *p.shift],
                     dtype=np.float64)
    shift *= constants.shift_inc
    pigment = np.array([*rcp["pigments"], *p.pigment], dtype='U10')
    a_l = np.zeros((p.n_s + n_rc, len(l)))
    e_l = np.zeros_like(a_l)
    norms = np.zeros(len(pigment))
    gamma = np.zeros(p.n_s + n_rc, dtype=np.float64)
    k_b = np.zeros(2 * (n_rc + p.n_s), dtype=np.float64)
    got_lookups = False
    k_cyc = np.nan
    if 'lookups' in kwargs:
        try:
            gammas, overlaps = kwargs['lookups']
            got_lookups = True
        except:
            gammas = None
            overlaps = None
            print("kwarg 'lookups' passed to solver incorrectly")
            print(kwargs['lookups'])
            raise
    # if we have the lookups, the shifts need to be integers, because
    # that's how they're encoded in the xarray (we don't want to worry about
    # rounding errors etc.). if we don't have them, shifts should be the
    # actual float values, so multiply by the increment
    if got_lookups:
        shift = np.array([*[0 for _ in range(n_rc)], *p.shift],
                         dtype=np.int64)
    else:
        shift = np.array([*[0.0 for _ in range(n_rc)], *p.shift],
                         dtype=np.float64)
        shift *= constants.shift_inc
    for i in range(p.n_s + n_rc):
        if got_lookups:
            gamma[i] = (n_p[i] * gammas[
                get_index("pigment", pigment[i]),
                get_index("shift", shift[i])
                ])
        else:
            a_l[i] = absorption(l, pigment[i], shift[i])
            e_l[i] = emission(l, pigment[i], shift[i])
            norms[i] = overlap(l, a_l[i], e_l[i])
            gamma[i] = (n_p[i] * constants.sig_chl *
                            overlap(l, fp_y, a_l[i]))

    for i in range(p.n_s + n_rc):
        if i < n_rc:
            # RCs - overlap/dG with 1st subunit (n_rc + 1 in list, so [n_rc])
            ind1 = i
            ind2 = n_rc
        elif i >= n_rc and i < (p.n_s + n_rc - 1):
            # one subunit and the next
            ind1 = i
            ind2 = i + 1
        n = float(n_p[ind1]) / float(n_p[ind2])
        if got_lookups:
            pass # need to revamp this
        else:
            inward  = overlap(l, a_l[ind1], e_l[ind2]) / norms[ind1]
            dgi = dG(peak(shift[ind2], pigment[ind2], 'ems'),
                    peak(shift[ind1], pigment[ind1], 'abs'),
                    1./n, constants.T)
            if dgi > 0.0:
                inward *= np.exp(-1.0 * dgi / (constants.T * kB))

            outward = overlap(l, e_l[ind1], a_l[ind2]) / norms[ind2]
            dgo = dG(peak(shift[ind1], pigment[ind1], 'ems'),
                    peak(shift[ind2], pigment[ind2], 'abs'),
                    n, constants.T)
            if dgo > 0.0:
                outward *= np.exp(-1.0 * dgo / (constants.T * kB))
        k_b[2 * i] = constants.k_hop * outward
        k_b[(2 * i) + 1] = constants.k_hop * inward
        '''
        the first n_rc pairs of rates are the transfer to and from
        the excited state of the RCs and the antenna. these are
        modified by the stoichiometry of these things. for now this
        is hardcoded but it's definitely possible to fix, especially
        if moving genome parameters into a file and generating from
        that: have a JSON parameter like "array": True/False and then
        "array_len": "n_s" or "rc", which can be used in the GA
        '''
        for j in range(n_rc):
            # TODO: check this before starting on samir's data
            if 'rho' in ga.genome_parameters:
                k_b[2 * j] *= (p.rho[j] * p.rho[-1])
                k_b[2 * j + 1] *= (p.rho[j] * p.rho[-1])
            if 'aff' in ga.genome_parameters:
                k_b[2 * j] *= p.aff[j]
                k_b[2 * j + 1] *= p.aff[j]

    '''
    rc.py generates matrices representing initial and final states
    and the processes that connect them. but some of those processes
    are genome-dependent, e.g. cyclic depends on the number of RCs, the
    size of the antenna, and so on. likewise trapping can only occur
    in the presence of some exciton manifold, i.e. the combined antenna-RC
    system, which rc.py doesn't know about. so i make the matrix out of
    strings in rc.py to denote which process is happening, and then here
    we do the quantitative part for this specific genome.
    '''
    n_rc_states = len(rcp["states"]) # total number of states of all RCs
    rc_mat = np.zeros((n_rc_states, n_rc_states), dtype=np.float64)
    for i in range(n_rc_states):
        for k in range(n_rc_states):
            rt = rcp['mat'][i][k]
            if rt != '':
                rc_mat[i][k] = rcm.rates[rt]
            if rt == "lin" and 'rho' in ga.genome_parameters:
                # do not ask why this works. it's to do with
                # how the RC states are indexed in rc.py and then
                # into the matrix. took a while to work out.
                which_rc = np.abs(k - i) // (n_rc - 1) % 2
                rc_mat[i][k] *= (p.rho[which_rc]
                    * p.rho[which_rc + 1])
            if rt == "ox" and 'diff_ratios' in kwargs:
                # get diffusion time for the relevant RC type
                if p.rc in kwargs['diff_ratios']:
                    ratio = kwargs['diff_ratios'][p.rc]
                rc_mat[i][k] = rcm.rates[rt] / (1.0 + ratio)
            if rt == "trap":
                '''
                this one's a problem, because trapping or detrapping
                change the state of the exciton manifold as well as the
                RC. to index these properly we need to make sure the
                population moves from the correct block, but this
                function doesn't know about the blocks, only the RC
                state. so set this to np.nan, look for the nans
                in the setup functions and deal with it there.
                '''
                rc_mat[i][k] = np.nan
            if rt == "cyc":
                # cyclic: multiply the rate by alpha etc.
                # we will need this below for nu_cyc
                rc_mat[i][k] = 0.0
                which_rc = ((n_rc - 1) -
                np.round(np.log(i - k) / np.log(4.0)).astype(int))
                k_cyc = rcm.rates["cyc"]
                if n_rc == 1:
                    # zeta = 11 to enforce nu_CHO == nu_cyc
                    k_cyc *= (11.0 + constants.alpha * np.sum(n_p))
                    rc_mat[i][k] = k_cyc
                # first photosystem cannot do cyclic
                elif n_rc > 1 and which_rc > 0:
                    k_cyc *= constants.alpha * np.sum(n_p)
                    rc_mat[i][k] = k_cyc
                # recombination can occur from any photosystem
                rc_mat[i][k] += rcm.rates["rec"]

    return gamma, k_b, rc_mat, k_cyc

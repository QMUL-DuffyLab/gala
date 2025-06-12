# -*- coding: utf-8 -*-
"""
14/5/25
@author: callum
putting all the different solvers in one module
"""
import os
import ctypes
import numpy as np
from scipy.optimize import nnls
from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
import constants
import light
import utils
import genetic_algorithm as ga
import rc as rcm

def solve(k, method='fortran', debug=False):
    '''
    solve the nnls problem using the given method.
    default's fortran, otherwise it'll use scipy.
    note that k should be given fortran-ordered if using
    the fortran solver; it's constructed that way below
    '''
    m = k.shape[0]
    n = k.shape[1]
    b = np.zeros(m, dtype=ctypes.c_double)
    b[-1] = 1.0
    if method == 'fortran':
        doubleptr = ctypes.POINTER(ctypes.c_double)
        intptr = ctypes.POINTER(ctypes.c_int)
        libnnls = ctypes.CDLL("./libnnls.so")
        libnnls.solve.argtypes = [doubleptr, doubleptr, doubleptr,
                                 intptr, intptr,
                                 intptr, doubleptr,
                                 intptr, doubleptr]
        libnnls.solve.restype = None
        mode = ctypes.c_int(0)
        maxiter = ctypes.c_int(3 * n)
        tol = ctypes.c_double(1e-13)

        b = np.zeros(k.shape[0], dtype=ctypes.c_double)
        b[-1] = 1.0
        p_eq_res = ctypes.c_double(0.0)
        p_eq = np.zeros(n, dtype=ctypes.c_double)
        libnnls.solve(k.ctypes.data_as(doubleptr),
                     b.ctypes.data_as(doubleptr),
                     p_eq.ctypes.data_as(doubleptr),
                     ctypes.c_int(m),
                     ctypes.c_int(n),
                     ctypes.byref(mode),
                     ctypes.byref(p_eq_res),
                     ctypes.byref(maxiter),
                     ctypes.byref(tol))
        if (mode.value < 0):
            p_eq = None
            p_eq_res = None
            if debug:
                print("Fortran reached max iterations")

    elif method == 'scipy':
        try:
            p_eq, p_eq_res = nnls(k, b, maxiter=1000)
        except RuntimeError:
            p_eq = None
            p_eq_res = None
            print("NNLS RuntimeError - reached iteration limit")
    return p_eq, p_eq_res


def antenna_only(l, ip_y, p, overlaps, gammas, debug=False):
    '''
    branched antenna, saturating RC
    l = set of wavelengths
    ip_y = irradiances at those wavelengths
    p = instance of constants.Genome
    set debug = True to output a dict with a load of info in it

    NB: this will not currently work. unsure if it'll ever be needed
    again, honestly
    '''
    n_p = np.array([constants.np_rc, *p.n_p], dtype=np.int32)
    # NB: shifts would have to be calculated here as well, if
    # we're including them down the line
    pigment = np.array([*p.rc, *p.pigment], dtype='U10')
    gamma = np.zeros(p.n_s, dtype=np.float64)
    k_b = np.zeros(2 * p.n_s, dtype=np.float64)

    for i in range(p.n_s):
        # the RC is at index 0, so increasing index means moving
        # outwards. the first index of overlaps is the emitter, so
        # inward is overlaps[outer_pigment][inner_pigment] and
        # outward is overlaps[inner_pigment][outer_pigment]
        gamma[i] = (n_p[i + len(p.rc)] * gammas[pigment[i + len(p.rc)]])
        # NB: this will need modifying if there are multiple RCs
        inward = overlaps[pigment[i + 1]][pigment[i]]
        outward = overlaps[pigment[i]][pigment[i + 1]]
        n = float(n_p[i]) / float(n_p[i + 1])
        dg = utils.dG(utils.peak(0.0, pigment[i]),
                utils.peak(0.0, pigment[i + 1]), n, constants.T)
        # [0] index below is transfer from RC to antenna, so outward first
        k_b[2 * i] = constants.k_hop * outward
        k_b[(2 * i) + 1] = constants.k_hop * inward
        if dg < 0.0:
            k_b[(2 * i) + 1] *= np.exp(dg / (constants.T * kB))
        elif dg > 0.0:
            k_b[2 * i] *= np.exp(-1.0 * dg / (constants.T * kB))

    side = (p.n_b * p.n_s) + 2
    twa = np.zeros((2 * side, 2 * side), dtype=np.longdouble)
    k = np.zeros(((2 * side) + 1, 2 * side), dtype=ctypes.c_double,
                 order='F')
    twa[1][0] = rcm.rates['red'] # 1e+2
    twa[2][0] = constants.k_diss # 2.5e+8
    twa[2][1] = rcm.rates['trap'] # 2e+11
    twa[3][1] = constants.k_diss
    twa[3][2] = rcm.rates['red'] # 1e+2
    js = list(range(4, 2 * side, 2 * p.n_s))
    for jind, j in enumerate(js):
        # two pairs of RC <-> rates at the bottom of each branch */
        twa[2][j]     = k_b[0] # 0 1 0   -> 1_i 0 0
        twa[j][2]     = k_b[1] # 1_i 0 0 -> 0 1 0
        twa[3][j + 1] = k_b[0] # 0 1 1   -> 1_i 0 1
        twa[j + 1][3] = k_b[1] # 1_i 0 1 -> 0 1 1
        for i in range(p.n_s):
            ind = j + (2 * i)
            twa[ind][0]       = constants.k_diss
            twa[ind + 1][1]   = constants.k_diss
            twa[ind + 1][ind] = rcm.rates['red']
            if p.connected:
                prevind = ind - (2 * p.n_s)
                nextind = ind + (2 * p.n_s)
                '''
                first four states are the trap and RC. if we're
                crossing the "boundary" (first <-> last)
                we need to take these into account
                '''
                if jind == 0: # first branch
                    prevind -= 4
                if jind == (p.n_b - 1): # final branch
                    nextind += 4
                # PBCs, essentially
                if prevind < 0:
                    prevind += 2 * side
                if nextind >= 2 * side:
                    nextind -= 2 * side
                if debug:
                    print(p.n_b, p.n_s, j, i, ind, prevind, nextind, 2 * side)
                '''
                8 possible transfers to consider:
                - both forward and backward transfer,
                - from both the clockwise and anticlockwise neighbour,
                - with the trap either empty or full.
                note: no need to consider dG; adjacent blocks are identical
                '''
                twa[ind][nextind] = constants.k_hop
                twa[nextind][ind] = constants.k_hop
                twa[ind + 1][nextind + 1] = constants.k_hop
                twa[nextind + 1][ind + 1] = constants.k_hop
                twa[ind][prevind] = constants.k_hop
                twa[prevind][ind] = constants.k_hop
                twa[ind + 1][prevind + 1] = constants.k_hop
                twa[prevind + 1][ind + 1] = constants.k_hop

            if i > 0:
                twa[ind][ind - 2]     = k_b[(2 * i) + 1] # empty trap
                twa[ind + 1][ind - 1] = k_b[(2 * i) + 1] # full trap
            if i < (p.n_s - 1):
                twa[ind][ind + 2]     = k_b[2 * (i + 1)] # empty
                twa[ind + 1][ind + 3] = k_b[2 * (i + 1)] # full
            twa[0][ind]     = gamma[i] # 0 0 0 -> 1_i 0 0
            twa[1][ind + 1] = gamma[i] # 0 0 1 -> 1_i 0 1

    for i in range(2 * side):
        for j in range(2 * side):
            if (i != j):
                k[i][j]      = twa[j][i]
                k[i][i]     -= twa[i][j]
        # add a row for the probability constraint
        k[2 * side][i] = 1.0

    b = np.zeros((2 * side) + 1, dtype=np.float64)
    b[-1] = 1.0
    p_eq, p_eq_res = solve(k, method='scipy')
    if p_eq is None:
        # couldn't find a solution - return fitness/efficiency of 0
        return np.array([0.0, 0.0, 0.0])

    n_eq = np.zeros(side, dtype=np.float64)
    for i in range(side):
        n_eq[0] += p_eq[(2 * i) + 1] # P(1_i, 1)
        if i > 0:
            n_eq[i] = p_eq[2 * i] + p_eq[(2 * i) + 1] # P(1_i, 0) + P(1_i, 1)
    nu_e = rcm.rates['red'] * n_eq[0]
    phi_e_g = nu_e / (nu_e + (constants.k_diss * np.sum(n_eq[1:])))

    # efficiency
    k_phi = np.zeros_like(k)
    gamma_sum = np.sum(gamma)
    gamma_norm = constants.gamma_fac * gamma / (gamma_sum)
    for j in range(4, 2 * side, 2 * p.n_s):
        for i in range(p.n_s):
            ind = j + (2 * i)
            twa[0][ind]     = gamma_norm[i]
            twa[1][ind + 1] = gamma_norm[i]

    for i in range(2 * side):
        ks = 0.0
        for j in range(2 * side):
            if (i != j):
                k_phi[i][j]  = twa[j][i]
                k_phi[i][i] -= twa[i][j]
        k_phi[2 * side][i] = 1.0

    b[:] = 0.0
    b[-1] = 1.0

    p_eq_low, p_eq_res_low = solve(k_phi, method='scipy')
    if p_eq_low is None:
        # couldn't find a solution - return fitness/efficiency of 0
        return np.array([0.0, 0.0, 0.0])

    n_eq_low = np.zeros(side, dtype=np.float64)
    for i in range(side):
        n_eq_low[0] += p_eq_low[(2 * i) + 1]
        if i > 0:
            n_eq_low[i] = p_eq_low[2 * i] + p_eq_low[(2 * i) + 1]
    nu_e_low = rcm.rates['red'] * n_eq_low[0]
    phi_e = nu_e_low / (nu_e_low + (constants.k_diss * np.sum(n_eq_low[1:])))

    if debug:
        return {
                'nu_e': nu_e,
                'nu_e_low': nu_e_low,
                'phi_e_g': phi_e_g,
                'phi_e': phi_e,
                'N_eq': n_eq,
                'N_eq_low': n_eq_low,
                'P_eq': p_eq,
                'P_eq_residuals': p_eq_res,
                'P_eq_low': p_eq_low,
                'P_eq_residuals_low': p_eq_res_low,
                'gamma': gamma,
                'gamma_total': np.sum(gamma),
                'K_mat': k,
                'k_b': k_b,
                }
    else:
        return np.array([nu_e, phi_e_g, phi_e])

def RC_only(rc_type, spectrum, **kwargs):
    '''
    parameters
    ----------
    `rc_type`: string corresponding to params above
    `spectrum`: input spectrum from light.py

    set up an RC-only system of equations of type `rc_type` (see params above)
    photosystems, and solve the resulting equations using scipy NNLS
    as in antenna.py and supersystem.py.
    this is fairly simple - there's no transfer rates to worry about,
    only excitation, dissipation, and the internal RC processes which
    are all generated and indexed previously.
    '''
    pdata = constants.pigment_data
    rcp = rcm.params[rc_type]
    n_rc = len(rcp["pigments"])
    n_rc_states = len(rcp["states"])
    fp_y = (spectrum[:, 0] * spectrum[:, 1]) / utils.hcnm
    if 'n_p' in kwargs:
        n_p = [kwargs['n_p'] for i in range(n_rc)]
    else:
        n_p = [pdata[rcp["pigments"][i]]['n_p'] for i in range(n_rc)]
    a_l = np.zeros((n_rc, len(spectrum[:, 0])), dtype=np.float64)
    gamma = np.zeros(n_rc, dtype=np.float64)
    for i in range(n_rc):
        a_l[i] = utils.absorption(spectrum[:, 0], rcp["pigments"][i], 0.0)
        gamma[i] = (n_p[i] * constants.sig_chl *
            utils.overlap(spectrum[:, 0], fp_y, a_l[i]))

    # detrapping regime
    detrap = 0.0 # none by default
    if 'detrap' in kwargs:
        detrap_type = kwargs['detrap']
        if detrap_type == "fast":
            rcm.rates["trap"]
        elif detrap_type == "thermal":
            detrap *= np.exp(-1.0) # -k_B T
        elif detrap_type == "energy_gap":
            detrap *= np.exp(-rcp["gap"])
        elif detrap_type == "none":
            detrap *= 0.0 # irreversible
        else:
            raise ValueError("Detrapping regime should be 'fast',"
              " 'thermal', 'energy_gap' or 'none'.")
        
    # n_rc_states for each exciton block, plus empty
    side = n_rc_states * (n_rc + 1)
    twa = np.zeros((side, side), dtype=np.float64)

    # generate states to go with p_eq indices
    # what order are these in? figure it out lol
    ast = []
    empty = tuple([0 for _ in range(n_rc)])
    ast.append(empty)
    for i in range(n_rc):
        el = [0 for _ in range(n_rc)]
        el[i] = 1
        ast.append(tuple(el))
    total_states = [s1 + tuple(s2) for s1 in ast for s2 in rcp["states"]]
    toti = {i: total_states[i] for i in range(len(total_states))}

    lindices = []
    cycdices = []
    js = list(range(0, side, n_rc_states))
    for jind, j in enumerate(js):
        # jind == 0 is empty antenna, 0 + n_rc_states is RC 1 occupied, etc
        # intra-RC processes are the same in each block
        for i in range(n_rc_states):
            ind = i + j # total index
            ts = toti[ind] # total state tuple
            initial = rcp["states"][i] # tuple with current RC state
            if i in rcp["nu_e_ind"]:
                lindices.append(ind)
            if i in rcp["nu_cyc_ind"]:
                cycdices.append(ind)

            for k in range(n_rc_states):
                final = rcp["states"][k]
                diff = tuple(final - initial)
                if diff in rcp["procs"]:
                    # get the type of process
                    rt = rcp["procs"][diff]
                    indf = rcp["indices"][tuple(final)] + j
                    ts = toti[indf] # total state tuple
                    # set element with the corresponding rate
                    if rt in ["lin", "red"]:
                        twa[ind][indf] = rcm.rates[rt]
                    if rt == "ox":
                        # if given, diff_ratios should be a dict of the
                        # form {'ox': 0.1, 'anox': 1.0}, etc.
                        # each number is the ratio of diffusion time to
                        # oxidation time, i.e. if it's > 1 substrate
                        # oxidation is diffusion-limited.
                        # ratio = 0 represents instantaneous diffusion;
                        # start there and modify if a ratio was given
                        ratio = 0.0
                        if 'diff_ratios' in kwargs:
                            if p.rc in kwargs['diff_ratios']:
                                ratio = kwargs['diff_ratios'][p.rc]
                        twa[ind][indf] = rcm.rates[rt] / (1.0 + ratio)
                    if rt == "trap":
                        # find which trap state is being filled here
                        # 3 states per rc, so integer divide by 3
                        which_rc = np.where(np.array(diff) == 1)[0][0]//3
                        '''
                        indf above assumes that the state of the antenna
                        doesn't change, which is not the case for trapping.
                        so zero out the above rate and then check: if
                        jind == which_rc + 1 we're in the correct block
                        (the exciton is moving from the correct RC), and
                        we go back to the empty antenna block
                        '''
                        twa[ind][indf] = 0.0
                        if jind == which_rc + 1:
                            indf = rcp["indices"][tuple(final)]
                            twa[ind][indf] = rcm.rates[rt]
                            # detrapping:
                            # - only possible if exciton manifold is empty
                            indf = (rcp["indices"][tuple(initial)] +
                                    ((which_rc + 1) * n_rc_states))
                            twa[k][indf] = detrap
                            rt = "detrap"
                    if rt == "cyc":
                        # cyclic: multiply the rate by alpha etc.
                        # we will need this below for nu(cyc)
                        which_rc = np.where(np.array(diff) == -1)[0][0]//3
                        k_cyc = rcm.rates["cyc"]
                        if n_rc == 1:
                            # zeta = 11 to enforce nu_CHO == nu_cyc
                            k_cyc *= (11.0 + constants.alpha * np.sum(n_p))
                            # k_cyc *= (constants.alpha * np.sum(n_p))
                            twa[ind][indf] = k_cyc
                            rt = "ano cyclic"
                        # first photosystem cannot do cyclic
                        elif n_rc > 1 and which_rc > 0:
                            k_cyc *= constants.alpha * np.sum(n_p)
                            twa[ind][indf] = k_cyc
                            rt = "cyclic"
                        # recombination can occur from any photosystem
                        twa[ind][indf] += rcm.rates["rec"]
            if jind > 0:
                # occupied exciton block -> empty due to dissipation
                # final state index is i because RC state is unaffected
                twa[ind][i] = constants.k_diss
            
            if jind > 0 and jind <= n_rc:
                twa[i][ind] = gamma[jind - 1] # absorption by RCs

    k = np.zeros((side + 1, side), dtype=np.float64,
                 order='F')
    for i in range(side):
        for j in range(side):
            if (i != j):
                k[i][j]      = twa[j][i]
                k[i][i]     -= twa[i][j]
        # add a row for the probability constraint
        k[side][i] = 1.0

    b = np.zeros(side + 1, dtype=np.float64)
    b[-1] = 1.0
    p_eq, p_eq_res = solve(k, method='fortran')

    nu_e = 0.0
    nu_cyc = 0.0
    trap_indices = [3*i + (n_rc) for i in range(n_rc)]
    oxidised_indices = [3*i + (n_rc + 1) for i in range(n_rc)]
    reduced_indices = [3*i + (n_rc + 2) for i in range(n_rc)]
    redox = np.zeros((n_rc, 2), dtype=np.float64)
    recomb = np.zeros(n_rc, dtype=np.float64)

    for i, p_i in enumerate(p_eq):
        s = toti[i]
        for j in range(n_rc):
            if s[trap_indices[j]] == 1:
                recomb += p_i * rcm.rates["rec"]
            if s[oxidised_indices[j]] == 1:
                redox[j, 0] += p_i
            if s[reduced_indices[j]] == 1:
                redox[j, 1] += p_i
        if i in lindices:
            nu_e += rcm.rates["red"] * p_i
        if i in cycdices:
            nu_cyc += k_cyc * p_i

    if 'debug' in kwargs and kwargs['debug']:
        return {
                "k": k,
                "twa": twa,
                "gamma": gamma,
                "p_eq": p_eq,
                "states": total_states,
                "lindices": lindices,
                "cycdices": cycdices,
                "nu_e": nu_e,
                "nu_cyc": nu_cyc,
                "redox": redox,
                "recomb": recomb,
                }
    else:
        return (nu_e, nu_cyc, redox, recomb)

def antenna_RC(p, spectrum, **kwargs):
    '''
    generate matrix for combined antenna-RC supersystem and solve it.

    parameters
    ----------
    l = set of wavelengths
    ip_y = irradiances at those wavelengths
    p = instance of constants.Genome
    set debug = True to output a dict with a load of info in it
    nnls = which NNLS version to use
    detrap_type = string for what detrapping regime to simulate

    outputs
    -------
    if debug == True:
    debug: a huge dict containing various parameters that are useful to me
    (and probably only me) in figuring out what the fuck is going on.
    else:
    TBC. probably nu_e and nu_cyc
    '''
    l = spectrum[:, 0]
    ip_y = spectrum[:, 1]
    fp_y = (ip_y * l) / utils.hcnm
    rcp = rcm.params[p.rc]
    n_rc = len(rcp["pigments"])
    rc_n_p = [constants.pigment_data[rc]["n_p"] for rc in rcp["pigments"]]
    n_p = np.array([*rc_n_p, *p.n_p], dtype=np.int32)
    # 0 shift for RCs. shifts stored as integer increments, so
    # multiply by shift_inc here
    shift = np.array([*[0.0 for _ in range(n_rc)], *p.shift],
                     dtype=np.float64)
    shift *= constants.shift_inc
    pigment = np.array([*rcp["pigments"], *p.pigment], dtype='U10')
    a_l = np.zeros((p.n_s + n_rc, len(l)))
    e_l = np.zeros_like(a_l)
    norms = np.zeros(len(pigment))
    gamma = np.zeros(p.n_s + n_rc, dtype=np.float64)
    k_b = np.zeros(2 * (n_rc + p.n_s), dtype=np.float64)
    for i in range(p.n_s + n_rc):
        a_l[i] = utils.absorption(l, pigment[i], shift[i])
        e_l[i] = utils.emission(l, pigment[i], shift[i])
        norms[i] = utils.overlap(l, a_l[i], e_l[i])
        gamma[i] = (n_p[i] * constants.sig_chl *
                        utils.overlap(l, fp_y, a_l[i]))

    # detrapping regime
    detrap = 0.0 # no detrapping by default
    if 'detrap' in kwargs:
        detrap_type = kwargs['detrap']
        if detrap_type == "fast":
            rcm.rates["trap"]
        elif detrap_type == "thermal":
            detrap *= np.exp(-1.0) # -k_B T
        elif detrap_type == "energy_gap":
            detrap *= np.exp(-rcp["gap"])
        elif detrap_type == "none":
            detrap *= 0.0 # irreversible
        else:
            raise ValueError("Detrapping regime should be 'fast',"
              " 'thermal', 'energy_gap' or 'none'.")

    for i in range(p.n_s + n_rc):
        ab = i
        el = -1
        if i < n_rc:
            # RCs - overlap/dG with 1st subunit (n_rc + 1 in list, so [n_rc])
            inward  = utils.overlap(l, a_l[i], e_l[n_rc]) / norms[i]
            el = n_rc
            outward = utils.overlap(l, e_l[i], a_l[n_rc]) / norms[n_rc]
            # print(inward, outward)
            n = float(n_p[i]) / float(n_p[n_rc])
            dg = utils.dG(utils.peak(shift[i], pigment[i]),
                    utils.peak(shift[n_rc], pigment[n_rc]), n, constants.T)
        elif i >= n_rc and i < (p.n_s + n_rc - 1):
            # one subunit and the next
            el = i + 1
            inward  = utils.overlap(l, a_l[i], e_l[i + 1]) / norms[i]
            outward = utils.overlap(l, e_l[i], a_l[i + 1]) / norms[i + 1]
            n = float(n_p[i]) / float(n_p[i + 1])
            dg = utils.dG(utils.peak(shift[i], pigment[i]),
                    utils.peak(shift[i + 1], pigment[i + 1]), n, constants.T)
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
        for i in range(n_rc):
            # odd - inward, even - outward
            if 'rho' in ga.genome_parameters:
                k_b[2 * i] *= (p.rho[i] * p.rho[-1])
                k_b[2 * i + 1] *= (p.rho[i] * p.rho[-1])
            if 'aff' in ga.genome_parameters:
                k_b[2 * i] *= p.aff[i]
                k_b[2 * i + 1] *= p.aff[i]
        if dg < 0.0:
            k_b[(2 * i) + 1] *= np.exp(dg / (constants.T * kB))
        elif dg > 0.0:
            k_b[2 * i] *= np.exp(-1.0 * dg / (constants.T * kB))

    n_rc_states = len(rcp["states"]) # total number of states of all RCs
    side = n_rc_states * ((p.n_b * p.n_s) + n_rc + 1)
    twa = np.zeros((side, side), dtype=np.longdouble)

    # generate states to go with p_eq indices
    # what order are these in? figure it out lol
    ast = []
    empty = tuple([0 for _ in range(n_rc + p.n_b * p.n_s)])
    ast.append(empty)
    for i in range(n_rc + p.n_b * p.n_s):
        el = [0 for _ in range(n_rc + p.n_b * p.n_s)]
        el[i] = 1
        ast.append(tuple(el))
    total_states = [s1 + tuple(s2) for s1 in ast for s2 in rcp["states"]]
    toti = {i: total_states[i] for i in range(len(total_states))}

    lindices = []
    cycdices = []
    js = list(range(0, side, n_rc_states))
    for jind, j in enumerate(js):
        # jind == 0 is empty antenna, 0 + n_rc_states is RC 1 occupied, etc
        # intra-RC processes are the same in each block
        for i in range(n_rc_states):
            ind = i + j # total index
            ts = toti[ind] # total state tuple
            initial = rcp["states"][i] # tuple with current RC state
            if i in rcp["nu_e_ind"]:
                lindices.append(ind)
            if i in rcp["nu_cyc_ind"]:
                cycdices.append(ind)

            for k in range(n_rc_states):
                final = rcp["states"][k]
                diff = tuple(final - initial)
                if diff in rcp["procs"]:
                    # get the type of process
                    rt = rcp["procs"][diff]
                    indf = rcp["indices"][tuple(final)] + j
                    ts = toti[indf] # total state tuple
                    # set the correct element with the corresponding rate
                    if rt == "red":
                        twa[ind][indf] = rcm.rates[rt]
                    if rt == "lin":
                        # the first place where the population decreases
                        # is the first in the chain of linear flow
                        which_rc = np.where(np.array(diff) == -1)[0][0]//3
                        twa[ind][indf] = rcm.rates[rt]
                        if 'rho' in ga.genome_parameters:
                            twa[ind][indf] *= (p.rho[which_rc]
                                * p.rho[which_rc + 1])
                    if rt == "ox":
                        # get diffusion time for the relevant RC type
                        ratio = 0.0
                        if 'diff_ratios' in kwargs:
                            if p.rc in kwargs['diff_ratios']:
                                ratio = kwargs['diff_ratios'][p.rc]
                        twa[ind][indf] = rcm.rates[rt] / (1.0 + ratio)
                    if rt == "trap":
                        which_rc = np.where(np.array(diff) == 1)[0][0]//3
                        # find which trap state is being filled here
                        # 3 states per rc, so integer divide by 3
                        '''
                        indf above assumes that the state of the antenna
                        doesn't change, which is not the case for trapping.
                        so zero out the above rate and then check: if
                        jind == which_rc + 1 we're in the correct block
                        (the exciton is moving from the correct RC), and
                        we go back to the empty antenna block
                        '''
                        twa[ind][indf] = 0.0
                        if jind == which_rc + 1:
                            indf = rcp["indices"][tuple(final)]
                            twa[ind][indf] = rcm.rates[rt]
                            # detrapping:
                            # - only possible if exciton manifold is empty
                            indf = (rcp["indices"][tuple(initial)] +
                                    ((which_rc + 1) * n_rc_states))
                            twa[k][indf] = detrap
                            rt = "detrap"
                    if rt == "cyc":
                        # cyclic: multiply the rate by alpha etc.
                        # we will need this below for nu_cyc
                        which_rc = np.where(np.array(diff) == -1)[0][0]//3
                        k_cyc = rcm.rates["cyc"]
                        if n_rc == 1:
                            # zeta = 11 to enforce nu_CHO == nu_cyc
                            k_cyc *= (11.0 + constants.alpha * np.sum(n_p))
                            # k_cyc *= (constants.alpha * np.sum(n_p))
                            twa[ind][indf] = k_cyc
                            rt = "ano cyclic"
                        # first photosystem cannot do cyclic
                        elif n_rc > 1 and which_rc > 0:
                            k_cyc *= constants.alpha * np.sum(n_p)
                            twa[ind][indf] = k_cyc
                            rt = "cyclic"
                        # recombination can occur from any photosystem
                        twa[ind][indf] += rcm.rates["rec"]

            if jind > 0:
                # occupied exciton block -> empty due to dissipation
                # final state index is i because RC state is unaffected
                twa[ind][i] = constants.k_diss
            
            if jind > 0 and jind <= n_rc:
                twa[i][ind] = gamma[jind - 1] # absorption by RCs

            # antenna rate stuff
            if jind > n_rc: # population in antenna subunit
                # index on branch
                bi = (jind - n_rc - 1) % p.n_s
                twa[i][ind] = gamma[n_rc + bi] # absorption by this block
                if bi == 0:
                    # root of branch - transfer to RC exciton states possible
                    for k in range(n_rc):
                        # transfer to RC 0 is transfer to jind 1
                        offset = (n_rc - k) * n_rc_states
                        # inward transfer to RC k
                        twa[ind][ind - offset] = k_b[2 * k + 1]
                        # outward transfer from RC k
                        twa[ind - offset][ind] = k_b[2 * k]
                if bi > 0:
                    # inward along branch
                    twa[ind][ind - n_rc_states] = k_b[2 * (n_rc + bi) - 1]
                if bi < (p.n_s - 1):
                    # outward allowed
                    twa[ind][ind + n_rc_states] = k_b[2 * (n_rc + bi) + 1]

    k = np.zeros((side + 1, side), dtype=ctypes.c_double,
                 order='F')
    for i in range(side):
        for j in range(side):
            if (i != j):
                k[i][j]      = twa[j][i]
                k[i][i]     -= twa[i][j]
        # add a row for the probability constraint
        k[side][i] = 1.0

    b = np.zeros(side + 1, dtype=np.float64)
    b[-1] = 1.0
    if 'nnls' in kwargs:
        p_eq, p_eq_res = solve(k, method=kwargs['nnls'])
    else:
        p_eq, p_eq_res = solve(k, method='fortran')

    # nu_e === nu_ch2o here; we treat them as identical
    nu_e = 0.0
    nu_cyc = 0.0
    # [::-1] to reverse here because otherwise redox below will
    # output the redox states of the RCs in reverse order
    trap_indices = [-(3 + 3*i) for i in range(n_rc)][::-1]
    oxidised_indices = [-(2 + 3*i) for i in range(n_rc)][::-1]
    reduced_indices = [-(1 + 3*i) for i in range(n_rc)][::-1]
    redox = np.zeros((n_rc, 2), dtype=np.float64)
    recomb = np.zeros(n_rc, dtype=np.float64)
    trap_states = []
    ox_states = []
    red_states = []

    '''
    check here whether the solver actually found a solution or not.
    do it here because redox and recomb have to be allocated, otherwise
    the shapes of the returned values would differ based on the
    success/failure of the solver, and we don't want that.
    if it failed, return a tuple that tells main.py it failed
    '''
    if p_eq_res == None:
        print("NNLS failure. Genome details:")
        print(p)
        return ({'nu_e': nu_e, 'nu_cyc': nu_cyc,
            'redox': redox, 'recomb': recomb}, -1)

    for i, p_i in enumerate(p_eq):
        s = toti[i]
        for j in range(n_rc):
            if s[trap_indices[j]] == 1:
                recomb += p_i * rcm.rates["rec"]
                trap_states.append(s)
            if s[oxidised_indices[j]] == 1:
                redox[j, 0] += p_i
                ox_states.append(s)
            if s[reduced_indices[j]] == 1:
                redox[j, 1] += p_i
                red_states.append(s)
        if i in lindices:
            nu_e += rcm.rates["red"] * p_i
        if i in cycdices:
            nu_cyc += k_cyc * p_i

    if 'debug' in kwargs and kwargs['debug']:
        return {
                "k": k,
                "twa": twa,
                "gamma": gamma,
                "k_b": k_b,
                "p_eq": p_eq,
                "states": total_states,
                "lindices": lindices,
                "cycdices": cycdices,
                "trap_states": trap_states,
                "ox_states": ox_states,
                "red_states": red_states,
                "redox": redox,
                "recomb": redox,
                "nu_e": nu_e,
                "nu_cyc": nu_cyc,
                'a_l': a_l,
                'e_l': e_l,
                'norms': norms,
                'k_b': k_b,
                }
    else:
        return ({'nu_e': nu_e, 'nu_cyc': nu_cyc,
            'redox': redox, 'recomb': recomb}, 0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    spectrum, output_prefix = light.spectrum_setup("stellar",
            Tstar=5772, Rstar=6.957E8, a=1.0, attenuation=0.0)
    print(len(spectrum[:, 0]))
    n_b = 2
    n_s = 2
    pigment = ['bchl_b', 'chl_f']
    n_p = [65 for _ in range(n_s)]
    shift = [0 for _ in range(n_s)]
    rc_type = "anox"
    names = rcm.params[rc_type]["pigments"] + pigment
    rho = [1.0, 1.0] # equal stoichiometry
    aff = [1.0] # ps_ox, ps_r affinity
    p = ga.Genome(rc_type, n_b, n_s, n_p, shift, pigment)

    od = antenna_RC(p, spectrum, debug=True)
    print(f"Branch rates k_b: {od['k_b']}")
    print(f"Raw overlaps of F'(p) A(p): {od['norms']}")

    side = len(od["p_eq"])
    for i in range(side):
        colsum = np.sum(od["k"][:side, i])
        rowsum = np.sum(od["k"][i, :])
        print(f"index {i}: state {od['states'][i]} sum(col[i]) = {colsum}, sum(row[i]) = {rowsum}")
    print(np.sum(od["k"][:side, :]))
    print(f"alpha = {constants.alpha}, rho = {rho}, aff = {aff}")
    print(f"p(0) = {od['p_eq'][0]}")
    print(f"nu_e = {od['nu_e']}")
    print(f"nu_cyc = {od['nu_cyc']}")

    n_rc = len(rcm.params[rc_type]["pigments"])
    sg = np.sum(od['gamma'][:n_rc]) + n_b * np.sum(od['gamma'][n_rc:])
    print(f"total excitation rate = {sg} s^-1")
    print(f"'efficiency' = {(od['nu_e'] + od['nu_cyc']) / sg}")
    print(f"k shape = {od['k'].shape}")
    for si, pi in zip(od["states"], od["p_eq"]):
        print(f"p_eq{si} = {pi}")
    print(f"k_b = {od['k_b']}")
    np.savetxt(f"out/antenna_{rc_type}_twa.dat", od["twa"], fmt='%.16e')
    np.savetxt(f"out/antenna_{rc_type}_k.dat", od["k"], fmt='%.6e')
    np.savetxt(f"out/antenna_{rc_type}_p_eq.dat", od["p_eq"], fmt='%.16e')
    with open(f"out/antenna_{rc_type}_results.dat", "w") as f:
    # print(od)
        f.write(f"alpha = {constants.alpha}, rho = {rho}, affinity = {aff}\n")
        f.write(f"gamma = {od['gamma']}\n")
        f.write(f"p(0) = {od['p_eq'][0]}\n")
        f.write(f"nu_e = {od['nu_e']}\n")
        f.write(f"nu_cyc = {od['nu_cyc']}\n")
        f.write(f"sum(gamma) = {np.sum(od['gamma'])}\n")
        for si, pi in zip(od["states"], od["p_eq"]):
            f.write(f"p_eq{si} = {pi}\n")
        f.write(f"kb = {od['k_b']}\n")

    print("States counted for nu(CH2O):")
    for i in od["lindices"]:
        print(f"{i}: {od['states'][i]}")
    print("States counted for nu(cyc):")
    for i in od["cycdices"]:
        print(f"{i}: {od['states'][i]}")
    print("Trap states")
    for s in od["trap_states"]:
        print(f"{s}")
    print("Oxidised states")
    for s in od["ox_states"]:
        print(f"{s}")
    print("Reduced states")
    for s in od["red_states"]:
        print(f"{s}")
    
    fig, ax = plt.subplots(nrows=len(names), figsize=(12,12), sharex=True)
    for i in range(len(names)):
        ax[i].plot(spectrum[:, 0], od['a_l'][i],
                color='C1', label=f"A ({names[i]})")
        ax[i].plot(spectrum[:, 0], od['e_l'][i],
                color='C0', label=f"F ({names[i]})")
        ax[i].legend()
        ax[i].grid(visible=True)
    fig.supylabel("intensity (arb)", x=0.001)
    ax[0].set_xlim(constants.x_lim)
    ax[-1].set_xlabel("wavelength (nm)")
    fig.savefig(f"out/{rc_type}_antenna_lineshape_test.pdf")
    plt.close(fig)


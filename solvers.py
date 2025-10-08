# -*- coding: utf-8 -*-
"""
14/5/25
@author: callum
putting all the different solvers in one module
"""
import os
import time
import ctypes
import numpy as np
from scipy.optimize import nnls as scipy_nnls
from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
import constants
import light
import utils
import genetic_algorithm as ga
import rc as rcm
import build_matrix

'''
note - the two different solver methods below (diag and NNLS)
necessarily have different things to return. the NNLS method will return
residuals so you can check how close the solution is if you choose; the
diagonalisation has the eigenvalues and eigenvectors, which I return for
inspection if the solver's set to debug mode. i've chosen to return
p_eq first and then k for both methods so that those are consistent at
least, but there's maybe a better way to do this by forwarding the debug
kwarg and changing the return values based on that, or something
'''

def diag(transfer_matrix, t=constants.tinf):
    '''
    calculate equilibrium occupation probabilites from the transfer
    matrix `transfer_matrix` at time t via diagonalisation
    '''
    side = transfer_matrix.shape[0]
    k = np.zeros((side, side), dtype=ctypes.c_double,
                 order='F')
    for i in range(side):
        for j in range(side):
            if i != j:
                k[i][j]      = transfer_matrix[j][i]
                k[i][i]     -= transfer_matrix[i][j]

    lam, C = np.linalg.eig(k)
    Cinv = np.linalg.inv(C)
    elt = np.zeros_like(C)
    p0 = np.zeros(k.shape[0])
    p0[0] = 1.0
    for i in range(k.shape[0]):
        elt[i, i] = np.exp(lam[i] * t)
    p_eq = np.matmul(np.matmul(np.matmul(C, elt), Cinv), p0)
    p_eq /= np.sum(p_eq)
    return np.real(p_eq), k, lam, C, Cinv

def nnls(transfer_matrix, method='fortran', debug=False):
    '''
    calculate equilibrium occupation probabilites from the transfer
    matrix `transfer_matrix` via non-negative least squares (NNLS).
    solves the nnls problem using the given method.
    default's fortran, otherwise it'll use scipy.
    '''

    side = transfer_matrix.shape[0]
    k = np.zeros((side + 1, side), dtype=ctypes.c_double,
                 order='F')
    for i in range(side):
        for j in range(side):
            if i != j:
                k[i][j]      = transfer_matrix[j][i]
                k[i][i]     -= transfer_matrix[i][j]
        # add a row for the probability constraint
        k[side][i] = 1.0

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
        try:
            libnnls.solve(k.ctypes.data_as(doubleptr),
                         b.ctypes.data_as(doubleptr),
                         p_eq.ctypes.data_as(doubleptr),
                         ctypes.c_int(m),
                         ctypes.c_int(n),
                         ctypes.byref(mode),
                         ctypes.byref(p_eq_res),
                         ctypes.byref(maxiter),
                         ctypes.byref(tol))
        except:
            print("libnnls error. what's happened here then?")
            print("parameters passed:")
            print(f"m = {ctypes.c_int(m)}")
            print(f"n = {ctypes.c_int(n)}")
            print(f"mode = {mode}")
            print(f"maxiter = {maxiter}")
            print(f"tol = {tol}")
            raise
        if mode.value < 0:
            p_eq = None
            p_eq_res = None
            if debug:
                print("Fortran reached max iterations")

    elif method == 'scipy':
        try:
            p_eq, p_eq_res = scipy_nnls(k, b, maxiter=1000)
        except RuntimeError:
            p_eq = None
            p_eq_res = None
            print("NNLS RuntimeError - reached iteration limit")
    return p_eq, k, p_eq_res

def RC_only(rc_type, spectrum, solver_method='diag', **kwargs):
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
            if i in rcp["inds"]["nu_e"]:
                lindices.append(ind)
            if i in rcp["inds"]["cyc"]:
                cycdices.append(ind)

            for k in range(n_rc_states):
                final = rcp["states"][k]
                diff = tuple(final - initial)
                if diff in rcp["procs"]:
                    # get the type of process
                    rt = rcp["procs"][diff]
                    indf = rcp["indices"][tuple(final)] + j
                    tf = toti[indf] # total state tuple
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
                            if rc_type in kwargs['diff_ratios']:
                                ratio = kwargs['diff_ratios'][rc_type]
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

    if solver_method == 'nnls':
        if 'nnls_method' in kwargs:
            p_eq, k, p_eq_res = nnls(twa, method=kwargs['nnls_method'])
        else:
            p_eq, k, p_eq_res = nnls(twa, method='fortran')
    elif solver_method == 'diag':
        if 'diag_time' in kwargs:
            p_eq, k, lam, C, Cinv = diag(twa, kwargs['diag_time'])
        else:
            p_eq, k, lam, C, Cinv = diag(twa)

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

def antenna_RC(p, spectrum, debug=False, do_redox=False,
        solver_method='diag', **kwargs):
    '''
    generate matrix for combined antenna-RC supersystem and solve it.

    parameters
    ----------
    p: instance of constants.Genome
    spectrum: a spectrum as output by light.py
    TODO: add details of possible kwargs!!

    outputs
    -------
    if debug == True:
    debug: a huge dict containing various parameters that are useful to me
    (and probably only me) in figuring out what the fuck is going on.
    else:
    nu_e, nu_c, possibly the redox information if do_redox = True
    '''
    output = {}
    start = time.time()

    gamma, k_b, rc_mat, k_cyc = utils.calc_rates(p, spectrum)

    n_rc = rcm.n_rc[p.rc]
    n_rc_states = 4**n_rc
    rcp = rcm.params[p.rc]

    if debug:
        end = time.time()
        output['setup_time'] = end - start
        output['gamma'] = gamma
        output['k_b'] = k_b
        start = time.time()

    twa = build_matrix.build_matrix(p.n_b, p.n_s, n_rc, rc_mat,
            constants.alpha, constants.k_diss, rcm.rates['trap'],
            rcm.rates['detrap'], gamma, k_b)

    if debug:
        output['twa'] = twa
        end = time.time()
        output['mat_time'] = end - start
        start = time.time()

    if solver_method == 'nnls':
        if 'nnls_method' in kwargs:
            p_eq, k, p_eq_res = nnls(twa, method=kwargs['nnls_method'])
        else:
            p_eq, k, p_eq_res = nnls(twa, method='fortran')
        if debug:
            output['p_eq'] = p_eq
            output['k'] = k
            output['p_eq_res'] = p_eq_res
    elif solver_method == 'diag':
        if 'diag_time' in kwargs:
            p_eq, k, lam, C, Cinv = diag(twa, kwargs['diag_time'])
        else:
            p_eq, k, lam, C, Cinv = diag(twa)
        if debug:
            output['p_eq'] = p_eq
            output['k'] = k
            output['lam'] = lam
            output['C'] = C

    if debug:
        end = time.time()
        solve_time = end - start
        output['solve_time'] = solve_time

    # nu_e === nu_ch2o here; we treat them as identical
    nu_e = 0.0
    nu_e_diag = 0.0
    nu_cyc = 0.0
    # [::-1] to reverse here because otherwise redox below will
    # output the redox states of the RCs in reverse order
    if do_redox:
        # generate states to go with p_eq indices
        ast = []
        empty = tuple([0 for _ in range(n_rc + p.n_b * p.n_s)])
        ast.append(empty)
        for i in range(n_rc + p.n_b * p.n_s):
            el = [0 for _ in range(n_rc + p.n_b * p.n_s)]
            el[i] = 1
            ast.append(tuple(el))
        total_states = [s1 + tuple(s2) for s1 in ast for s2 in rcp["states"]]
        toti = {i: total_states[i] for i in range(len(total_states))}
        tots = {total_states[i]: i for i in range(len(total_states))}
        if debug:
            output['states'] = toti

        trap_indices = [-(3 + 3*i) for i in range(n_rc)][::-1]
        oxidised_indices = [-(2 + 3*i) for i in range(n_rc)][::-1]
        reduced_indices = [-(1 + 3*i) for i in range(n_rc)][::-1]
        redox = np.zeros((n_rc, 2), dtype=np.float64)
        recomb = np.zeros(n_rc, dtype=np.float64)
        trap_states = []
        ox_states = []
        red_states = []

    for i, p_i in enumerate(p_eq):
        if i % n_rc_states in rcp["inds"]['nu_e']:
            nu_e += rcm.rates["red"] * p_i
        if i % n_rc_states in rcp["inds"]['cyc']:
            nu_cyc += k_cyc * p_i

        if do_redox:
            for j in range(n_rc):
                s = toti[i]
                if s[trap_indices[j]] == 1:
                    recomb += p_i * rcm.rates["rec"]
                    trap_states.append(i)
                if s[oxidised_indices[j]] == 1:
                    redox[j, 0] += p_i
                    ox_states.append(i)
                if s[reduced_indices[j]] == 1:
                    redox[j, 1] += p_i
                    red_states.append(i)
            if debug:
                output['trap_states'] = trap_states
                output['ox_states'] = ox_states
                output['red_states'] = red_states

    # |= is the new way to merge dicts but I have some stupid python
    # thing going on with jupyter where it's using 3.8 even
    # though i should have 3.11 on, so |= doesn't work
    od = {'nu_e': nu_e, 'nu_cyc': nu_cyc}
    if do_redox:
        od = {**od, **{"redox": redox, "recomb": redox}}
    if debug:
        od = {**od, **output}
    return od

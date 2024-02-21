# -*- coding: utf-8 -*-
"""
17/1/24
@author: callum
branched antenna with saturating reaction centre
"""

from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
from dataclasses import dataclass, field
from scipy.optimize import nnls
import numpy as np
import numpy.typing as npt
import constants
import ctypes

hcnm = (h * c) / (1.0E-9)

def get_lineshape(l, pigment, lp_offset):
    '''
    return lineshape of pigment shifted by lp
    '''
    params = constants.pigment_data[pigment]
    lp = [x + lp_offset for x in params['lp']]
    g = gauss(l, lp, params['w'], params['amp'])
    return g

def gauss(l, lp, w, a = None):
    '''
    return a gaussian lineshape. if you give it one peak
    it'll use that, if you give it a list of peaks it'll
    add them together. assumes that w and a are arrays
    of the same length as lp.
    '''
    g = np.zeros_like(l)
    if isinstance(lp, float):
        g = np.exp(-1.0 * (l - lp)**2/(2.0 * w**2))
    else:
        for i in range(len(lp)):
            g += a[i] * np.exp(-1.0 * (l - lp[i])**2/(2.0 * w[i]**2))
    n = np.trapz(g, l)
    return g/n

def overlap(l, f1, f2):
    return np.trapz(f1 * f2, l)

def dG(l1, l2, n, T):
    h12 = hcnm * ((l1 - l2) / (l1 * l2))
    s12 = -kB * np.log(n)
    return h12 - (s12 * T)

def peak(lp_offset, pigment):
    '''
    lp is now an offset, not an actual peak - this function
    returns the 0-0 line for a given index on a given individual
    '''
    params = constants.pigment_data[pigment]
    return lp_offset + params['lp'][0]

def antenna(l, ip_y, p, debug=False, test_lstsq=False):
    '''
    branched antenna, saturating RC
    p should be a genome either as defined above or as defined
    in constants.py (they take different arguments!).
    an example is given at the bottom in the name = main section
    set debug = True to output a dict with a load of info in it,
    set test_lstsq = True to solve with numpy's lstsq in addition
    to scipy's non-negative least squares solver and output both.
    '''
    fp_y = (ip_y * l) / hcnm
    # gonna switch over and use these instead
    n_p = np.array([constants.np_rc, *p.n_p], dtype=np.int32)
    # 0 offset for the RC!
    lp = np.array([0., *p.lp], dtype=np.float64)
    pigment = np.array(['rc', *p.pigment], dtype='U10')
    lines = np.zeros((p.n_s + 1, len(l)))
    gamma = np.zeros(p.n_s, dtype=np.float64)
    k_b = np.zeros(2 * p.n_s, dtype=np.float64)
    for i in range(p.n_s + 1):
        lines[i] = get_lineshape(l, pigment[i], lp[i])
        if i > 0:
            gamma[i - 1] = (n_p[i] * constants.sig_chl *
                            overlap(l, fp_y, lines[i]))
            # print("i, gamma[i], overlap", i, gamma[i - 1],
            #       overlap(l, fp_y, lines[i]))

    for i in range(p.n_s):
        de = overlap(l, lines[i], lines[i + 1])
        n = float(n_p[i]) / float(n_p[i + 1])
        dg = dG(peak(lp[i], pigment[i]),
                peak(lp[i + 1], pigment[i + 1]), n, constants.T)
        if i == 0:
            rate = constants.k_lhc_rc
        else:
            rate = constants.k_hop
        rate *= de
        k_b[2 * i] = rate
        k_b[(2 * i) + 1] = rate
        if dg < 0.0:
            k_b[(2 * i) + 1] *= np.exp(dg / (constants.T * kB))
        elif dg > 0.0:
            k_b[2 * i] *= np.exp(-1.0 * dg / (constants.T * kB))

    side = (p.n_b * p.n_s) + 2
    twa = np.zeros((2 * side, 2 * side), dtype=np.longdouble)
    k = np.zeros(((2 * side) + 1, 2 * side), dtype=ctypes.c_double, 
                 order='F')
    twa[1][0] = constants.k_con # 1e+2
    twa[2][0] = constants.k_diss # 2.5e+8
    twa[2][1] = constants.k_trap # 2e+11
    twa[3][1] = constants.k_diss
    twa[3][2] = constants.k_con
    for j in range(4, 2 * side, 2 * p.n_s):
        # two pairs of RC <-> rates at the bottom of each branch */
        twa[2][j]     = k_b[0] # 0 1 0   -> 1_i 0 0
        twa[j][2]     = k_b[1] # 1_i 0 0 -> 0 1 0
        twa[3][j + 1] = k_b[0] # 0 1 1   -> 1_i 0 1
        twa[j + 1][3] = k_b[1] # 1_i 0 1 -> 0 1 1
        for i in range(p.n_s):
            ind = j + (2 * i)
            twa[ind][0]       = constants.k_diss
            twa[ind + 1][1]   = constants.k_diss
            twa[ind + 1][ind] = constants.k_con
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
    if test_lstsq:
        p_eq_lstsq, p_eq_res_lstsq, rank, s = np.linalg.lstsq(k, b, rcond=None)
    else:
        p_eq_lstsq = None
        p_eq_res_lstsq = None

    # doubleptr = ctypes.POINTER(ctypes.c_double)
    # libnnls = ctypes.CDLL("./libnnls.so")
    # libnnls.nnls.argtypes = []
    # libnnls.nnls.restype = None
    # mode = ctypes.c_int(0)
    # res = ctypes.c_double(0.0)
    # maxiter = ctypes.c_int(3 * (2 * side))
    # tol = ctypes.c_double(1e-6)
    # p_eq_f = np.zeros(2 * side, dtype=ctypes.c_double)
    # print("testing fortran")
    # libnnls.nnls(k.ctypes.data_as(doubleptr),
    #              b.ctypes.data_as(doubleptr),
    #              p_eq_f.ctypes.data_as(doubleptr),
    #              mode, res, maxiter, tol)
    # print("fortran returned")

    try:
        p_eq, p_eq_res = nnls(k, b)
    except RuntimeError:
        p_eq = None
        p_eq_res = None
        nu_e = 0.0
        phi_e_g = 0.0
        phi_e = 0.0
        if debug:
            print("RuntimeError - nnls reached iteration limit. high intensity")
        return np.array([nu_e, phi_e_g, phi_e])

    # print("sum of difference between python and fortran: ", 
    #       np.sum(p_eq - p_eq_f))
    # print("Scipy nnls:")
    # print(p_eq, k @ p_eq, (k @ p_eq) - b)

    n_eq = np.zeros(side, dtype=np.float64)
    for i in range(side):
        n_eq[0] += p_eq[(2 * i) + 1] # P(1_i, 1)
        if i > 0:
            n_eq[i] = p_eq[2 * i] + p_eq[(2 * i) + 1] # P(1_i, 0) + P(1_i, 1)
    nu_e = constants.k_con * n_eq[0]
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

    if test_lstsq:
        p_eq_lstsq_low, p_eq_res_lstsq_low, rank, s = np.linalg.lstsq(k_phi,
                                                            b, rcond=None)
    else:
        p_eq_lstsq_low = None
        p_eq_res_lstsq_low = None

    try:
        p_eq_low, p_eq_res_low = nnls(k_phi, b)
    except RuntimeError:
        p_eq_low = None
        p_eq_res_low = None
        nu_e = 0.0
        phi_e_g = 0.0
        phi_e = 0.0
        if debug:
            print("RuntimeError - nnls reached iteration limit. low intensity")
        return np.array([nu_e, phi_e_g, phi_e])

    if np.any(p_eq_low < 0.0):
        print("negative probabilities in p_eq_low!")
        print(p_eq_low)

    n_eq_low = np.zeros(side, dtype=np.float64)
    for i in range(side):
        n_eq_low[0] += p_eq_low[(2 * i) + 1]
        if i > 0:
            n_eq_low[i] = p_eq_low[2 * i] + p_eq_low[(2 * i) + 1]
    nu_e_low = constants.k_con * n_eq_low[0]
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
                'P_eq_lstsq': p_eq_lstsq,
                'P_eq_low': p_eq_low,
                'P_eq_residuals_low': p_eq_res_low,
                'P_eq_low_lstsq': p_eq_lstsq_low,
                'gamma': gamma,
                'gamma_total': np.sum(gamma),
                'K_mat': k,
                }
    else:
        return np.array([nu_e, phi_e_g, phi_e])

if __name__ == '__main__':

    ts = "5800K"
    file = "PHOENIX/Scaled_Spectrum_PHOENIX_" + ts + ".dat"
    d = np.loadtxt(file)

    # changed behaviour - now the RC's added inside antenna
    n_b = 1
    n_p = [200]
    lp = [-6.0]
    # w = [10.0, 10.0, 10.0]
    n_s = len(n_p)
    # pigments = ['chl_a', 'chl_b', 'r_pe']
    pigments = ['chl_a']
    test = constants.genome(n_b, n_s, n_p, lp, pigments)

    od = antenna(d[:, 0], d[:, 1], test, True, True)
    print(od)

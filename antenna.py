# -*- coding: utf-8 -*-
"""
17/1/24
@author: callum
branched antenna with saturating reaction centre
"""

from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
from scipy.optimize import nnls
import numpy as np
import constants
import ctypes
import plots
import light

hcnm = (h * c) / (1.0E-9)

'''
a lot of the stuff we need to build the matrices is precomputable;
this class precomputes it. note that i still need to decide once and
for all how the averaged chlorophyll instead of named pigments is gonna
work, but in the end it might be easier to overload this class to take
a max and min peak and increment, and assume the averaged chlorophyll
if those are given. 
'''
class LookupTables:
    def __init__(self, spectrum, pigments, rcs):
        self.gamma = self.gamma_calc(spectrum, pigments)
        self.h     = self.enthalpy_calc(pigments, rcs)
        self.s     = self.entropy_calc(pigments, rcs)

    @classmethod
    def lineshapes(self, spectrum, peak_bounds, increment):
        peak = peak_bounds[0]
        while peak <= peak_bounds[1]:
            l = get_lineshape(spectrum[:, 0], ["avg"], peak)
            peak += increment

    def gamma_calc(self, spectrum, pigments):
        lines = np.zeros(len(pigments))
        gamma = np.zeros(len(pigments))
        for i in range(len(pigments)):
            lines[i] = get_lineshape(spectrum[:, 0], pigments[i], 0.)
            gamma[i] = (constants.sig_chl *
                                overlap(*spectrum, lines[i]))
        return gamma
    def enthalpy_calc(self, pigments, rcs):
        ntot = len(pigments) + len(rcs)
        h = np.zeros((ntot, ntot))
        # h12 = hcnm * ((l1 - l2) / (l1 * l2))
        return h
    def entropy_calc(self, pigments, rcs):
        ntot = len(pigments) + len(rcs)
        s = np.zeros((ntot, ntot))
        # s12 = -kB * np.log(n)
        return s



def get_lineshape(l, pigment, shift):
    '''
    return lineshape of pigment shifted by lp
    '''
    params = constants.pigment_data[pigment]
    lp = [x + shift for x in params['lp']]
    g = gauss(l, lp, params['w'], params['amp'])
    return g

def gauss(l, lp, w, a = None):
    '''
    return a normalised gaussian lineshape. if you give it one peak
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

def peak(shift, pigment):
    '''
    returns the 0-0 line for a given index on a given individual
    '''
    params = constants.pigment_data[pigment]
    return shift + params['lp'][0]

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
            p_eq, p_eq_res = nnls(k, b)
        except RuntimeError:
            p_eq = None
            p_eq_res = None
            if debug:
                print("RuntimeError - nnls reached iteration limit")

    return p_eq, p_eq_res

def antenna(l, ip_y, p, debug=False):
    '''
    branched antenna, saturating RC
    l = set of wavelengths
    ip_y = irradiances at those wavelengths
    p = instance of constants.Genome
    set debug = True to output a dict with a load of info in it
    '''
    fp_y = (ip_y * l) / hcnm
    n_p = np.array([constants.np_rc, *p.n_p], dtype=np.int32)
    # 0 offset for the RC!
    shift = np.array([0., *p.shift], dtype=np.float64)
    pigment = np.array([constants.rc_type, *p.pigment], dtype='U10')
    lines = np.zeros((p.n_s + 1, len(l)))
    gamma = np.zeros(p.n_s, dtype=np.float64)
    k_b = np.zeros(2 * p.n_s, dtype=np.float64)
    for i in range(p.n_s + 1):
        lines[i] = get_lineshape(l, pigment[i], shift[i])
        if i > 0:
            gamma[i - 1] = (n_p[i] * constants.sig_chl *
                            overlap(l, fp_y, lines[i]))

    for i in range(p.n_s):
        de = overlap(l, lines[i], lines[i + 1])
        n = float(n_p[i]) / float(n_p[i + 1])
        dg = dG(peak(shift[i], pigment[i]),
                peak(shift[i + 1], pigment[i + 1]), n, constants.T)
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
            twa[ind + 1][ind] = constants.k_con
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

    p_eq_low, p_eq_res_low = solve(k_phi, method='scipy')
    if p_eq_low is None:
        # couldn't find a solution - return fitness/efficiency of 0
        return np.array([0.0, 0.0, 0.0])

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
                'P_eq_low': p_eq_low,
                'P_eq_residuals_low': p_eq_res_low,
                'gamma': gamma,
                'gamma_total': np.sum(gamma),
                'K_mat': k,
                }
    else:
        return np.array([nu_e, phi_e_g, phi_e])

if __name__ == '__main__':

    ts = "5800K"
    f = "spectra/PHOENIX/Scaled_Spectrum_PHOENIX_" + ts + ".dat"
    d = np.loadtxt(f)

    # changed behaviour - now the RC's added inside antenna
    n_b = 5
    n_p = [90, 60, 80, 20, 50]
    shift = [-6.0, -8.0, -5.0, 5.0, -1.0]
    # w = [10.0, 10.0, 10.0]
    n_s = len(n_p)
    pigments = ['chl_a', 'chl_b', 'r_apc', 'r_pc', 'r_pe']
    test = constants.Genome(n_b, n_s, n_p, shift, pigments)
    plots.draw_antenna(test, "test_from_python.svg")

    od = antenna(d[:, 0], d[:, 1], test, True)
    print(od)

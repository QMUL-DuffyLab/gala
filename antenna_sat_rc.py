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

hcnm = (h * c) / (1.0E-9)

@dataclass()
class gaussian_genome:
    n_b: int = 0
    n_s: int = 0
    n_p: int = field(default_factory=lambda: np.empty([], dtype=np.int64))
    lp: float = field(default_factory=lambda: np.empty([], dtype=np.float64))
    w: float = field(default_factory=lambda: np.empty([], dtype=np.float64))
    pigment: str = field(default_factory=lambda: np.empty([], dtype='U10'))
    nu_e: float = np.nan
    phi_f: float = np.nan

def get_lineshape(l, p, i):
    '''
    return the lineshape corresponding to index i of genome p
    '''
    if hasattr(p, 'pigment'):
        # get the set of pigment lineshapes
        params = constants.pigment_data[p.pigment[i]]
        lp_offset = p.lp[i] - params['lp'][0]
        lp = [x + lp_offset for x in params['lp']]
        g = gauss(l, lp, params['w'], params['amp'])
    else:
        # if we're generating gaussians ourselves to stand in for
        # pigment absorption then at least l_p and width must be defined
        if hasattr(p, 'amp'): # but maybe not amplitudes
            g = gauss(l, p.lp[i], p.w[i], p.amp[i])
        else:
            g = gauss(l, p.lp[i], p.w[i])
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
    n_s = p.n_s - 1 # rc included
    fp_y = (ip_y * l) / hcnm
    lines = np.zeros((p.n_s, len(l)))
    gamma = np.zeros(n_s, dtype=np.float64)
    k_b = np.zeros(2 * n_s, dtype=np.float64)
    for i in range(p.n_s):
        lines[i] = get_lineshape(l, p, i)
        if i > 0:
            gamma[i - 1] = (p.n_p[i] * constants.sig_chl *
                            overlap(l, fp_y, lines[i]))

    for i in range(n_s):
        de = overlap(l, lines[i], lines[i + 1])
        n = float(p.n_p[i]) / float(p.n_p[i + 1])
        dg = dG(p.lp[i], p.lp[i + 1], n, constants.T)
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

    side = (p.n_b * n_s) + 2
    twa = np.zeros((2 * side, 2 * side), dtype=np.longdouble)
    k = np.zeros(((2 * side) + 1, 2 * side), dtype=np.float64)
    twa[1][0] = constants.k_con # 1e+2
    twa[2][0] = constants.k_diss # 2.5e+8
    twa[2][1] = constants.k_trap # 2e+11
    twa[3][1] = constants.k_diss
    twa[3][2] = constants.k_con
    for j in range(4, 2 * side, 2 * n_s):
        # two pairs of RC <-> rates at the bottom of each branch */
        twa[2][j]     = k_b[0] # 0 1 0   -> 1_i 0 0
        twa[j][2]     = k_b[1] # 1_i 0 0 -> 0 1 0
        twa[3][j + 1] = k_b[0] # 0 1 1   -> 1_i 0 1
        twa[j + 1][3] = k_b[1] # 1_i 0 1 -> 0 1 1
        for i in range(n_s):
            ind = j + (2 * i)
            twa[ind][0]       = constants.k_diss
            twa[ind + 1][1]   = constants.k_diss
            twa[ind + 1][ind] = constants.k_con
            if i > 0:
                twa[ind][ind - 2]     = k_b[(2 * i) + 1] # empty trap
                twa[ind + 1][ind - 1] = k_b[(2 * i) + 1] # full trap
            if i < (n_s - 1):
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

    p_eq, p_eq_res = nnls(k, b)
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
    for j in range(4, 2 * side, 2 * n_s):
        for i in range(n_s):
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

    p_eq_low, p_eq_res_low = nnls(k_phi, b)

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

    # note that n_p, lp and w include the RC as the first element!
    # this is just so i can generate everything in one set of loops
    n_b = 1
    # n_p = [1, 100, 100, 100, 100]
    # lp  = [constants.lp_rc, 670.0, 660.0, 650.0, 640.0]
    # w   = [constants.w_rc, 10.0, 10.0, 10.0, 10.0]
    n_p = [1, 50, 20, 100]
    lp = [constants.lp_rc, 650.0, 660.0, 620.0]
    w = [constants.w_rc, 10.0, 10.0, 10.0]
    n_s = len(n_p)
    # uncomment these to use the whole thing with named pigments
    # pigments = ['rc', 'chl_a', 'chl_b', 'r_pe']
    # test = constants.genome(n_b, n_s, n_p, lp, w)
    test = gaussian_genome(n_b, n_s, n_p, lp, w)

    od = antenna(d[:, 0], d[:, 1], test, True, True)
    print(od)

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

def precalculate_peak_locations(pigment, lmin, lmax, increment):
    op = constants.pigment_data[pigment]['lp'][0]
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

def precalculate_delta_G(pigment, rcs, shift_peak):
    ntot = len(shift_peak.keys()) + len(rcs)
    rce = [constants.pigment_data[rc]['lp'][0] for rc in rcs]
    dh = np.zeros((ntot, ntot))
    ds = np.zeros_like(dh)
    energies = np.zeros(n_tot)
    for i, l1 in enumerate(shift_peak.values()):
        energies[i] = l1
    for i in range(len(rcs)):
        energies[len(shift_peak.values()) + i] = rce[i]
    for i, l1 in enumerate(energies):
        for j, l2 in enumerate(energies):
            dh[i, j] = hcnm * ((l1 - l2) / (l1 * l2))
    # entropy - actually we could just calculate the lower triangle
    # of this matrix since it's symmetric, but whatever
    nbounds = constants.bounds['n_p']
    for i in range(n_bounds[0], n_bounds[1] + 1):
        for j in range(n_bounds[0], n_bounds[1] + 1):
            n = float(i / j)
            ds[i, j] = -kB * np.log(n)
    return dh, ds

class LookupTables:
    def __init__(self, spectrum, pigments, rcs):
        self.gamma = self.gamma_calc(spectrum, pigments)
        self.h     = self.enthalpy_calc(pigments, rcs)
        self.s     = self.entropy_calc(pigments, rcs)

    @classmethod
    def lineshapes(self, spectrum, peak_bounds, increment):
        peak = peak_bounds[0]
        while peak <= peak_bounds[1]:
            l = absorption(spectrum[:, 0], ["avg"], peak)
            peak += increment

    def gamma_calc(self, spectrum, pigments):
        lines = np.zeros(len(pigments))
        gamma = np.zeros(len(pigments))
        for i in range(len(pigments)):
            lines[i] = absorption(spectrum[:, 0], pigments[i], 0.)
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

def absorption(l, pigment, shift):
    '''
    return lineshape of pigment shifted by lp
    NB: we multiply by shift_inc outside this function: really it would make
    more sense to change this and just use the Genome and an index, i think
    '''
    params = constants.pigment_data[pigment]
    lp = [x + shift for x in params['lp']]
    g = gauss(l, lp, params['w'], params['amp'])
    return g

def emission(l, pigment, shift):
    '''
    return emission lineshape for these parameters
    NB: we multiply by shift_inc outside this function: really it would make
    more sense to change this and just use the Genome and an index, i think
    '''
    params = constants.pigment_data[pigment]
    lp = [x + shift for x in params['lp']]
    # reflect through the 0-0 line
    for i in range(1, len(lp)):
        lp[i] += 2.0 * (lp[0] - lp[i])
    g = gauss(l, lp, params['w'], params['amp'])
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
    # 0 offset for the RC! shifts stored as integer increments, so
    # multiply by shift_inc here
    shift = np.array([0., *p.shift], dtype=np.float64) * constants.shift_inc
    pigment = np.array([*p.rc, *p.pigment], dtype='U10')
    a_l = np.zeros((len(pigment), len(l)))
    norms = np.zeros(len(pigment))
    e_l = np.zeros_like(a_l)
    gamma = np.zeros(p.n_s, dtype=np.float64)
    k_b = np.zeros(2 * p.n_s, dtype=np.float64)
    for i in range(p.n_s + 1):
        # NB: come back to this if putting shifts back in!!
        a_l[i] = absorption(l, pigment[i], shift[i])
        e_l[i] = emission(l, pigment[i], shift[i])
        norms[i] = overlap(l, a_l[i], e_l[i])
        if i > 0:
            gamma[i - 1] = (n_p[i] * constants.sig_chl *
                            overlap(l, fp_y, a_l[i]))

    for i in range(p.n_s):
        # here's where it gets interesting. the RC is at index 0, so
        # increasing index means moving outwards; hence, a_l[i], e_l[i + 1]
        # is emission from the outer block and absorption by the inner block
        # then we normalise by requiring that the overlap of the absorbing
        # block *with an identical emitting block* should be 1.
        inward = overlap(l, a_l[i], e_l[i + 1]) / norms[i]
        outward = overlap(l, e_l[i], a_l[i + 1]) / norms[i + 1]
        print(f"pigment pair: {pigment[i]}, {pigment[i + 1]}. inward rate: {inward}, outward rate: {outward}")
        n = float(n_p[i]) / float(n_p[i + 1])
        dg = dG(peak(shift[i], pigment[i]),
                peak(shift[i + 1], pigment[i + 1]), n, constants.T)
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
    twa[1][0] = constants.k_out # 1e+2
    twa[2][0] = constants.k_diss # 2.5e+8
    twa[2][1] = constants.k_trap # 2e+11
    twa[3][1] = constants.k_diss
    twa[3][2] = constants.k_out
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
            twa[ind + 1][ind] = constants.k_out
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
    nu_e = constants.k_out * n_eq[0]
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
    nu_e_low = constants.k_out * n_eq_low[0]
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
                'a_l': a_l,
                'e_l': e_l,
                'norms': norms,
                'k_b': k_b,
                }
    else:
        return np.array([nu_e, phi_e_g, phi_e])



if __name__ == '__main__':

    print("antenna.py test:")
    print("----------------")
    # spectrum, output_prefix = light.spectrum_setup("marine", depth=10.0)
    fraction = 0.6
    spectrum, output_prefix = light.spectrum_setup("far-red",
                                                   fraction=fraction)
    print(f"Input spectrum: {output_prefix}")
    outdir = os.path.join("out", "tests")
    os.makedirs(outdir, exist_ok=True)
    plot_prefix = os.path.join(outdir, "antenna_fr_chl_f")
    cost = 0.01
    n_b = 2
    pigment = ['chl_f']
    n_s = len(pigment)
    n_p = [60]
    no_shift = [0.0 for _ in range(n_s)]
    rc = ["fr_rc"]
    names = rc + pigment
    # extra RC parameters are at the end of the Genome constructor
    # so ignore them. think that's fine to do. not used here anyway
    p = constants.Genome(n_b, n_s, n_p, no_shift,
            pigment, rc)
    print(f"Genome: {p}")

    od = antenna(spectrum[:, 0], spectrum[:, 1], p, True)
    fit = od['nu_e'] - (cost * p.n_b * np.sum(p.n_p))

    outfile = f"{plot_prefix}_{fraction}.info.txt"
    with open(outfile, "w") as f:
        f.write(f"Genome: {p}\n")
        f.write(f"Branch rates k_b: {od['k_b']}\n")
        f.write(f"Raw overlaps of F'(p) A(p): {od['norms']}\n")
        f.write(f"nu_e, phi_e, phi_e_g: {od['nu_e']}, {od['phi_e']}, {od['phi_e_g']}\n")
        f.write(f"Fitness for cost = {cost}: {fit}")
    print(f"Branch rates k_b: {od['k_b']}")
    print(f"Raw overlaps of F'(p) A(p): {od['norms']}")
    print(f"nu_e, phi_e, phi_e_g: {od['nu_e']}, {od['phi_e']}, {od['phi_e_g']}")
    print(f"Fitness for cost = {cost}: {fit}")

    fig, ax = plt.subplots(nrows=len(names), figsize=(12,12), sharex=True)
    for i in range(len(names)):
        ax[i].plot(spectrum[:, 0], od['a_l'][i],
                color='C1', label=f"A ({names[i]})")
        ax[i].plot(spectrum[:, 0], od['e_l'][i],
                color='C0', label=f"F ({names[i]})")
        ax[i].legend()
        ax[i].grid(visible=True)

    fig.supylabel("intensity (arb)", x=0.001)
    ax[0].set_xlim([400., 800.])
    ax[-1].set_xlabel("wavelength (nm)")
    fig.savefig(f"{plot_prefix}_lineshapes.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(nrows=len(od['k_b']),
            figsize=(12,3*len(od['k_b'])), sharex=True)
    for i in range(p.n_s):
        ax[2 * i].plot(spectrum[:, 0], od['a_l'][i],
                color='C1', label=f"A ({names[i]})")
        ax[2 * i].plot(spectrum[:, 0], od['e_l'][i + 1],
                color='C0', label=f"F ({names[i + 1]})")
        ax[2 * i + 1].plot(spectrum[:, 0], od['a_l'][i + 1],
                color='C1', label=f"A ({names[i + 1]})")
        ax[2 * i + 1].plot(spectrum[:, 0], od['e_l'][i],
                color='C0', label=f"F ({names[i]})")
        ax[2 * i].legend(fontsize=26)
        ax[2 * i + 1].legend(fontsize=26)
        ax[2 * i].grid(visible=True)
        ax[2 * i + 1].grid(visible=True)

    fig.supylabel("intensity (arb)", x=0.001)
    ax[0].set_xlim([400., 800.])
    ax[-1].set_xlabel("wavelength (nm)")
    fig.savefig(f"{plot_prefix}_overlaps.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(spectrum[:, 0], spectrum[:, 1],
             color=plots.incident_cols["far-red"],
             alpha=fraction)
    ax.set_xlim([400., 800.])
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("intensity (arb)")
    ax.grid(visible=True)
    fig.savefig(f"{plot_prefix}_{output_prefix}.pdf")
    plt.close(fig)

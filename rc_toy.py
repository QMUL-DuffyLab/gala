import numpy as np
import ctypes
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from matplotlib import cm, ticker, colors
import constants
import antenna as la
import light
from scipy.constants import Boltzmann as kB

'''
look how clever i am
'''
def rc_only(rc_params, debug=True):
    one_rc = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
              (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    # combine one_rc with itself to make the possible states of the supersystem
    two_rc = [s1 + s2 for s1 in one_rc for s2 in one_rc]
    # assign each combination to an index for use in the array below
    indices = {state: i for state, i in zip(two_rc, range(len(two_rc)))}
    # assign the array
    side = len(two_rc)
    twa = np.zeros((side, side))
    two_rc = np.array(two_rc)

    lindices = []
    cycdices = []
    for si in two_rc:
        if si[4] == 1 and si[5] == 0:
            # [n_^{ox}, n^{E}_e, 1, 0]
            cycdices.append(indices[tuple(si)])
        if si[1] == 1 and si[2] == 0 and si[4] == 0 and si[5] == 1:
            # [n_^{ox}_e, 1, 0, n^{E}_e, 0, 1]
            lindices.append(indices[tuple(si)])

    # keys in the dict here are the population differences between
    # initial and final states, for each type of process; values are rates
    processes = {
            (1, 0, 0, 0, 0, 0):   rc_params["gamma_ox"],
            (0, 0, 0, 1, 0, 0):   rc_params["gamma_E"],
            (-1, 0, 0, 0, 0, 0):  rc_params["k_diss"],
            (0, 0, 0, -1, 0, 0):  rc_params["k_diss"],
            (-1, 1, 0, 0, 0, 0):  rc_params["k_trap"],
            (0, 0, 0, -1, 1, 0):  rc_params["k_trap"],
            (0, 0, -1, 0, 0, 0):  rc_params["k_o2"],
            (0, -1, 1, 0, 0, -1): rc_params["k_lin"],
            (0, 0, 0, 0, -1, 1):  rc_params["k_out"],
            (0, 0, 0, 0, -1, 0):  rc_params["k_cyc"],
            }
    # loop over, check difference, assign rate if necessary. bish bash bosh
    # also add the relevant indices for linear and cyclic flow to lists
    for si in two_rc:
        for sf in two_rc:
            diff = tuple(sf - si)
            if diff in processes:
                index = indices[tuple(si)], indices[tuple(sf)]
                twa[index] = processes[diff]
    # set up nnls
    # in theory you should be able to construct this matrix as you go
    # but in practice i always fuck it up somehow. so just do it here
    k = np.zeros((side + 1, side))
    for i in range(side):
        for j in range(side):
            if (i != j):
                k[i][j]  = twa[j][i]
                k[i][i] -= twa[i][j]
        k[side][i] = 1.0

    b = np.zeros(side + 1)
    b[-1] = 1.0

    nu_lin = 0.0
    nu_cyc = 0.0
    try:
        p_eq, p_eq_res = nnls(k, b)
        for state, p in zip(two_rc, p_eq):
            if indices[tuple(state)] in lindices:
                nu_lin += k_lin * p
            if indices[tuple(state)] in cycdices:
                nu_cyc += k_cyc * p
    except RuntimeError:
        p_eq = None
        p_eq_res = None
        nu_cyc = np.nan
        nu_lin = np.nan
    if debug:
        return {k: k,
                p_eq: p_eq,
                p_eq_res: p_eq_res,
                nu_lin: nu_lin,
                nu_cyc: nu_cyc,
                }
    else:
        return nu_cyc / nu_lin

def antenna_rc(l, ip_y, p, debug=False):
    '''
    branched antenna, saturating RC
    l = set of wavelengths
    ip_y = irradiances at those wavelengths
    p = instance of constants.Genome
    set debug = True to output a dict with a load of info in it
    '''
    # NB: need to implement alpha in the GA
    k_cyc = p.alpha * constants.k_lin
    
    # NB: need to add both RCs, ox and then E, at the start!
    # then k_b will match below
    fp_y = (ip_y * l) / la.hcnm
    # assumes constant (and equal) number of pigments in RCs
    n_p = np.array([constants.np_rc, constants.np_rc, *p.n_p],
            dtype=np.int32)
    # 0 shift for RCs. shifts stored as integer increments, so
    # multiply by shift_inc here
    shift = np.array([0., 0., *p.shift], dtype=np.float64) * constants.shift_inc
    pigment = np.array([*p.rc, *p.pigment], dtype='U10')
    lines = np.zeros((p.n_s + 2, len(l)))
    gamma = np.zeros(p.n_s + 2, dtype=np.float64)
    k_b = np.zeros(2 * (p.n_s + 2), dtype=np.float64)
    for i in range(p.n_s + 2):
        lines[i] = la.get_lineshape(l, pigment[i], shift[i])
        gamma[i] = (n_p[i] * constants.sig_chl *
                        la.overlap(l, fp_y, lines[i]))

    for i in range(p.n_s + 2):
        if i < 2:
            # RCs - overlap/dG with first subunit (3rd in list, so [2])
            de = la.overlap(l, lines[i], lines[2])
            n = float(n_p[i]) / float(n_p[2])
            dg = la.dG(la.peak(shift[i], pigment[i]),
                    la.peak(shift[2], pigment[2]), n, constants.T)
        elif i < p.n_s + 1:
            # one subunit and the next
            de = la.overlap(l, lines[i], lines[i + 1])
            n = float(n_p[i]) / float(n_p[i + 1])
            dg = la.dG(la.peak(shift[i], pigment[i]),
                    la.peak(shift[i + 1], pigment[i + 1]), n, constants.T)
        rate = constants.k_hop * de
        k_b[2 * i] = rate
        k_b[(2 * i) + 1] = rate
        if dg < 0.0:
            k_b[(2 * i) + 1] *= np.exp(dg / (constants.T * kB))
        elif dg > 0.0:
            k_b[2 * i] *= np.exp(-1.0 * dg / (constants.T * kB))

    '''
    RC stuff: get the indices etc
    '''
    one_rc = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
              (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    # combine one_rc with itself to make the possible states of the supersystem
    two_rc = [s1 + s2 for s1 in one_rc for s2 in one_rc]
    n_rc = len(two_rc)
    # assign each combination to an index for use in the array below
    indices = {state: i for state, i in zip(two_rc, range(n_rc))}
    two_rc = np.array(two_rc)
    # keys in the dict here are the population differences between
    # initial and final states, for each type of process; values are rates
    processes = {
            (1, 0, 0, 0, 0, 0): gamma[0], # rc_ox is first in lists
            (0, 0, 0, 1, 0, 0): gamma[1], # rc_E second
            (-1, 0, 0, 0, 0, 0):  constants.k_diss,
            (0, 0, 0, -1, 0, 0):  constants.k_diss,
            (-1, 1, 0, 0, 0, 0):  constants.k_trap,
            (0, 0, 0, -1, 1, 0):  constants.k_trap,
            (0, 0, -1, 0, 0, 0):  constants.k_o2,
            (0, -1, 1, 0, 0, -1): constants.k_lin,
            (0, 0, 0, 0, -1, 1):  constants.k_out,
            (0, 0, 0, 0, -1, 0):  k_cyc,
            }

    side = n_rc * ((p.n_b * p.n_s) + 1)
    twa = np.zeros((side, side), dtype=np.longdouble)

    lindices = []
    cycdices = []
    js = list(range(0, side, n_rc))
    for jind, j in enumerate(js):
        # jind == 0 is empty antenna, etc
        for i in range(n_rc):
            # NB: can just generate the RC matrix once and then do
            # for m in range(n_rc):
            #     indf = m + j
            #     twa[ind][indf] = t_rc[i][m]
            ind = i + j # total index
            rc_state = two_rc[i] # tuple with current RC state
            # RC rate stuff first
            if rc_state[4] == 1 and rc_state[5] == 0:
                # [n_^{ox}, n^{E}_e, 1, 0]
                cycdices.append(ind)
            if (rc_state[1] == 1 and rc_state[2] == 0
            and rc_state[4] == 0 and rc_state[5] == 1):
                # [n_^{ox}_e, 1, 0, n^{E}_e, 0, 1]
                lindices.append(ind)
            for sf in two_rc:
                diff = tuple(sf - rc_state)
                if diff in processes:
                    indf = indices[tuple(sf)] + j
                    twa[ind][indf] = processes[diff]

            # antenna rate stuff
            if jind > 0: # population in antenna subunit
                twa[ind][i] = constants.k_diss # dissipation from antenna
                si = ((j // n_rc) - 1) % p.n_s 
                twa[i][ind] = gamma[si + 2] # absorption by this block
                if si == 0:
                    # root of branch - antenna <-> RC transfer possible
                    if rc_state[0] == 1: # ox -> antenna possible
                        # index of final RC state
                        rcf = indices[(0, *rc_state[1:])]
                        twa[rcf][ind] = k_b[0] # backtransfer from e^ox
                    if rc_state[0] == 0: # antenna -> ox possible
                        rcf = indices[(1, *rc_state[1:])]
                        twa[ind][rcf] = k_b[1] # transfer to e^ox
                    if rc_state[3] == 1: # E -> antenna possible
                        rcf = indices[(*rc_state[0:3], 0, *rc_state[4:])]
                        twa[rcf][ind] = k_b[2] # backtransfer from e^E
                    if rc_state[3] == 0: # antenna -> E possible
                        rcf = indices[(*rc_state[0:3], 1, *rc_state[4:])]
                        twa[ind][rcf] = k_b[3] # transfer to e^E
                if p.connected:
                    # need to think about this logic - indexing correct?
                    prevind = ind - (p.n_s * n_rc)
                    nextind = ind + (p.n_s * n_rc)
                    '''
                    first n_rc are for empty antenna. if we're
                    crossing the "boundary" (first <-> last)
                    we need to take this into account
                    '''
                    branch_number = jind // p.n_s
                    if branch_number == 0: # first branch
                        prevind -= n_rc
                    if branch_number == (p.n_b - 1): # final branch
                        nextind += n_rc
                    # PBCs, essentially
                    if prevind < 0:
                        prevind += side
                    if nextind >= side:
                        nextind -= side
                    '''
                    4 possible transfers to consider:
                    - both forward and backward transfer,
                    - from both the clockwise and anticlockwise neighbour,
                    adjacent blocks are identical so no need to calculate dG
                    '''
                    twa[ind][nextind] = constants.k_hop
                    twa[nextind][ind] = constants.k_hop
                    twa[ind][prevind] = constants.k_hop
                    twa[prevind][ind] = constants.k_hop

                if si > 0:
                    twa[ind][ind - n_rc]     = k_b[(2 * si) + 3]
                if si < (p.n_s - 1):
                    twa[ind][ind + n_rc]     = k_b[2 * (si + 2)]


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
    p_eq, p_eq_res = la.solve(k, method='scipy')

    # need to fix this loop for branches and subunits etc
    nu_lin = 0.0
    nu_cyc = 0.0
    for i, p in enumerate(p_eq):
        if i in lindices:
            nu_lin += constants.k_lin * p
        if i in cycdices:
            nu_cyc += k_cyc * p

    if debug:
        return {
                "k": k,
                "twa": twa,
                "gamma": gamma,
                "kb": k_b,
                "p_eq": p_eq,
                "lindices": lindices,
                "cycdices": cycdices,
                "nu_lin": nu_lin,
                "nu_cyc": nu_cyc,
                }
    else:
        return nu_cyc / nu_lin

if __name__ == "__main__":

    spectrum, output_prefix = light.spectrum_setup("red")
    n_b = 2
    n_s = 3
    n_p = [70, 60, 50]
    shift_good = [10.0, 0.0, -10.0]
    shift_bad = [100.0, 90.0, 200.0]
    pigment = ['averaged', 'averaged', 'averaged']
    rc = ["rc_ox", "rc_E"]
    alpha = 1.0
    p = constants.Genome(n_b, n_s, n_p, shift_good, pigment, rc, alpha)

    od = antenna_rc(spectrum[:, 0], spectrum[:, 1], p, True)
    print(od)
    print(len(od["cycdices"]) == 12 * ((n_b * n_s) + 1))
    print(len(od["lindices"]) == 4 * ((n_b * n_s) + 1))
    print(np.sum(od["gamma"]))

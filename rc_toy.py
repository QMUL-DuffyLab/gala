import numpy as np
import ctypes
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from matplotlib import cm, ticker, colors
import constants
import antenna as la
import light
from scipy.constants import Boltzmann as kB

def antenna_rc(l, ip_y, p, debug=False, nnls='scipy'):
    '''
    combined antenna-RC model.
    NB: this is probably not going to work as-is for a non-oxygenic system,
    yet. need to think about how to implement them generally. but it works
    for rc_ox and rc_E in the oxygenic case.

    l = set of wavelengths
    ip_y = irradiances at those wavelengths
    p = instance of constants.Genome
    set debug = True to output a dict with a load of info in it
    nnls = which NNLS version to use
    '''
    k_cyc = p.alpha * constants.k_lin
    
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
    '''
    NB: this can be modified for different RC types by something like
    if len(rcs) == 1:
        final_rc = one_rc
    elif len(rcs) == 2:
        final_rc = [s1 + s2 for s1 in one_rc for s2 in one_rc]
    else:
        print("TOO MANY RCS!!")
    i think that should be fine, there will never be more than two
    '''
    two_rc = [s1 + s2 for s1 in one_rc for s2 in one_rc]
    n_rc = len(two_rc)
    # assign each combination to an index for use in the array below
    indices = {state: i for state, i in zip(two_rc, range(n_rc))}
    two_rc = np.array(two_rc)
    # keys in the dict here are the population differences between
    # initial and final states, for each type of process; values are rates
    processes = {
            # NB: these four processes could also be transfer to and
            # from the antenna, but we only use this dict for the RC-RC
            # terms, and the transfer's taken care of afterwards, so
            # this should be fine
            (1, 0, 0, 0, 0, 0): gamma[0], # rc_ox is first in lists
            (0, 0, 0, 1, 0, 0): gamma[1], # rc_E second
            (-1, 0, 0, 0, 0, 0):  constants.k_diss,
            (0, 0, 0, -1, 0, 0):  constants.k_diss,
            (-1, 1, 0, 0, 0, 0):  constants.k_trap,
            (0, 0, 0, -1, 1, 0):  constants.k_trap,
            (0, 0, -1, 0, 0, 0):  constants.k_o2,
            (0, -1, 1, 0, 0, -1): p.eta * constants.k_lin,
            (0, 0, 0, 0, -1, 1):  constants.k_out,
            (0, 0, 0, 0, -1, 0):  k_cyc,
            }

    side = n_rc * ((p.n_b * p.n_s) + 1)
    twa = np.zeros((side, side), dtype=np.longdouble)

    lindices = []
    cycdices = []
    js = list(range(0, side, n_rc))
    for jind, j in enumerate(js):
        # jind == 0 is empty antenna, 0 + n_rc is first antenna block, etc
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
            # now loop over the states again, get the population change,
            # and insert the correct rate if it matches a process
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
                    # need to calculate the index of the final RC state:
                    # antenna-RC transfer changes the state of both
                    if rc_state[0] == 1: # ox -> antenna possible
                        # index of final RC state
                        rcf = indices[(0, *rc_state[1:])]
                        # backtransfer from e^ox
                        twa[rcf][ind] = (p.eta / p.phi) * k_b[0]
                    if rc_state[0] == 0: # antenna -> ox possible
                        rcf = indices[(1, *rc_state[1:])]
                        # transfer to e^ox
                        twa[ind][rcf] = (p.phi / p.eta) * k_b[1]
                    if rc_state[3] == 1: # E -> antenna possible
                        rcf = indices[(*rc_state[0:3], 0, *rc_state[4:])]
                        # backtransfer from e^E
                        twa[rcf][ind] = (1.0 / p.phi) * k_b[2]
                    if rc_state[3] == 0: # antenna -> E possible
                        rcf = indices[(*rc_state[0:3], 1, *rc_state[4:])]
                        # transfer to e^E
                        twa[ind][rcf] = p.phi * k_b[3]
                if p.connected:
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
    p_eq, p_eq_res = la.solve(k, method=nnls)

    # need to fix this loop for branches and subunits etc
    nu_lin = 0.0
    nu_cyc = 0.0
    for i, p_i in enumerate(p_eq):
        if i in lindices:
            nu_lin += p.eta * constants.k_lin * p_i
        if i in cycdices:
            nu_cyc += k_cyc * p_i

    if debug:
        # generate states to go with p_eq indices
        ast = []
        empty = tuple([0 for _ in range(p.n_b * p.n_s)])
        ast.append(empty)
        for i in range(p.n_b * p.n_s):
            el = [0 for _ in range(p.n_b * p.n_s)]
            el[i] = 1
            ast.append(tuple(el))
        total_states = [s1 + tuple(s2) for s1 in ast for s2 in two_rc]


    w_e = nu_lin + nu_cyc
    w_red = w_e / (1.0 + (p.alpha * constants.k_lin / constants.k_out))
    if debug:
        return {
                "k": k,
                "twa": twa,
                "gamma": gamma,
                "kb": k_b,
                "p_eq": p_eq,
                "states": total_states,
                "lindices": lindices,
                "cycdices": cycdices,
                "nu_lin": nu_lin,
                "nu_cyc": nu_cyc,
                "w_e": w_e,
                "w_red": w_red,
                }
    else:
        return nu_cyc / nu_lin

if __name__ == "__main__":

    spectrum, output_prefix = light.spectrum_setup("marine", depth=10.0)
    n_b = 1
    n_s = 1
    n_p = [10 for _ in range(n_s)]
    no_shift = [0.0 for _ in range(n_s)]
    pigment = ['apc']
    rc = ["rc_ox", "rc_E"]
    alpha = 0.1
    # phi seems to have no effect - why
    phi = 0.1
    eta = 2.0
    p = constants.Genome(n_b, n_s, n_p, no_shift,
            pigment, rc, alpha, phi, eta)

    od = antenna_rc(spectrum[:, 0], spectrum[:, 1], p, True)
    # print(od)
    print(f"alpha = {alpha}, phi = {phi}, eta = {eta}")
    print(f"p(0) = {od['p_eq'][0]}")
    print(f"w_e = {od['w_e']}")
    print(f"w_red = {od['w_red']}")
    print(f"sum(gamma) = {np.sum(od['gamma'])}")
    for si, pi in zip(od["states"], od["p_eq"]):
        print(f"p_eq{si} = {pi}")

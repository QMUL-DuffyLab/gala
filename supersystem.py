import ctypes
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from matplotlib import cm, ticker, colors
from scipy.constants import Boltzmann as kB
import constants
import light
import antenna as la
import rc

def supersystem(l, ip_y, p, debug=False, nnls='scipy'):
    '''
    generate matrix for combined antenna-RC supersystem and solve it.

    parameters
    ----------
    l = set of wavelengths
    ip_y = irradiances at those wavelengths
    p = instance of constants.Genome
    set debug = True to output a dict with a load of info in it
    nnls = which NNLS version to use

    outputs
    -------
    if debug == True:
    debug: a huge dict containing various parameters that are useful to me
    (and probably only me) in figuring out what the fuck is going on.
    else:
    TBC. probably nu_ch2o and nu_cyc
    '''
    k_cyc = p.alpha * constants.k_lin
    
    fp_y = (ip_y * l) / la.hcnm
    rcp = rc.params[p.rc]
    n_rc = len(rcp["pigments"])
    rc_n_p = [constants.pigment_data[rc]["n_p"] for rc in rcp["pigments"]]
    n_p = np.array([*rc_n_p, *p.n_p], dtype=np.int32)
    # 0 shift for RCs. shifts stored as integer increments, so
    # multiply by shift_inc here
    shift = np.array([0., 0., *p.shift], dtype=np.float64) * constants.shift_inc
    pigment = np.array([*rcp["pigments"], *p.pigment], dtype='U10')
    a_l = np.zeros((p.n_s + n_rc, len(l)))
    e_l = np.zeros_like(a_l)
    norms = np.zeros(len(pigment))
    gamma = np.zeros(p.n_s + n_rc, dtype=np.float64)
    k_b = np.zeros(2 * (p.n_s + n_rc), dtype=np.float64)
    for i in range(p.n_s + n_rc):
        a_l[i] = la.absorption(l, pigment[i], shift[i])
        e_l[i] = la.emission(l, pigment[i], shift[i])
        norms[i] = la.overlap(l, a_l[i], e_l[i])
        gamma[i] = (n_p[i] * constants.sig_chl *
                        la.overlap(l, fp_y, a_l[i]))

    # NB: this needs checking for logic for all types
    for i in range(p.n_s + n_rc):
        if i < n_rc:
            # RCs - overlap/dG with 1st subunit (n_rc + 1 in list, so [n_rc])
            inward  = la.overlap(l, a_l[i], e_l[n_rc]) / norms[i]
            outward = la.overlap(l, e_l[i], a_l[n_rc]) / norms[n_rc]
            n = float(n_p[i]) / float(n_p[n_rc])
            dg = la.dG(la.peak(shift[i], pigment[i]),
                    la.peak(shift[2], pigment[2]), n, constants.T)
        elif i >= n_rc and i < p.n_s + 1:
            # one subunit and the next
            inward  = la.overlap(l, a_l[i], e_l[i + 1]) / norms[i]
            outward = la.overlap(l, e_l[i], a_l[i + 1]) / norms[i + 1]
            n = float(n_p[i]) / float(n_p[i + 1])
            dg = la.dG(la.peak(shift[i], pigment[i]),
                    la.peak(shift[i + 1], pigment[i + 1]), n, constants.T)
        print(inward, outward)
        k_b[2 * i] = constants.k_hop * outward
        k_b[(2 * i) + 1] = constants.k_hop * inward
        if dg < 0.0:
            k_b[(2 * i) + 1] *= np.exp(dg / (constants.T * kB))
        elif dg > 0.0:
            k_b[2 * i] *= np.exp(-1.0 * dg / (constants.T * kB))

    n_rc_states = len(rcp["states"]) # total number of states of all RCs
    side = n_rc_states * ((p.n_b * p.n_s) + n_rc + 1)
    twa = np.zeros((side, side), dtype=np.longdouble)

    # generate states to go with p_eq indices
    ast = []
    empty = tuple([0 for _ in range(n_rc + p.n_b * p.n_s)])
    ast.append(empty)
    for i in range(n_rc + p.n_b * p.n_s):
        el = [0 for _ in range(n_rc + p.n_b * p.n_s)]
        el[i] = 1
        ast.append(tuple(el))
    total_states = [s1 + tuple(s2) for s1 in ast for s2 in rcp["states"]]
    total_indices = {i: total_states[i] for i in range(len(total_states))}

    lindices = rcp["nu_ch2o_ind"]
    cycdices = rcp["nu_cyc_ind"]
    js = list(range(0, side, n_rc_states))
    for jind, j in enumerate(js):
        # jind == 0 is empty antenna, 0 + n_rc_states is RC 1 occupied, etc
        for i in range(n_rc_states):
            ind = i + j # total index
            initial = rcp["states"][i] # tuple with current RC state
            for k in range(n_rc_states):
                final = rcp["states"][k]
                diff = tuple(final - initial)
                if diff in rcp["procs"]:
                    # NB: this won't work yet. need to figure detrapping/cyc
                    # get the type of process
                    rt = rcp["procs"][diff]
                    indf = indices[tuple(final)] + j
                    twa[ind][indf] = rc.rates[rt]
                    if rt == "cyc":
                        # this is both detrapping and cyclic
                        # cyclic: multiply the rate by alpha etc.
                        # alpha should no longer be a genome param!
                        twa[ind][indf] *= p.alpha * np.sum(n_p)
                        # detrapping:
                        # - only possible if exciton manifold is empty
                        # - excitation must go back to the correct photosystem
                        if jind == 0:
                            which_rc = np.where(np.array(diff) == -1)[0][0]//2
                            indf = (indices[tuple(final)] + j + 
                                    (which_rc * n_rc_states))
                            detrap = rc.rates["trap"] * np.exp(-rcp["gap"])
                            twa[ind][indf] = detrap
                        

            '''
            NB: (29/01/2025) I think it's mostly fixed up to here.
            Below it's probably still wrong because the excited state of the
            RC is still counted in all this whereas now it should be outside
            the rc_states loop. need to go through all this and figure it out.
            will probably need to move some stuff (gauss, overlap calc etc.)
            into a separate lineshapes.py file as well.
            '''
            twa[ind][i] = constants.k_diss # dissipation from antenna
            si = ((j // n_rc_states) - 1) % p.n_s
            twa[i][ind] = gamma[si + 2] # absorption by this block
            if si == 0:
                # root of branch - antenna <-> RC transfer possible
                # need to calculate the index of the final RC state:
                # antenna-RC transfer changes the state of both
                state = total_states[ind]
                ans = ind // n_rc_states # antenna block index
                rcs = ind % n_rc_states # rc index within block
                ti = (j // n_rc) - 1
                if rc_state[0] == 0 and state[ti] == 1: # antenna -> ox possible
                    rcf = indices[(1, *rc_state[1:])]
                    print(f"A -> OX. {j} {ind} {total_states[ind]} -> {rcf} {total_states[rcf]}: {k_b[1]:.2e}")
                    print(f"OX -> A. {j} {rcf} {total_states[rcf]} -> {ind} {total_states[ind]}: {k_b[0]:.2e}")
                    # antenna <--> e^{ox}
                    twa[ind][rcf] = (p.phi / p.eta) * k_b[1]
                    twa[rcf][ind] = (p.eta / p.phi) * k_b[0]
                if rc_state[3] == 0 and state[ti] == 1: # antenna -> E possible
                    rcf = indices[(*rc_state[0:3], 1, *rc_state[4:])]
                    # antenna <--> e^E
                    print(f"A -> E.  {j} {ind} {total_states[ind]} -> {rcf} {total_states[rcf]}: {k_b[3]:.2e}")
                    print(f"E -> A.  {j} {rcf} {total_states[rcf]} -> {ind} {total_states[ind]}: {k_b[2]:.2e}")
                    twa[ind][rcf] = p.phi * k_b[3]
                    twa[rcf][ind] = (1.0 / p.phi) * k_b[2]

            # antenna rate stuff
            if jind > 0: # population in antenna subunit
                if p.connected:
                    prevind = ind - (p.n_s * n_rc_states)
                    nextind = ind + (p.n_s * n_rc_states)
                    '''
                    first n_rc are for empty antenna. if we're
                    crossing the "boundary" (first <-> last)
                    we need to take this into account
                    '''
                    branch_number = jind // p.n_s
                    if branch_number == 0: # first branch
                        prevind -= n_rc_states
                    if branch_number == (p.n_b - 1): # final branch
                        nextind += n_rc_states
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
                    twa[ind][ind - n_rc_states] = k_b[(2 * si) + 3]
                if si < (p.n_s - 1):
                    twa[ind][ind + n_rc_states] = k_b[2 * (si + 2)]


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

    w_e = nu_lin + nu_cyc
    w_red = w_e / (1.0 + (p.alpha * constants.k_lin / constants.k_red))
    if debug:
        return {
                "k": k,
                "twa": twa,
                "gamma": gamma,
                "k_b": k_b,
                "p_eq": p_eq,
                "states": total_states,
                "lindices": lindices,
                "cycdices": cycdices,
                "nu_lin": nu_lin,
                "nu_cyc": nu_cyc,
                "w_e": w_e,
                "w_red": w_red,
                'a_l': a_l,
                'e_l': e_l,
                'norms': norms,
                'k_b': k_b,
                }
    else:
        return nu_cyc / nu_lin

if __name__ == "__main__":

    # spectrum, output_prefix = light.spectrum_setup("marine", depth=10.0)
    spectrum, output_prefix = light.spectrum_setup("marine", depth=10.0)
    n_b = 5
    pigment = ['apc', 'pc', 'r-pe']
    n_s = len(pigment)
    n_p = [50 for _ in range(n_s)]
    no_shift = [0.0 for _ in range(n_s)]
    rc = ["rc_ox", "rc_E"]
    names = rc + pigment
    alpha = 1.0
    # test effect of phi
    phi = 2.0
    eta = 2.0
    p = constants.Genome(n_b, n_s, n_p, no_shift,
            pigment, rc, alpha, phi, eta)

    od = antenna_rc(spectrum[:, 0], spectrum[:, 1], p, True)
    print(f"Branch rates k_b: {od['k_b']}")
    print(f"Raw overlaps of F'(p) A(p): {od['norms']}")
    # print(f"nu_e, phi_e, phi_e_g: {od['nu_e']}, {od['phi_e']}, {od['phi_e_g']}")

    side = len(od["p_eq"])
    for i in range(side):
        colsum = np.sum(od["k"][:side, i])
        rowsum = np.sum(od["k"][i, :])
        print(f"index {i}: state {od['states'][i]} sum(col[i]) = {colsum}, sum(row[i]) = {rowsum}")
    print(np.sum(od["k"][:side, :]))
    print(f"alpha = {alpha}, phi = {phi}, eta = {eta}")
    print(f"p(0) = {od['p_eq'][0]}")
    print(f"w_e = {od['w_e']}")
    print(f"w_red = {od['w_red']}")
    print(f"sum(gamma) = {np.sum(od['gamma'])}")
    for si, pi in zip(od["states"], od["p_eq"]):
        print(f"p_eq{si} = {pi}")
    print(f"k_b = {od['k_b']}")
    np.savetxt("out/antenna_rc_twa.dat", od["twa"])
    with open("out/antenna_rc_results.dat", "w") as f:
    # print(od)
        f.write(f"alpha = {alpha}, phi = {phi}, eta = {eta}\n")
        f.write(f"p(0) = {od['p_eq'][0]}\n")
        f.write(f"w_e = {od['w_e']}\n")
        f.write(f"w_red = {od['w_red']}\n")
        f.write(f"sum(gamma) = {np.sum(od['gamma'])}\n")
        for si, pi in zip(od["states"], od["p_eq"]):
            f.write(f"p_eq{si} = {pi}\n")
    
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
    fig.savefig("out/rc_antenna_lineshape_test.pdf")
    plt.close(fig)


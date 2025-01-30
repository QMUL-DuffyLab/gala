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
        '''
        the first n_rc pairs of rates are the transfer to and from
        the excited state of the RCs and the antenna. these are
        modified by the stoichiometry of these things. for now this
        is hardcoded but it's definitely possible to fix, especially
        if moving genome parameters into a file and generating from
        that: have a JSON parameter like "array": True/False and then
        "array_len": "n_s" or "rc", which can be used in the GA
        '''
        if n_rc == 1:
            # odd - inward, even - outward
            k_b[1] *= p.phi
            k_b[0] *= (1.0 / p.phi)
        elif n_rc == 2:
            k_b[1] *= (p.phi / p.eta)
            k_b[0] *= (p.eta / p.phi)
            k_b[3] *= p.phi
            k_b[2] *= (1.0 / p.phi)
        elif n_rc == 3:
            k_b[1] *= (p.phi / p.eta)
            k_b[0] *= (p.eta / p.phi)
            k_b[3] *= (p.phi / p.zeta)
            k_b[2] *= (p.zeta / p.phi)
            k_b[5] *= p.phi
            k_b[4] *= (1.0 / p.phi)
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

    lindices = []
    cycdices = []
    js = list(range(0, side, n_rc_states))
    for jind, j in enumerate(js):
        # jind == 0 is empty antenna, 0 + n_rc_states is RC 1 occupied, etc
        # intra-RC processes are the same in each block
        for i in range(n_rc_states):
            ind = i + j # total index
            initial = rcp["states"][i] # tuple with current RC state
            if i in rcp["nu_ch2o_ind"]:
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
                    # set the correct element with the corresponding rate
                    twa[ind][indf] = rc.rates[rt]
                    if rt == "trap":
                        # find which trap state is being filled here
                        which_rc = np.where(np.array(diff) == 1)[0][0]//2
                        '''
                        if jind = which_rc + 1, that means population
                        is coming from the corresponding RC exciton state
                        (since jind = 0 is the empty state, jind = 1 is
                        RC 1, and so on). in that case, the above trap rate
                        is correct. otherwise it would correspond to transfer
                        from a different RC or an antenna block, which is
                        not allowed, so zero it back out
                        '''
                        if jind != which_rc + 1:
                            twa[ind][indf] = 0.0
                    if rt == "cyc":
                        # this is both detrapping and cyclic
                        # cyclic: multiply the rate by alpha etc.
                        # we will need this below for nu(cyc)
                        k_cyc = rc.rates["cyc"]
                        if n_rc == 1:
                            k_cyc *= (1.0 + constants.alpha * np.sum(n_p))
                            twa[ind][indf] = k_cyc
                        else:
                            k_cyc *= constants.alpha * np.sum(n_p)
                            twa[ind][indf] = k_cyc
                        # detrapping:
                        # - only possible if exciton manifold is empty
                        # - excitation must go back to the correct photosystem
                        if jind == 0:
                            which_rc = np.where(np.array(diff) == -1)[0][0]//2
                            indf = (rcp["indices"][tuple(final)] + j + 
                                    (which_rc * n_rc_states))
                            detrap = rc.rates["trap"] * np.exp(-rcp["gap"])
                            twa[ind][indf] = detrap
                        

            '''
            will probably need to move some stuff (gauss, overlap calc etc.)
            into a separate lineshapes.py file as well.

            Also: the indices dict coming from rc.py - which way round?
            above we want (state tuple) -> index, but for the lindices
            and cycdices we want index -> (state tuple), i think. figure that
            out too :)
            '''
            if jind > 0:
                # occupied exciton block -> empty due to dissipation
                # final state index is i because RC state is unaffected
                twa[ind][i] = constants.k_diss
            
            if jind > 0 and jind <= n_rc:
                twa[i][ind] = gamma[jind - 1] # absorption by RCs

            # antenna rate stuff
            if jind > n_rc: # population in antenna subunit
                if p.connected:
                    prevind = ind - (p.n_s * n_rc_states)
                    nextind = ind + (p.n_s * n_rc_states)
                    branch_number = (jind - n_rc - 1) // p.n_s
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
                    two pairs of transfers are possible, between
                    clockwise and anticlockwise neighbour blocks.
                    branches are identical so dG \equiv 0
                    '''
                    twa[ind][nextind] = constants.k_hop
                    twa[nextind][ind] = constants.k_hop
                    twa[ind][prevind] = constants.k_hop
                    twa[prevind][ind] = constants.k_hop

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
                    twa[ind][ind + n_rc_states] = k_b[2 * (n_rc + bi)]


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

    nu_ch2o = 0.0
    nu_cyc = 0.0
    for i, p_i in enumerate(p_eq):
        if i in lindices:
            nu_ch2o += rc.rates["red"] * p_i
        if i in cycdices:
            nu_cyc += k_cyc * p_i

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
                "nu_ch2o": nu_ch2o,
                "nu_cyc": nu_cyc,
                'a_l': a_l,
                'e_l': e_l,
                'norms': norms,
                'k_b': k_b,
                }
    else:
        return nu_ch2o

if __name__ == "__main__":

    spectrum, output_prefix = light.spectrum_setup("marine", depth=2.0)
    n_b = 1
    pigment = ['apc']
    n_s = len(pigment)
    n_p = [50 for _ in range(n_s)]
    no_shift = [0.0 for _ in range(n_s)]
    rc_type = "ox"
    names = rc.params[rc_type]["pigments"] + pigment
    # test effect of phi
    phi = 1.0
    eta = 1.0
    p = constants.Genome(n_b, n_s, n_p, no_shift,
            pigment, rc_type, phi, eta)

    od = supersystem(spectrum[:, 0], spectrum[:, 1], p, True)
    print(f"Branch rates k_b: {od['k_b']}")
    print(f"Raw overlaps of F'(p) A(p): {od['norms']}")

    side = len(od["p_eq"])
    for i in range(side):
        colsum = np.sum(od["k"][:side, i])
        rowsum = np.sum(od["k"][i, :])
        print(f"index {i}: state {od['states'][i]} sum(col[i]) = {colsum}, sum(row[i]) = {rowsum}")
    print(np.sum(od["k"][:side, :]))
    print(f"alpha = {constants.alpha}, phi = {phi}, eta = {eta}")
    print(f"p(0) = {od['p_eq'][0]}")
    print(f"nu_ch2o = {od['nu_ch2o']}")
    print(f"nu_cyc = {od['nu_cyc']}")
    print(f"sum(gamma) = {np.sum(od['gamma'])}")
    for si, pi in zip(od["states"], od["p_eq"]):
        print(f"p_eq{si} = {pi}")
    print(f"k_b = {od['k_b']}")
    np.savetxt("out/antenna_rc_twa.dat", od["twa"])
    with open("out/antenna_rc_results.dat", "w") as f:
    # print(od)
        f.write(f"alpha = {constants.alpha}, phi = {phi}, eta = {eta}\n")
        f.write(f"p(0) = {od['p_eq'][0]}\n")
        f.write(f"nu_ch2o = {od['nu_ch2o']}\n")
        f.write(f"nu_cyc = {od['nu_cyc']}\n")
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


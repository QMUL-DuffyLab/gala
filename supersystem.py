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

def solve(l, ip_y, p, debug=False, nnls='fortran',
        detrap_type="none", tau_diff=0.0):
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
    tau_diff = diffusion time for the oxidation substrate (seconds)

    outputs
    -------
    if debug == True:
    debug: a huge dict containing various parameters that are useful to me
    (and probably only me) in figuring out what the fuck is going on.
    else:
    TBC. probably nu_e and nu_cyc
    '''
    # NB: the tau_diff and detrap_type might or might not be
    # simulation-wide parameters later on; if not, if they need to be
    # put into the genome somewhere i suppose
    
    fp_y = (ip_y * l) / la.hcnm
    rcp = rc.params[p.rc]
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
        # print(f"{i}, {pigment[i]}")
        a_l[i] = la.absorption(l, pigment[i], shift[i])
        e_l[i] = la.emission(l, pigment[i], shift[i])
        norms[i] = la.overlap(l, a_l[i], e_l[i])
        gamma[i] = (n_p[i] * constants.sig_chl *
                        la.overlap(l, fp_y, a_l[i]))

    # detrapping regime
    detrap = rc.rates["trap"]
    if detrap_type == "fast":
        pass
    elif detrap_type == "thermal":
        detrap *= np.exp(-1.0) # -k_B T
    elif detrap_type == "energy_gap":
        detrap *= np.exp(-rcp["gap"])
    elif detrap_type == "none":
        detrap *= 0.0 # irreversible
    else:
        raise ValueError("Detrapping regime should be 'fast',"
          " 'thermal', 'energy_gap' or 'none'.")

    # NB: this needs checking for logic for all types
    # print("KB CALC:")
    for i in range(p.n_s + n_rc):
        ab = i
        el = -1
        if i < n_rc:
            # RCs - overlap/dG with 1st subunit (n_rc + 1 in list, so [n_rc])
            inward  = la.overlap(l, a_l[i], e_l[n_rc]) / norms[i]
            el = n_rc
            outward = la.overlap(l, e_l[i], a_l[n_rc]) / norms[n_rc]
            # print(inward, outward)
            n = float(n_p[i]) / float(n_p[n_rc])
            dg = la.dG(la.peak(shift[i], pigment[i]),
                    la.peak(shift[n_rc], pigment[n_rc]), n, constants.T)
            # print("DG:")
            # print(f"peak 1, {pigment[i]}, {la.peak(shift[i], pigment[i])}")
            # print(f"peak 1, {pigment[n_rc]}, {la.peak(shift[n_rc], pigment[n_rc])}")
            # print(f"{n_p[i]}, {n_p[n_rc]}, {constants.T}, {dg}")
        elif i >= n_rc and i < (p.n_s + n_rc - 1):
            # one subunit and the next
            el = i + 1
            inward  = la.overlap(l, a_l[i], e_l[i + 1]) / norms[i]
            outward = la.overlap(l, e_l[i], a_l[i + 1]) / norms[i + 1]
            # print(inward, outward)
            n = float(n_p[i]) / float(n_p[i + 1])
            dg = la.dG(la.peak(shift[i], pigment[i]),
                    la.peak(shift[i + 1], pigment[i + 1]), n, constants.T)
        # print(f"{i}, pig[{ab}] = {pigment[ab]}, pig[{el}] = {pigment[el]}, {inward}, {outward}")
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
        # rho is [rho_ox, rho_i, rho_r, rho_ant]
        for i in range(n_rc):
            # odd - inward, even - outward
            k_b[2 * i] *= p.aff[i] * ((p.rho[i] * p.rho[-1]) / (n_rc + 1.0))
            k_b[2 * i + 1] *= p.aff[i] * ((p.rho[i] * p.rho[-1]) / (n_rc + 1.0))
        if dg < 0.0:
            k_b[(2 * i) + 1] *= np.exp(dg / (constants.T * kB))
        elif dg > 0.0:
            k_b[2 * i] *= np.exp(-1.0 * dg / (constants.T * kB))
    # print("KB CALC DONE")
    # print(k_b)

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
                    # print(diff)
                    # set the correct element with the corresponding rate
                    if rt == "red":
                        twa[ind][indf] = rc.rates[rt]
                    if rt == "lin":
                        # the first place where the population decreases
                        # is the first in the chain of linear flow
                        which_rc = np.where(np.array(diff) == -1)[0][0]//3
                        twa[ind][indf] = rc.rates[rt] * (p.rho[which_rc]
                                * p.rho[which_rc + 1])
                    if rt == "ox":
                        tau_ox = (tau_diff + 1.0 / rc.rates[rt])
                        twa[ind][indf] = 1.0 / tau_ox
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
                            twa[ind][indf] = rc.rates[rt]
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
                        k_cyc = rc.rates["cyc"]
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
                        twa[ind][indf] += rc.rates["rec"]

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
                # print(f"{toti[ind]} -> {toti[i]}: k_diss")
            
            if jind > 0 and jind <= n_rc:
                twa[i][ind] = gamma[jind - 1] # absorption by RCs
                # print(f"{toti[i]} -> {toti[ind]}: gamma[{jind - 1}]")

            # antenna rate stuff
            if jind > n_rc: # population in antenna subunit
                # index on branch
                bi = (jind - n_rc - 1) % p.n_s
                twa[i][ind] = gamma[n_rc + bi] # absorption by this block
                # print(f"{toti[i]} -> {toti[ind]}: gamma[{n_rc + bi}]")
                if bi == 0:
                    # root of branch - transfer to RC exciton states possible
                    for k in range(n_rc):
                        # transfer to RC 0 is transfer to jind 1
                        offset = (n_rc - k) * n_rc_states
                        # inward transfer to RC k
                        twa[ind][ind - offset] = k_b[2 * k + 1]
                        # print(f"{toti[ind]} -> {toti[ind-offset]}: kb[{2 * k + 1}], ind={ind}, offset={offset}")
                        # outward transfer from RC k
                        twa[ind - offset][ind] = k_b[2 * k]
                        # print(f"{toti[ind-offset]} -> {toti[ind]}: kb[{2 * k}], ind={ind}, offset={offset}")
                if bi > 0:
                    # inward along branch
                    twa[ind][ind - n_rc_states] = k_b[2 * (n_rc + bi) - 1]
                    # print(f"{toti[ind]} -> {toti[ind-n_rc_states]}: kb[{2 * (n_rc + bi)}], bi = {bi}")
                if bi < (p.n_s - 1):
                    # outward allowed
                    twa[ind][ind + n_rc_states] = k_b[2 * (n_rc + bi) + 1]
                    # print(f"{toti[ind]} -> {toti[ind+n_rc_states]}: kb[{2 * (n_rc + bi) + 1}], bi = {bi}")

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
    p_eq, p_eq_res = la.solve(k, method='fortran')

    # nu_e === nu_ch2o here; we treat them as identical
    nu_e = 0.0
    nu_cyc = 0.0
    trap_indices = [-(3 + 3*i) for i in range(n_rc)]
    oxidised_indices = [-(2 + 3*i) for i in range(n_rc)]
    reduced_indices = [-(1 + 3*i) for i in range(n_rc)]
    redox = np.zeros((n_rc, 2), dtype=np.float64)
    recomb = np.zeros(n_rc, dtype=np.float64)
    trap_states = []
    ox_states = []
    red_states = []
    # print(trap_indices)
    # print(oxidised_indices)
    # print(reduced_indices)

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
        return (nu_e, nu_cyc, redox, recomb, -1)

    for i, p_i in enumerate(p_eq):
        s = toti[i]
        for j in range(n_rc):
            if s[trap_indices[j]] == 1:
                recomb += p_i * rc.rates["rec"]
                trap_states.append(s)
            if s[oxidised_indices[j]] == 1:
                redox[j, 0] += p_i
                ox_states.append(s)
            if s[reduced_indices[j]] == 1:
                redox[j, 1] += p_i
                red_states.append(s)
        if i in lindices:
            nu_e += rc.rates["red"] * p_i
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
                "trap_states": trap_states,
                "ox_states": ox_states,
                "red_states": red_states,
                "nu_e": nu_e,
                "nu_cyc": nu_cyc,
                'a_l': a_l,
                'e_l': e_l,
                'norms': norms,
                'k_b': k_b,
                }
    else:
        return (nu_e, nu_cyc, redox, recomb, 0)

if __name__ == "__main__":

    spectrum, output_prefix = light.spectrum_setup("filtered", filter="red")
    print(len(spectrum[:, 0]))
    n_b = 2
    n_s = 1
    pigment = ['chl_a' for _ in range(n_s)]
    n_p = [70 for _ in range(n_s)]
    no_shift = [0 for _ in range(n_s)]
    rc_type = "ox"
    names = rc.params[rc_type]["pigments"] + pigment
    rho = [1.0, 1.0, 1.0] # equal stoichiometry
    aff = [1.0, 1.0] # ps_ox, ps_r affinity
    p = constants.Genome(n_b, n_s, n_p, no_shift,
            pigment, rc_type, rho, aff)

    od = solve(spectrum[:, 0], spectrum[:, 1], p, True)
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

    n_rc = len(rc.params[rc_type]["pigments"])
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


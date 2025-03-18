# -*- coding: utf-8 -*-
"""
27/01/2025
@author: callum

"""
import os
import numpy as np
import argparse
import itertools
import constants
import light
import antenna as la

# these were in constants.py but they're only needed here
# dict because we need to know where cyclic/detrapping are
# and insert both rates in the combined antenna-RC matrix.
rates = {
"trap" : 1.0 / 10.0E-12,
"ox"   : 1.0 / 1.0E-3,
"lin"  : 1.0 / 10.0E-3,
"cyc"  : 1.0 / 10.0E-3,
"red"  : 1.0 / 10.0E-3,
"rec"  : 1.0,
}

def parameters(pigments, gap):
    '''
    for a given set of photosystems, generate a dict of intra-RC
    processes and the lists of indices that pertain to linear and
    cyclic electron flow respectively for use in supersystem.py.
    
    quite proud of this one, actually. it might look a little gross
    but it has to be because the population changes for each process
    are different etc. etc. etc.
    
    parameters
    ----------
    pigments: the list of lineshapes for the RC pigments. really we
    only need this here for its length, but we want as part of the
    dict of RC supersystem parameters anyway.
    gap: the energy gap between states in units of k_B T. again, we
    don't actually need this here, but it does in the dict of
    supersystem parameters.
    
    outputs
    -------
    params: a dict of the RC supersystem parameters. values are:
        - pigments: the list of pigments
        - gap: the energy gap
        - procs: the intra-RC processes, as a dict, with the key being
        the population change associated with the process and the value
        being a string which we will need to use later. the reason it's
        a string is that detrapping and cyclic electron flow are not
        necessarily distinguishable from the point of view of the RC,
        but they have different rates and we need to insert them in 
        different places, so we can't just insert one number.
        - nu_ch2o_ind: indices of the RC states we need to sum over
        to get the rate of carbon fixation $\nu(CH_2O)$
        - nu_cyc_ind: indices of the RC states we need to sum over to
        get the rate of cyclic electron flow $\nu(\text{cyc})$
    '''
    n_rc = len(pigments)
    one_rc = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    for i in range(n_rc):
        n_states = len(one_rc)**n_rc
        states = np.array(list(map(list, itertools.product(one_rc, repeat=n_rc)))).reshape(n_states, n_rc * len(one_rc[0]))
        indices = {tuple(states[j]): j for j in range(n_states)}
    procs = {}
    nu_ch2o_ind = []
    nu_cyc_ind  = []
    for i in range(n_states):
        initial = states[i]
        if initial[-1] == 1 or initial[-3] == 1: # n^r_R, n^r_T
            nu_ch2o_ind.append(i)
        if n_rc == 1:
            if initial[0] == 1:
                nu_cyc_ind.append(i)
        elif n_rc > 1:
            trap_indices = [3 * k for k in range(1, n_rc)]
            if any(initial[trap_indices] == 1):
                nu_cyc_ind.append(i)
        for j in range(n_states):
            final = states[j]
            diff = final - initial
            for k in range(n_rc):
                kt = 3 * k # index of the trap state of PS_k
                if diff[kt] == 1:
                    all_zero = True
                    for m in range(len(diff)):
                        if m != kt and diff[m] != 0:
                            all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "trap"})
                if diff[kt] == -1:
                    all_zero = True
                    for m in range(len(diff)):
                        if m != kt and diff[m] != 0:
                            all_zero = False
                    if all_zero:
                        # this is not just cyclic but also detrapping!
                        # in fact if n_rc > 1 and k = 0, this is only
                        # detrapping, since ps_ox can't do cyclic.
                        # check for "cyc" in supersystem.py and insert both
                        procs.update({tuple(diff): "cyc"})
                if (tuple(diff[0:3]) == (-1, 0, 1) or
                    tuple(diff[0:3]) == (0, -1, 0)):
                    all_zero = True
                    for m in range(len(diff)):
                        if m >= 3 and diff[m] != 0:
                                all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "ox"})
                if (n_rc > 1 and k < (n_rc - 1)):
                    lin_diffs = [
                        (-1, 1, 0, -1, 0, 1),
                        (-1, 1, 0, 0, -1, 0),
                        (0, 0, -1, -1, 0, 1),
                        (0, 0, -1, 0, -1, 0),
                        ]
                    if (tuple(diff[kt:kt + 6]) in lin_diffs):
                        all_zero = True
                        for m in range(len(diff)):
                            if m < kt or m >= kt + 6:
                                if diff[m] != 0:
                                    all_zero = False
                        if all_zero:
                            procs.update({tuple(diff): "lin"})
                if (tuple(diff[-3:]) == (-1, 1, 0) or
                    tuple(diff[-3:]) == (0, 0, -1)):
                    all_zero = True
                    for m in range(len(diff)):
                        if m < len(diff) - 3 and diff[m] != 0:
                            all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "red"})
    params = {
            "pigments": pigments,
            "gap": gap,
            "states": states, # use an index to return a state
            "indices": indices, # use tuple of state to return index
            "procs": procs,
            "nu_ch2o_ind": nu_ch2o_ind,
            "nu_cyc_ind": nu_cyc_ind,
            }
    return params

params = {
    "ox":   parameters(["ps_r", "ps_r"], 10.0),
    "frl":  parameters(["ps_r_frl", "ps_r_frl"], 10.0),
    "anox": parameters(["ps_anox"], 10.0),
    "exo":  parameters(["ps_exo","ps_exo", "ps_exo"], 10.0),
}

def solve(rc_type, spectrum, detrap_type, tau_diff, n_p, per_rc=True, debug=False):
    '''
    parameters
    ----------
    `rc_type`: string corresponding to params above
    `spectrum`: input spectrum from light.py
    `detrap_type`: string corresponding to detrapping regime
    `tau_diff`: float - diffusion time for whatever's being oxidised
    `n_p`: Number of pigments
    `per_rc`: if True, then give n_p pigments per photosystem; if
    False, divide n_p evenly between them

    set up an RC-only system with of type `rc_type` (see params above)
    with total excitation rate `gamma` shared equally between
    photosystems, and solve the resulting equations using scipy NNLS
    as in antenna.py and supersystem.py.
    this is fairly simple - there's no transfer rates to worry about,
    only excitation, dissipation, and the internal RC processes which
    are all generated and indexed previously.
    '''
    rcp = params[rc_type]
    n_rc = len(rcp["pigments"])
    n_rc_states = len(rcp["states"])
    fp_y = (spectrum[:, 0] * spectrum[:, 1]) / la.hcnm
    # NB: next two lines assume all photosystems are identical
    if not per_rc:
        n_p /= n_rc
    a_l = la.absorption(spectrum[:, 0], rcp["pigments"][0], 0.0)
    g_per_rc = (n_p * constants.sig_chl *
            la.overlap(spectrum[:, 0], fp_y, a_l))

    # detrapping regime
    detrap = rates["trap"]
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
                    ts = toti[indf] # total state tuple
                    # set element with the corresponding rate
                    if rt in ["lin", "red"]:
                        twa[ind][indf] = rates[rt]
                    if rt == "ox":
                        tau_ox = (tau_diff + 1.0 / rates[rt])
                        twa[ind][indf] = 1.0 / tau_ox
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
                            twa[ind][indf] = rates[rt]
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
                        k_cyc = rates["cyc"]
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
                        twa[ind][indf] += rates["rec"]
            if jind > 0:
                # occupied exciton block -> empty due to dissipation
                # final state index is i because RC state is unaffected
                twa[ind][i] = constants.k_diss
            
            if jind > 0 and jind <= n_rc:
                twa[i][ind] = g_per_rc # absorption by RCs

    k = np.zeros((side + 1, side), dtype=np.float64,
                 order='F')
    for i in range(side):
        for j in range(side):
            # if twa[i][j] != 0.0:
            #     print(f"{toti[i]} -> {toti[j]} = {twa[i][j]}")
            if (i != j):
                k[i][j]      = twa[j][i]
                k[i][i]     -= twa[i][j]
        # add a row for the probability constraint
        k[side][i] = 1.0

    b = np.zeros(side + 1, dtype=np.float64)
    b[-1] = 1.0
    p_eq, p_eq_res = la.solve(k, method='scipy')

    nu_ch2o = 0.0
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
                recomb += p_i * rates["rec"]
            if s[oxidised_indices[j]] == 1:
                redox[j, 0] += p_i
            if s[reduced_indices[j]] == 1:
                redox[j, 1] += p_i
        if i in lindices:
            nu_ch2o += rates["red"] * p_i
        if i in cycdices:
            nu_cyc += k_cyc * p_i

    if debug:
        return {
                "k": k,
                "twa": twa,
                "gamma": g_per_rc * n_rc,
                "p_eq": p_eq,
                "states": total_states,
                "lindices": lindices,
                "cycdices": cycdices,
                "nu_ch2o": nu_ch2o,
                "nu_cyc": nu_cyc,
                "redox": redox,
                "recomb": recomb,
                "tau_ox": tau_ox,
                }
    else:
        return nu_ch2o


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="simple test of RC only with given gamma",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required arguments
    parser.add_argument('-t', '--temperature', type=int, required=True,
            help=r'Stellar temperature')
    parser.add_argument('-r', '--rc_type', type=str, required=True,
            help=r'Density of quenchers \rho_q')
    parser.add_argument('-dt', '--detrap_type', type=str, required=True,
            help=r'Detrapping regime: "fast", "thermal", "energy_gap" or "none"')
    parser.add_argument('-p', '--per_rc', type=bool,
            default=True,
            help=f"Print quantities per RC")
    parser.add_argument('-d', '--debug', type=bool,
            default=True,
            help=f"Print debug information")
    args = parser.parse_args()

    spectrum, out_name = light.spectrum_setup("phoenix",
            temperature=args.temperature)
    res = solve(args.rc_type, spectrum, args.detrap_type, 100,
            args.per_rc, args.debug)
    outpath = os.path.join("out", f"{args.temperature}K",
            f"{args.rc_type}")
    os.makedirs(outpath, exist_ok=True)

    if args.debug:
        np.savetxt(f"{outpath}_twa.txt", res["twa"], fmt='%.16e')
        np.savetxt(f"{outpath}_k.txt", res["k"], fmt='%.6e')
        np.savetxt(f"{outpath}_p_eq.txt", res["p_eq"], fmt='%.16e')
        print(f"RC type: {args.rc_type}")
        print(f"excitation rate per photosystem: {res['gamma']} s^-1")
        print(f"p(0) = {res['p_eq'][0]}")
        print(f"nu_ch2o = {res['nu_ch2o']}")
        print(f"nu_cyc = {res['nu_cyc']}")
        side = len(res["p_eq"])
        for si, pi in zip(res["states"], res["p_eq"]):
            print(f"p_eq{si} = {pi}")
        print(f"redox (ox, red) for each RC = {res['redox']}")

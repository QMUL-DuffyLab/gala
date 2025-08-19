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

# these were in constants.py but they're only needed here
# dict because we need to know where cyclic/detrapping are
# and insert both rates in the combined antenna-RC matrix.
rates = {
"trap" : 1.0 / 10.0E-12,
"ox"   : 1.0 / 1.0E-3,
"lin"  : 1.0 / 10.0E-3,
"cyc"  : 1.0 / 10.0E-3,
"red"  : 1.0 / 10.0E-3,
"rec"  : 0.0,
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
        - nu_e_ind: indices of the RC states we need to sum over
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
    nu_e_ind = []
    nu_cyc_ind  = []
    mat = np.zeros((n_states, n_states), dtype='U10')
    for i in range(n_states):
        initial = states[i]
        if initial[-1] == 1 or initial[-3] == 1: # n^r_R, n^r_T
            nu_e_ind.append(i)
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
                        # mat[i][j] = rates["trap"]
                        mat[i][j] = "trap"
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
                        # mat[i][j] = rates["cyc"]
                        mat[i][j] = "cyc"
                if (tuple(diff[0:3]) == (-1, 0, 1) or
                    tuple(diff[0:3]) == (0, -1, 0)):
                    all_zero = True
                    for m in range(len(diff)):
                        if m >= 3 and diff[m] != 0:
                                all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "ox"})
                        # mat[i][j] = rates["ox"]
                        mat[i][j] = "ox"
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
                            # mat[i][j] = rates["lin"]
                            mat[i][j] = "lin"
                if (tuple(diff[-3:]) == (-1, 1, 0) or
                    tuple(diff[-3:]) == (0, 0, -1)):
                    all_zero = True
                    for m in range(len(diff)):
                        if m < len(diff) - 3 and diff[m] != 0:
                            all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "red"})
                        # mat[i][j] = rates["red"]
                        mat[i][j] = "red"
    params = {
            "pigments": pigments,
            "n_p": [constants.pigment_data[p]['n_p'] for p in pigments],
            "gap": gap,
            "states": states, # use an index to return a state
            "indices": indices, # use tuple of state to return index
            "procs": procs,
            "nu_e_ind": nu_e_ind,
            "nu_cyc_ind": nu_cyc_ind,
            "mat": mat,
            }
    return params

params = {
    "ox":   parameters(["ps_ox", "ps_r"], 10.0),
    "ox_id":   parameters(["ps_r", "ps_r"], 10.0),
    "frl":  parameters(["ps_r_frl", "ps_r_frl"], 10.0),
    "anox": parameters(["ps_anox"], 10.0),
    "exo":  parameters(["ps_exo","ps_exo", "ps_exo"], 10.0),
}

n_rc = {rct: len(params[rct]["pigments"]) for rct in params.keys()}

if __name__ == "__main__":
    import solvers
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
    res = solvers.rc_only(args.rc_type, spectrum, args.detrap_type, 100,
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
        print(f"nu_e = {res['nu_e']}")
        print(f"nu_cyc = {res['nu_cyc']}")
        side = len(res["p_eq"])
        for si, pi in zip(res["states"], res["p_eq"]):
            print(f"p_eq{si} = {pi}")
        print(f"redox (ox, red) for each RC = {res['redox']}")

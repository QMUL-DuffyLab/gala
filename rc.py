# -*- coding: utf-8 -*-
"""
27/01/2025
@author: callum

"""
import numpy as np
import itertools

# these were in constants.py but they're only needed here
# dict because we need to know where cyclic/detrapping are
# and insert both rates in the combined antenna-RC matrix.
rates = {
"trap" : 1.0 / 10.0E-12,
"ox"   : 1.0 / 1.0E-3,
"lin"  : 1.0 / 10.0E-3,
"cyc"  : 1.0 / 10.0E-3,
"red"  : 1.0 / 10.0E-3,
}

def parameters(pigments, gap):
    '''
    for a given set of photosystems, generate a dict of intra-RC
    processes and the lists of indices that pertain to linear and
    cyclic electron flow respectively for use in antenna.py.
    
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
    one_rc = [[0,0], [1,0], [0,1]]
    for i in range(n_rc):
        n_states = len(one_rc)**n_rc
        states = np.array(list(map(list, itertools.product(one_rc, repeat=n_rc)))).reshape(n_states, n_rc * len(one_rc[0]))
        indices = {j: states[j] for j in range(n_states)}
    procs = {}
    nu_ch2o_ind = []
    nu_cyc_ind  = []
    for i in range(n_states):
        initial = indices[i]
        if initial[-1] == 1: # last element is always n^r_R
            nu_ch2o_ind.append(i)
        if n_rc == 1: 
            if initial[0] == 1:
                nu_cyc_ind.append(i)
        elif n_rc > 1:
            trap_indices = [2 * k for k in range(1, n_rc)]
            if any(initial[trap_indices] == 1):
                nu_cyc_ind.append(i)
        for j in range(n_states):
            final = indices[j]
            diff = final - initial
            for k in range(n_rc):
                kt = 2 * k # index of the trap state of PS_k
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
                        # check for "cyc" in antenna.py and insert both
                        procs.update({tuple(diff): "cyc"})
                if tuple(diff[0:2]) == (-1, 1):
                    all_zero = True
                    for m in range(len(diff)):
                        if m >= 2 and diff[m] != 0:
                                all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "ox"})
                if (n_rc > 1 and k < (n_rc - 1) and 
                    tuple(diff[kt + 1:kt + 4]) == (-1, -1, 1)):
                    all_zero = True
                    for m in range(len(diff)):
                        if m < kt + 1 or m > kt + 3:
                            if diff[m] != 0:
                                all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "lin"})
                if diff[-1] == -1:
                    all_zero = True
                    for m in range(len(diff)):
                        if m != len(diff) - 1 and diff[m] != 0:
                            all_zero = False
                    if all_zero:
                        procs.update({tuple(diff): "red"})
    params = {
            "pigments": pigments,
            "gap": gap,
            "states": states,
            "procs": procs,
            "nu_ch2o_ind": nu_ch2o_ind,
            "nu_cyc_ind": nu_cyc_ind,
            }
    return params

rc_params = {
    "ox":   parameters(["ps_ox", "ps_r"], 17.0),
    "frl":  parameters(["ps_ox_frl", "ps_r_frl"], 10.0),
    "anox": parameters(["ps_anox"], 14.0),
    "exo":  parameters(["ps_exo","ps_exo", "ps_exo"], 10.0),
}

if __name__ == "__main__":
    for item in rc_params:
        d = rc_params[item]
        print("----")
        print(f"{item}:")
        print("----")
        print(f"Pigments = {d['pigments']}")
        print(f"Energy gap = {d['gap']} k_B T")
        print(f"Processes:")
        for proc in d['procs']:
            print(f"{proc} -> {d['procs'][proc]}")
        print(r"States for $\nu(CH_2O)$:")
        for i in d['nu_ch2o_ind']:
            print(f"index {i}: {d['states'][i]}")
        print(r"States for $\nu(\text{cyc})$:")
        for i in d['nu_cyc_ind']:
            print(f"index {i}: {d['states'][i]}")
        print()


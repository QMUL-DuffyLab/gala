# -*- coding: utf-8 -*-
"""
14/5/25
@author: callum
putting all the different solvers in one module
"""
import os
import time
import ctypes
import numpy as np
from scipy.optimize import nnls as scipy_nnls
from numpy.dtypes import StringDType
import constants
import light
import utils
import genetic_algorithm as ga

def diag(transfer_matrix, t=constants.tinf, debug=False):
    '''
    calculate equilibrium occupation probabilites from the transfer
    matrix `transfer_matrix` at time t via diagonalisation
    '''
    side = transfer_matrix.shape[0]
    k = np.zeros((side, side), dtype=ctypes.c_double,
                 order='F')
    for i in range(side):
        for j in range(side):
            if i != j:
                k[i][j]      = transfer_matrix[j][i]
                k[i][i]     -= transfer_matrix[i][j]

    try:
        lam, C = np.linalg.eig(k)
        Cinv = np.linalg.inv(C)
        elt = np.zeros_like(C)
        p0 = np.zeros(k.shape[0])
        p0[0] = 1.0
        for i in range(k.shape[0]):
            elt[i, i] = np.exp(lam[i] * t)
        p_eq = np.matmul(np.matmul(np.matmul(C, elt), Cinv), p0)
        # p_eq /= np.sum(p_eq)
    except np.linalg.LinAlgError:
        print("Singular matrix. oopsy daisy")
        p_eq = None
        k = None
        lam = None
        C = None
        Cinv = None

    if debug:
        return p_eq, k, lam, C, Cinv
    else:
        return p_eq

def nnls(transfer_matrix, debug=False):
    '''
    calculate equilibrium occupation probabilites from the transfer
    matrix `transfer_matrix` at time t via diagonalisation
    '''
    side = transfer_matrix.shape[0]
    k = np.zeros((side + 1, side), dtype=np.float64, order='F')
    for i in range(side):
        for j in range(side):
            if i != j:
                k[i][j]      = transfer_matrix[j][i]
                k[i][i]     -= transfer_matrix[i][j]
        k[side][i] = 1.0
    b = np.zeros(side + 1, dtype=np.float64)
    b[-1] = 1.0 
    try:
        p_eq, p_eq_res = scipy_nnls(k, b)
    except RuntimeError: # iteration limit was reached
        p_eq = np.zeros(side)
        p_eq[0] = 1.0
        p_eq_res = None
    if debug:
        return p_eq, p_eq_res
    else:
        return p_eq

def build_matrix(p, fif, debug=True):
    '''
    TODO: update this when the function's written
    '''
    # NB: units???
    lam_cs = np.zeros_like(p['k_cs'])
    k_rc = np.zeros_like(lam_cs)
    dg_rc = np.zeros_like(lam_cs)
    lam_rc = np.zeros_like(lam_cs)
    # NB: figure out signs here lol. trap energies negative?
    lam_cs = p['dE0'] - p['e'][:, 0] # \lambda_{cs} ~ -dG_{cs}
    # reorganisation energy is given in wavenumbers in constants.py
    ltilde_ev = utils.ev_nm(utils.nm_wvn(constants.l_tilde))
    lam_rc = lam_cs + ltilde_ev - np.sqrt(ltilde_ev * np.abs(lam_cs))
    dg_rc = lam_cs - (p['dE0'] + ltilde_ev)
    k_rc = p['k_cs'] * (np.sqrt(lam_cs / lam_rc) 
                     * np.exp(-utils.beta_ev * 
                              (lam_rc + dg_rc)**2/(4.0 * lam_rc)))
    # take closest entry in the spectrum to get a fractional flux value
    # note that i'm storing all the energies in eV so convert them here
    inds = [np.argmin(np.abs(fif[:, 0] - e0)) 
            for e0 in utils.ev_nm(p['dE0'])]
    gamma = fif[np.array(inds), 1] # unsure if this will work
    rates = constants.rates
    if debug:
        fw_rates = np.zeros_like(p['e'])
        bw_rates = np.zeros_like(p['e'])
        trap_rates = np.zeros_like(lam_cs)
        detrap_rates = np.zeros_like(lam_cs)
        oxlin_rates = np.zeros(constants.n_rc + 1)

    n_rc = len(lam_cs)
    n_states = (2 * (p['n_t'] + 1)) + 1
    offsets = np.array([1, *np.cumprod(n_states)][:-1])
    side = np.prod(n_states)
    t = np.zeros((side, side), dtype=ga.ft)
    tuples = np.zeros((side, n_rc), dtype=int)
    string_reps = np.zeros(side, dtype=StringDType)
    curr = np.zeros(n_rc, dtype=int)
    strs = np.zeros(n_rc, dtype=StringDType)
    final = np.zeros_like(curr)
    for i in range(side):
        # get the set of indices for each RC
        for j in range(n_rc):
            curr[j] = (i // offsets[j]) % n_states[j]
            final[j] = curr[j]
        tuples[i] = curr
        total_str = ""
        for rci, state in enumerate(curr):
            base_str = ["P", *["T" for _ in range(p['n_t'][rci])]]
            pigment_oxidised = state % 2
            if pigment_oxidised:
                '''
                then oxidation can occur. this can be via the
                donor if we're on the first RC, or via linear
                electron flow from the previous RC if we're not
                '''
                base_str[0] = "P+"
                if state == 1:
                    final[rci] = final[rci] - 1
                else:
                    final[rci] = final[rci] + 1
                if rci == 0:
                    # pigment can be reduced by donor
                    final_ind = np.dot(final, offsets)
                    # this is the constraint on the ionisation potential,
                    # essentially. won't work yet
                    rr = utils.db(-constants.e_donor,
                                       -p['i'][0], rates['ox'], 0.0)
                    t[i][final_ind] += rr[0]
                else:
                    # now we need to figure out the
                    # required state of the previous RC for linear flow
                    # to this RC. is the electron on the final trap
                    prev_state = curr[rci - 1]
                    prev_trap = (prev_state - 3) // 2
                    # NB: prev pigment is also the final state
                    # of the previous trap! if the previous pigment
                    # is still oxidised, then the electron's lost from
                    # the final trap, and we have the pigment oxidised
                    # but traps neutral. if the pigment's been reduced,
                    # the previous RC is going back to its g/s.
                    prev_pigment = prev_state % 2
                    if prev_trap == p['n_t'][rci - 1] - 1:
                        final[rci - 1] = prev_pigment
                        final_ind = np.dot(final, offsets)
                        # again, this is probably not right yet
                        # and actually might not be correct to do this
                        rr = utils.db(
                            p['e'][rci - 1][prev_trap],
                            -p['i'][rci],
                            rates['lin'], 0.0)
                        t[i][final_ind] += rr[0]
                if debug:
                    oxlin_rates[rci] = rr[0]
                # reset final to be equal to current
                for j in range(n_rc):
                    final[j] = curr[j]
            if state == 0: # ground state
                # state == 2 is photoexcitation 
                final[rci] = 2
                # this is the overall index of the final state
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += gamma[rci]
                for j in range(n_rc):
                    final[j] = curr[j]
            # state == 1 dealt with above - oxidation is only possible process
            if state == 2: # photoexcited
                base_str[0] = "P*"
                # charge separation 
                final[rci] = 3
                final_ind = np.dot(final, offsets)
                tdt = utils.db(
                        -p['i'][rci] + p['dE0'][rci], # check signs and units
                        p['e'][rci][0],
                        p['k_cs'][rci],
                        p['k_cs'][rci])
                if np.any(tdt > 1E12):
                    print("trap rates all out of whack here")
                    print(f"i_p = {p['i'][rci]}")
                    print(f"dE0ph = {p['dE0'][rci]}")
                    print(f"e[trap 1] = {p['e'][rci][0]}")
                    print(f"k_cs = {p['k_cs'][rci]:8.4e}")
                    print(f"trap = {tdt[0]:8.4e}")
                    print(f"detrap = {tdt[1]:8.4e}")
                    print(f"details of db call:")
                    rates = np.array([p['k_cs'][rci], p['k_cs'][rci]])
                    e1 = -p['i'][rci] + p['dE0'][rci]
                    e2 = p['e'][rci][0]
                    gap = e1 - e2
                    fac = np.exp(-gap * utils.beta_ev)
                    index = (int(np.sign(gap)) + 1) // 2
                    rates[index] *= fac
                    print(f"e1 = {e1:6.4f}, e2 = {e2:6.4f}, gap = {gap:6.4f}")
                    print(f"index = {index}")
                    print(f"fac = {fac:8.4e}")
                    print(f"rates = {rates}")
                    raise TypeError
                t[i][final_ind] += tdt[0]
                if debug:
                    trap_rates[rci] = tdt[0]
                    detrap_rates[rci] = tdt[1]
                # detrapping is just the other way round
                t[final_ind][i] += tdt[1]
                for j in range(n_rc):
                    final[j] = curr[j]
                # dissipation
                final[rci] = 0
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += rates['diss']
                for j in range(n_rc):
                    final[j] = curr[j]
            if state == 3: # primary CS
                base_str[0] = "P+"
                base_str[1] = "T-"
               # detrapping dealt with above
               # recombination
                final[rci] = 0
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += k_rc[rci]
                for j in range(n_rc):
                    final[j] = curr[j]
            if state >= 3:
                # this is just my convention of how the indexing works
                trap_index = (state - 3) // 2
                base_str[trap_index + 1] = "T-"
                if trap_index < p['n_t'][rci] - 1:
                    # convention here is that p['k'][rci][i] is
                    # the rate of transfer between traps i <--> i + 1
                    # and then we apply detailed balance based on those
                    # respective trap energies. so for trap index = 0,
                    # we do the 0 <--> 1 rates and so on. hence we stop
                    # the loop before we get to the final trap
                    fwbw = utils.db(
                            p['e'][rci][trap_index],
                            p['e'][rci][trap_index + 1],
                            p['k'][rci][trap_index],
                            p['k'][rci][trap_index]
                            )
                    # forward rate: T_{i} -> T_{i + 1}
                    final[rci] = final[rci] + 2
                    final_ind = np.dot(final, offsets)
                    t[i][final_ind] += fwbw[0]
                    # backward rate: T_{i + 1} -> T_{i}
                    t[final_ind][i] += fwbw[1]
                    for j in range(n_rc):
                        final[j] = curr[j]
                    if debug:
                        fw_rates[rci][trap_index] = fwbw[0]
                        bw_rates[rci][trap_index] = fwbw[1]
                if trap_index == p['n_t'][rci] - 1:
                    # if we're on the final trap all the transfer rates
                    # between traps have been set; all that's left is
                    # cyclic and/or reduction at the acceptor, because
                    # linear flow has also been dealt with above
                    if pigment_oxidised and rci > 0:
                        # cyclic - NB can do (if pigment_oxidised and rci > 0)
                        # to mimic real system where PSII can't do cyclic,
                        # or have a vector k_cyc(n_rc) and set k_cyc[0] = 0.0
                        final[rci] = 0
                        final_ind = np.dot(final, offsets)
                        t[i][final_ind] += rates['cyc']
                        for j in range(n_rc):
                            final[j] = curr[j]
                    if rci == n_rc - 1:
                        # final RC - terminal trap reduces acceptor
                        # this neutralises the trap but does nothing to
                        # the pigment, so if the pigment's oxidised it
                        # stays oxidised, and if it's not we go back to g/s
                        final[rci] = pigment_oxidised
                        final_ind = np.dot(final, offsets)
                        rr = utils.db(p['e'][rci][trap_index],
                                    constants.e_acceptor, rates['red'], 0.0)
                        if debug:
                            oxlin_rates[-1] = rr[0]
                        t[i][final_ind] += rr[0]
                        for j in range(n_rc):
                            final[j] = curr[j]
            strs[rci] = " ".join(base_str)
        string_reps[i] = " ".join(strs)
    if debug:
        return {
                't': t,
                'tuples': tuples,
                'string_reps': string_reps,
                'gamma': gamma,
                'lam_cs': lam_cs,
                'k_rc': k_rc,
                'dg_rc': dg_rc,
                'lam_rc': lam_rc,
                'fw_rates': fw_rates,
                'bw_rates': bw_rates,
                'trap_rates': trap_rates,
                'detrap_rates': detrap_rates,
                'oxlin_rates': oxlin_rates,
                }
    else:
        return t, tuples, string_reps

def solve(p, fif, debug=False, atol=1e-5, **solver_kwargs):
    output_deranged = False
    output = 0.0
    redox = np.zeros((constants.n_rc, 2), dtype=ga.ft)
    tt, tuples, string_reps = build_matrix(p, fif, debug)
    # do NNLS in first instance
    p_eq = nnls(tt, debug)
    if np.abs(np.sum(p_eq) - 1.0) > atol:
        output_deranged = True
    if np.abs(p_eq[0] - 1.0) > atol:
        for ii, p_eq_i in enumerate(p_eq):
            curr = tuples[ii]
            for kk in range(constants.n_rc):
                if curr[kk] == 1:
                    # this is the state P+ T T ... T (photosystem oxidised)
                    redox[kk][0] += p_eq_i
                if curr[kk] > 3 and curr[kk] % 2 == 0:
                    # these correspond to P T- T ... T, P T T- ... T, P T T ... T-
                    redox[kk][1] += p_eq_i
                if (kk == constants.n_rc - 1 and
                    (curr[kk] - 3) // 2 == p['n_t'][kk] - 1):
                    # electron on final trap of final rc
                    output += p_eq_i * constants.rates['red']

    return output, redox, output_deranged

'''
NB:
for calculation of the redox states of each RC and also the output,
there's some clever shit we can do with the offsets and stuff. e.g.
for a given rc we can use the offsets to determine which indices of
the total matrix correspond to it being P+; the sum of p_eq(those indices)
will be the amount of time it's oxidised, the sum of all the others will be
the amount of time it's reduced, and so on. for the output it's just any
index where the final trap is reduced. just need to figure out some
clever arithmetic for it and then it should be fine to get rid of the
string representations and tuples in the final code - then we can keep this
version with the dict output and all that as a separate function that isn't
called and point to it in the documentation, and also use it in a test to
confirm that the two functions give identical output
'''

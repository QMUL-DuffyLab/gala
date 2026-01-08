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
from scipy.constants import h, c
from scipy.constants import Boltzmann as kB
import constants
import light
import utils
import genetic_algorithm as ga
import build_matrix

def diag(transfer_matrix, t=constants.tinf):
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

    lam, C = np.linalg.eig(k)
    Cinv = np.linalg.inv(C)
    elt = np.zeros_like(C)
    p0 = np.zeros(k.shape[0])
    p0[0] = 1.0
    for i in range(k.shape[0]):
        elt[i, i] = np.exp(lam[i] * t)
    p_eq = np.matmul(np.matmul(np.matmul(C, elt), Cinv), p0)
    p_eq /= np.sum(p_eq)
    return np.real(p_eq), k, lam, C, Cinv

def build_matrix(p, fif):
    '''
    TODO: update this when the function's written
    '''
    # NB: units???
    lam_cs = np.zeros_like(p['k_cs'])
    k_rc = np.zeros_like(lam_cs)
    dg_rc = np.zeros_like(lam_cs)
    lam_rc = np.zeros_like(lam_cs)
    lam_cs = p['dE0'] - p['e'][:, 0] # \lambda_{cs} ~ -dG_{cs}
    lam_rc = lam_cs + constants.l_tilde - np.sqrt(constants.l_tilde * dg_cs)
    dg_rc = lam_cs - (p['dE0'] + constants.l_tilde)
    k_rc = p['k_cs'] * (np.sqrt(lam_cs / lam_rc) 
                     * exp(-beta * (lam_rc + dg_rc)**2/(4.0 * lam_rc)))
    # take closest entry in the spectrum to get a fractional flux value
    # note that i'm storing all the energies in eV so convert them here
    inds = [np.argmin(fif[:, 0] - e0) for e0 in utils.ev_nm(p['dE0'])]
    gamma = fif[np.array(inds), 1] # unsure if this will work
    # also NB: need pairs of fw and bw trap transfer rates
    rates = constants.rc_rates

    n_rc = len(lam_cs)
    n_states = (2 * (p['n_t'] + 1)) + 1
    offsets = np.array([1, *np.cumprod(n_states)][:-1])
    side = np.prod(n_states)
    t = np.zeros((side, side), dtype=StringDType)
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
                    rr = utils.db_pair(constants.e_donor,
                                       p['i'][0], rates['ox'], 0.0)
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
                        rr = utils.db_pair(
                            p['e'][rci - 1][prev_trap],
                            p['i'][rci],
                            rates['lin'], 0.0)
                        t[i][final_ind] += rr[0]
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
                tdt = utils.db_pair(
                        -p['i'][rci] + p['dE0'][rci], # check signs and units
                        p['e'][rci][0],
                        p['k_cs'][rci],
                        p['k_cs'][rci])
                t[i][final_ind] += tdt[0]
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
                    fwbw = utils.db_pair(
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
                if trap_index == p['n_t'][rci] - 1:
                    # if we're on the final trap all the transfer rates
                    # between traps have been set; all that's left is
                    # cyclic and/or reduction at the acceptor, because
                    # linear flow has also been dealt with above
                    if pigment_oxidised:
                        # cyclic - NB can do (if pigment_oxidised and rci > 0)
                        # to mimic real system where PSII can't do cyclic,
                        # or have a vector k_cyc(n_rc) and set k_cyc[0] = 0.0
                        final[rci] = 0
                        final_ind = np.dot(final, offsets)
                        t[i][final_ind] += k_cyc
                        for j in range(n_rc):
                            final[j] = curr[j]
                    if rci == n_rc - 1:
                        # final RC - terminal trap reduces acceptor
                        # how is this distinguishable from cyclic?
                        # do we need a separate final acceptor state?
                        final_ind = np.dot(final, offsets)
                        rr = utils.db_pair(p['e'][trap_index],
                                    constants.e_acceptor, k_out, 0.0)
                        t[i][final_ind] += rr[0]
                        for j in range(n_rc):
                            final[j] = curr[j]
            strs[rci] = " ".join(base_str)
        string_reps[i] = " ".join(strs)
    return t, tuples, string_reps

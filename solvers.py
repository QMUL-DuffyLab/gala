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
import rc as rcm
import build_matrix

'''
here is where the solvers for the RC optimisation will go.
'''

# spectrometer on mars rovers etc.?? what can they do up there?
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

def build_matrix(population, index):
    '''
    TODO: update this when the function's written
    '''
    p = population[index]
    # NB: units???
    beta = 1.0 / (Boltzmann * 300.0 * (1. / (100.0 * c))) # in wavenumbers
    de0ph = 0.0
    '''
    dg_cs = de0ph - p.E[0] # this is actually -deltaG_{cs}
    lam_rc = dg_cs + p.line_reorg - np.sqrt(p.line_reorg * dg_cs)
    dg_rc = dg_cs - (de0ph + p.line_reorg)
    k_rc = p.k_cs * (np.sqrt(dg_cs / lam_rc) 
                     * exp(-beta * (lam_rc + dg_rc)**2/(4.0 * lam_rc)))
    '''
    k_rc = "recombination"
    gamma = "gamma"
    k_ox = "oxidation"
    k_r = 0.0 # unsure what this should be but i guess fast
    k_out = "output"
    # detailed balance on this i guess
    k_dt ="detrapping"
    k_lin = "linear"
    k_cyc = "cyc"

    n_rc = len(n_traps)
    n_states = np.array(2 * (n_traps + 1) + 1)
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
            base_str = ["P", *["T" for _ in range(n_traps[rci])]]
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
                    t[i][final_ind] += k_ox
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
                    if prev_trap == n_traps[rci - 1] - 1:
                        final[rci - 1] = prev_pigment
                        final_ind = np.dot(final, offsets)
                        t[i][final_ind] += k_lin # what should this be???
                for j in range(n_rc):
                    final[j] = curr[j]
            if state == 0: # ground state
                # state == 2 is photoexcitation 
                final[rci] = 2
                # this is the overall index of the final state
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += f"gamma[{rci}]"
                for j in range(n_rc):
                    final[j] = curr[j]
            # state == 1 is dealt with above - oxidation is the only possible process
            if state == 2: # photoexcited
                base_str[0] = "P*"
                # charge separation 
                final[rci] = 3
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += f"cs[{rci}]"
                for j in range(n_rc):
                    final[j] = curr[j]
                # dissipation
                final[rci] = 0
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += "dissipation"
                for j in range(n_rc):
                    final[j] = curr[j]
            if state == 3: # primary CS
                base_str[0] = "P+"
                base_str[1] = "T-"
               # detrapping
                final[rci] = 2
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += k_dt
                for j in range(n_rc):
                    final[j] = curr[j]
               # recombination
                final[rci] = 0
                final_ind = np.dot(final, offsets)
                t[i][final_ind] += k_rc
                for j in range(n_rc):
                    final[j] = curr[j]
            if state >= 3:
                # some logic to do with which trap we're on
                # detailed balance on the rates and so on
                trap_index = (state - 3) // 2
                base_str[trap_index + 1] = "T-"
                if trap_index == n_traps[rci] - 1:
                    if pigment_oxidised:
                        # cyclic - NB can do (if pigment_oxidised and rci > 0 to mimic real system)
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
                        t[i][final_ind] += k_out
                        for j in range(n_rc):
                            final[j] = curr[j]
                if trap_index > 0:
                    # need to do detailed balance on fw and bw
                    #bw = p.rates[rci][trap_index] # T_i -> T_i - 1
                    bw = f"bw from {trap_index}"
                    final[rci] = final[rci] - 2
                    final_ind = np.dot(final, offsets)
                    t[i][final_ind] += bw
                    for j in range(n_rc):
                        final[j] = curr[j]
                if trap_index < n_traps[rci] - 1:
                    #fw = p.rates[rci][trap_index] # T_i -> T_i + 1
                    fw = f"fw from {trap_index}"
                    final[rci] = final[rci] + 2
                    final_ind = np.dot(final, offsets)
                    t[i][final_ind] += fw
                    for j in range(n_rc):
                        final[j] = curr[j]
            strs[rci] = " ".join(base_str)
        string_reps[i] = " ".join(strs)
    return t, tuples, string_reps

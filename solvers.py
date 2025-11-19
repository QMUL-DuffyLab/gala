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

def rc_matrix(p, gamma):
    '''
    build and solve the equations of motion for the RC model.
    the RC basically consists of a donor, a pigment,
    n_traps >= 1 trap states and an acceptor, and can be
    represented as a tuple (D P T ... A). the photoexcited
    pigment is denoted P* after which primary charge separation
    happens, followed by a set of branching reactions which end up
    at (D+ P T ... A-) and then subsequent carbon reduction etc.
    The only difficulty comes from allowing arbitrary numbers of
    trap states and generating indices for all possible states of
    the system (see various comments below), but it's not that bad.
    Outputs the equilibrium probabilities and the total output
    '''
    # NB: units???
    beta = 1.0 / (kB * 300.0 * (1. / (100.0 * c))) # in wavenumbers
    de0ph = 0.0
    dg_cs = de0ph - p.E[0] # this is actually -deltaG_{cs}
    lam_rc = dg_cs + p.line_reorg - np.sqrt(p.line_reorg * dg_cs)
    dg_rc = dg_cs - (de0ph + p.line_reorg)
    k_rc = p.k_cs * (np.sqrt(dg_cs / lam_rc) 
                     * exp(-beta * (lam_rc + dg_rc)**2/(4.0 * lam_rc)))
    gamma = 0.0
    k_ox = 1.0 / (1.0E-3)
    k_r = 0.0 # unsure what this should be but i guess fast
    k_out = 1.0 / (10.0E-3)
    # detailed balance on this i guess
    k_dt = 0.0

    side = 2 * (p.n_traps + 2)
    t = np.zeros((side, side), dtype=np.float64)
    '''
    index 0 is the ground state
    index 1 is the photoexcited state (D P* T ... T A)
    index 2 is the initial charge-separated state (D P+ T- ... T A)
    '''
    t[0][1] = gamma
    t[1][0] = constants.k_diss
    k_cs = 
    t[1][2] = p.k_cs
    t[2][1] = k_dt
    t[2][0] = k_rc
    for i in range(p.n_traps):
        '''
        notes:
        Consider i here as indexing over each trap state being
        reduced in turn, so i = 0 is (D P T- T ... T A).
        For each trap i being reduced there are two possibilities:
        either the pigment is oxidised, or the donor is oxidised
        (either (D P+ T ... T- ... T A) or (D+ P T ... T- ... T A)).
        these are denoted `dn` (donor neutral) and `do` (donor oxidised)
        and i give them total indices 2i + 2 and 2i + 3.
        dn -> do always happens with rate k_ox by definition.
        if we're at the final trap then the reduction of the trap
        occurs with rate k_r, also by definition.
        if there are multiple traps, the intermediate rates are given
        by the gaps and rates defined in the genome; these are
        reversible e- transfers from one trap to the next
        which i denote `fw` and `bw`, forward being away from the donor.
        '''
        dn = 2*i + 2
        do = 2*i + 3
        t[dn][do] = k_ox
        if i == p.n_traps - 1:
            t[dn][dn + 2] = k_r
            t[do][do + 2] = k_r
        else:
            # need to do detailed balance on fw and bw
            fw = 0.0 # T_i -> T_i + 1
            t[dn][dn + 2] = fw
            t[do][do + 2] = fw
        if i > 0:
            bw = 0.0 # T_i -> T_i - 1
            t[dn][dn - 2] = bw
            t[do][do - 2] = bw
    '''
    the way the indexing is set up, the final row is always
    (D+ P T ... T A-), so this is always true
    '''
    t[side - 1][0] = k_out
    # diagonalisation is the same as for the main GA
    p_eq, k, lam, C, Cinv = diag(t)

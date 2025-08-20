import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport NAN, log

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef cnp.ndarray build_matrix(int n_b, int n_s, int n_rc,
        cnp.ndarray[DTYPE_t, ndim=2] rc_mat,
        double alpha, double k_diss, double k_trap, double k_detrap,
        cnp.ndarray[DTYPE_t, ndim=1] gamma, cnp.ndarray[DTYPE_t, ndim=1] k_b):
    
    cdef int n_rc_states = 4**n_rc # total number of states of all RCs
    cdef int side = n_rc_states * ((n_b * n_s) + n_rc + 1)
    cdef int i, j, k, logi, ind, indf, which_rc, jind, bi, offset

    cdef cnp.ndarray[DTYPE_t, ndim=2] twa = np.zeros([side, side], dtype=DTYPE)
    cdef DTYPE_t [:, :] twav = twa
    cdef DTYPE_t [:, :] rcv = rc_mat
    cdef DTYPE_t [:] gv = gamma
    cdef DTYPE_t [:] kv = k_b

    for j in prange(0, side, n_rc_states, nogil=True):
        jind = j // n_rc_states
        # jind == 0 is empty antenna, 0 + n_rc_states is RC 1 occupied, etc
        # intra-RC processes are the same in each block
        for i in range(n_rc_states):
            ind = i + j # total index
            for k in range(n_rc_states):
                indf = k + j
                twav[ind][indf] = rcv[i][k]
                if rcv[i][k] == NAN: # trapping
                    which_rc = (n_rc - 1) - <int>(log(k - i) / log(4.0))
                    '''
                    indf above assumes that the state of the antenna
                    doesn't change, which is not the case for trapping.
                    so zero out the above rate and then check: if
                    jind == which_rc + 1 we're in the correct block
                    (the exciton is moving from the correct RC), and
                    we go back to the empty antenna block
                    '''
                    twav[ind][indf] = 0.0
                    if jind == which_rc + 1:
                        twav[ind][k] = k_trap
                        # detrapping:
                        indf = i + ((which_rc + 1) * n_rc_states)
                        twav[k][indf] = k_detrap

            if jind > 0:
                # occupied exciton block -> empty due to dissipation
                # final state index is i because RC state is unaffected
                twav[ind][i] = k_diss
            
            if jind > 0 and jind <= n_rc:
                twav[i][ind] = gv[jind - 1] # absorption by RCs

            # antenna rate stuff
            if jind > n_rc: # population in antenna subunit
                # index on branch
                bi = (jind - n_rc - 1) % n_s
                twav[i][ind] = gv[n_rc + bi] # absorption by this block
                if bi == 0:
                    # root of branch - transfer to RC exciton states possible
                    for k in range(n_rc):
                        # transfer to RC 0 is transfer to jind 1
                        offset = (n_rc - k) * n_rc_states
                        # inward transfer to RC k
                        twav[ind][ind - offset] = kv[2 * k + 1]
                        # outward transfer from RC k
                        twav[ind - offset][ind] = kv[2 * k]
                if bi > 0:
                    # inward along branch
                    twav[ind][ind - n_rc_states] = kv[2 * (n_rc + bi) - 1]
                if bi < (n_s - 1):
                    # outward allowed
                    twav[ind][ind + n_rc_states] = kv[2 * (n_rc + bi) + 1]
    return twa

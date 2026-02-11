# cython: language_level=3
# cython: profile=True
import numpy as np
import constants
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport isnan, log
from libc.stdio cimport printf
cimport cython

cnp.import_array()
DTYPE = np.float64
ITYPE = np.int64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t

cdef packed struct genome_type:
    DTYPE_t[constants.n_rc]  dE0
    DTYPE_t[constants.n_rc]  i_p
    DTYPE_t[constants.n_rc]  k_cs
    ITYPE_t[constants.n_rc]  n_t
    DTYPE_t[constants.n_rc, constants.n_t_max]  k_t
    DTYPE_t[constants.n_rc, constants.n_t_max]  e_t

@cython.boundscheck(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] build_matrix(genome_type p,
        cnp.ndarray[DTYPE_t, ndim=2] fif):
    '''
    TODO: write docstring :)
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
                    rr = utils.db(constants.e_donor,
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
                        output_rate = rr[0]
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
        return t, tuples, output_rate

@cython.boundscheck(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] build_matrix(int n_b, int n_s, int n_rc,
        cnp.ndarray[DTYPE_t, ndim=2] rc_mat,
        double alpha, double k_diss, double k_trap, double k_detrap,
        cnp.ndarray[DTYPE_t, ndim=1] gamma, cnp.ndarray[DTYPE_t, ndim=1] k_b):
    
    cdef int n_rc_states = <int>(4**n_rc) # total number of states of all RCs
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
                if not isnan(rcv[i][k]):
                    twav[ind][indf] = rcv[i][k]
                else: # trapping
                    which_rc = (n_rc - 1) - <int>(log(k - i) / log(4.0))
                    '''
                    indf above assumes that the state of the antenna
                    doesn't change, which is not the case for trapping.
                    so zero out the above rate and then check: if
                    jind == which_rc + 1 we're in the correct block
                    (the exciton is moving from the correct RC), and
                    we go back to the empty antenna block
                    '''
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
                        twav[ind][offset + i] = kv[2 * k + 1]
                        # outward transfer from RC k
                        twav[offset + i][ind] = kv[2 * k]
                if bi > 0:
                    # inward along branch
                    twav[ind][ind - n_rc_states] = kv[2 * (n_rc + bi) - 1]
                if bi < (n_s - 1):
                    # outward allowed
                    twav[ind][ind + n_rc_states] = kv[2 * (n_rc + bi)]
    return twa

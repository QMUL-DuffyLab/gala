# -*- coding: utf-8 -*-
"""
06/11/2023
@author: callum
"""

import ctypes
import timeit
import numpy as np
import constants
import genetic_algorithm as ga

if __name__ == "__main__":
    la = ctypes.cdll.LoadLibrary("./libantenna.so")
    la.antenna.argtypes = [ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]

    rng = np.random.default_rng()

    # these will be args eventually i guess
    ts = 2600
    init_type = 'random' # can be radiative or random
    avgs_file = "out/avgs_{:4d}K.dat".format(ts)
    avgsq_file = "out/avgsq_{:4d}K.dat".format(ts)
    best_file = "out/best_{:4d}K.dat".format(ts)

    spectrum_file = constants.spectrum_prefix \
                    + '{:4d}K'.format(ts) \
                    + constants.spectrum_suffix
    # instead of unpacking and pulling both arrays, pull once
    # and split by column. this stops the arrays being strided,
    # so we can use them in the C code below without messing around
    phoenix_data = np.loadtxt(spectrum_file)
    l    = phoenix_data[:, 0]
    ip_y = phoenix_data[:, 1]
    l_c = np.ctypeslib.as_ctypes(np.ascontiguousarray(l))
    ip_y_c = np.ctypeslib.as_ctypes(np.ascontiguousarray(ip_y))

    population = [None for _ in range(constants.n_individuals)]
    avgs  = np.zeros(8)
    avgsq = np.zeros(8)
    running_avgs  = []
    running_avgsq = []
    n_s_changes = np.zeros(2)
    c_time = 0.0
    gen = 0
    total_start = timeit.default_timer()
    # initialise population
    for i in range(constants.n_individuals):
        population[i] = ga.initialise_individual(rng, init_type)

    while gen < constants.max_generations:
        avgs.fill(0.0)
        avgsq.fill(0.0)
        for i in range(constants.n_individuals):
            '''
            setup for calling C version
            '''
            n_b = population[i].n_b
            n_s = population[i].n_s
            side = (n_b * n_s) + 2
            n_p   = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.uint32))
            lp    = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.float64))
            width = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.float64))
            n_p[0]   = constants.rc_params[0]
            lp[0]    = constants.rc_params[2]
            width[0] = constants.rc_params[3]
            for j in range(n_s):
                n_p[j + 1]   = population[i].n_p[j]
                lp[j + 1]    = population[i].lp[j]
                width[j + 1] = population[i].w[j]
            n_eq   = (ctypes.c_double * side)()
            nu_phi = np.ctypeslib.as_ctypes(np.zeros(2, dtype=np.float64))
            kp = (ctypes.c_double * len(constants.k_params))(*constants.k_params)

            # start timer here to time the actual function only lol
            c_start = timeit.default_timer()
            la.antenna(l_c, ip_y_c,
                    ctypes.c_double(constants.sig_chl),
                    ctypes.c_double(constants.rc_params[1]), kp,
                    ctypes.c_double(constants.T),
                    n_p, lp, width,
                    ctypes.c_uint(n_b), ctypes.c_uint(n_s),
                    ctypes.c_uint(len(l)), n_eq, nu_phi)
            population[i].nu_e  = nu_phi[0]
            population[i].phi_f = nu_phi[1]
            c_time += timeit.default_timer() - c_start
            avgs[0]  += nu_phi[0]
            avgsq[0] += nu_phi[0]**2
            avgs[1]  += nu_phi[1]
            avgsq[1] += nu_phi[1]**2
            avgs[2]  += nu_phi[0] * nu_phi[1]
            avgsq[2] += (nu_phi[0] * nu_phi[1])**2
            avgs[3]  += np.sum(n_p[1:]) / (len(n_p) - 1) # don't count the RC
            avgsq[3] += np.sum(np.square(n_p[1:])) / (len(n_p) - 1)
            avgs[4]  += np.sum(lp[1:]) / (len(lp) - 1) # don't count the RC
            avgsq[4] += np.sum(np.square(lp[1:])) / (len(lp) - 1)
            avgs[5]  += np.sum(width[1:]) / (len(lp) - 1)
            avgsq[5] += np.sum(np.square(width[1:])) / (len(lp) - 1)
            avgs[6]  += n_b
            avgsq[6] += n_b**2
            avgs[7]  += n_s
            avgsq[7] += n_s**2

        avgs /= constants.n_individuals
        avgsq /= constants.n_individuals
        running_avgs.append(avgs)
        running_avgsq.append(avgsq)
        print("Generation {:4d}: ".format(gen))
        print("================")
        print(f"<ν_e>     = {avgs[0]:10.4n}\t<ν_e^2>       = {avgsq[0]:10.4n}")
        print(f"<φ_f>     = {avgs[1]:10.4n}\t<φ_f^2>       = {avgsq[1]:10.4n}")
        print(f"<φ_f ν_e> = {avgs[2]:10.4n}\t<(φ_f ν_e)^2> = {avgsq[2]:10.4n}")
        print(f"<n_p>     = {avgs[3]:10.4n}\t<n_p^2>       = {avgsq[3]:10.4n}")
        print(f"<λ_p>     = {avgs[4]:10.4n}\t<λ_p^2>       = {avgsq[4]:10.4n}")
        print(f"<w>       = {avgs[5]:10.4n}\t<w^2>         = {avgsq[5]:10.4n}")
        print(f"<n_b>     = {avgs[6]:10.4n}\t<n_b^2>       = {avgsq[6]:10.4n}")
        print(f"<n_s>     = {avgs[7]:10.4n}\t<n_s^2>       = {avgsq[7]:10.4n}")

        survivors, best = ga.selection(rng, population)
        avg_nb_surv = np.sum(np.array([s.n_b
                                for s in survivors]) / len(survivors))
        avg_ns_surv = np.sum(np.array([s.n_s
                                for s in survivors]) / len(survivors))
        print("Average n_b, n_s of survivors: ", avg_nb_surv, avg_ns_surv)
        print("\n")

        avg_survivor_fitness = 0.0 # calculate this here
        with open(best_file, "w") as f:
            f.write(str(best))
        f.close()
        # old_pop = population.copy()
        # old_avg_ns = np.sum(np.array([len(p['params']) - 1
        #                         for p in population]) / len(population))
        population = ga.reproduction(rng, survivors, population)
        # new_avg_ns = np.sum(np.array([len(p['params']) - 1
        #                         for p in new_pop]) / len(new_pop))
        # print("old <n_s> = ", old_avg_ns, " new <n_s> = ", new_avg_ns)
        for i in range(constants.n_individuals):
            p = rng.random()
            if p < constants.mutation_rate:
                population[i] = ga.mutation(rng, population[i], n_s_changes)
        # print(old_pop == population)
        # print("n_s mutation changes: ", n_s_changes)
        gen += 1


np.savetxt(avgs_file, np.array(running_avgs))
np.savetxt(avgsq_file, np.array(running_avgsq))

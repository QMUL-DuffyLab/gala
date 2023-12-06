# -*- coding: utf-8 -*-
"""
06/11/2023
@author: callum
"""

import ctypes
from datetime import datetime
import pandas as pd
import numpy as np
import constants
import genetic_algorithm as ga
import plots

if __name__ == "__main__":
    la = ctypes.cdll.LoadLibrary("./libantenna.so")
    la.antenna.argtypes = [ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]

    rng = np.random.default_rng()
    df = {}
    df['params'] = constants.c_dict

    # these will be args eventually i guess
    temps = [2300, 2600, 2800, 3300, 3700, 3800, 4300, 4400, 4800, 5800]
    for ts in temps:
        print("T = ", ts)
        n_runs = 3
        init_type = 'random' # can be radiative or random
        avgs_prefix  = "out/avgs_{:4d}K".format(ts)
        avgsq_prefix = "out/avgsq_{:4d}K".format(ts)
        np_prefix    = "out/np_{:4d}K".format(ts)
        npsq_prefix  = "out/npsq_{:4d}K".format(ts)
        lp_prefix    = "out/lp_{:4d}K".format(ts)
        lpsq_prefix  = "out/lpsq_{:4d}K".format(ts)
        w_prefix    = "out/w_{:4d}K".format(ts)
        wsq_prefix  = "out/wsq_{:4d}K".format(ts)
        nlw_pop_prefix = "out/nlw_pop_{:4d}K".format(ts)
        best_prefix  = "out/best_{:4d}K".format(ts)

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

        for i in range(n_runs):
            # dftr = df["{:4d}".format(ts)]["{:1d}".format(i)]
            df['input_file'] = spectrum_file
            population = [None for _ in range(constants.population_size)]
            avgs_file = avgs_prefix + "_r{:1d}.dat".format(i)
            avgsq_file = avgsq_prefix + "_r{:1d}.dat".format(i)
            best_file = best_prefix + "_r{:1d}.dat".format(i)
            np_file = np_prefix + "_r{:1d}.dat".format(i)
            npsq_file = npsq_prefix + "_r{:1d}.dat".format(i)
            lp_file = lp_prefix + "_r{:1d}.dat".format(i)
            lpsq_file = lpsq_prefix + "_r{:1d}.dat".format(i)
            w_file = w_prefix + "_r{:1d}.dat".format(i)
            wsq_file = wsq_prefix + "_r{:1d}.dat".format(i)
            nlw_pop_file = nlw_pop_prefix + "_r{:1d}.dat".format(i)
            with open(best_file, "w") as f:
                pass # start a new file
            f.close()

            avgs  = np.zeros(9)
            avgsq = np.zeros(9)
            np_avg   = np.zeros(constants.bounds['n_s'][1])
            lp_avg   = np.zeros(constants.bounds['n_s'][1])
            w_avg    = np.zeros(constants.bounds['n_s'][1])
            np_avgsq = np.zeros(constants.bounds['n_s'][1])
            lp_avgsq = np.zeros(constants.bounds['n_s'][1])
            w_avgsq = np.zeros(constants.bounds['n_s'][1])
            nlw_pop   = np.zeros(constants.bounds['n_s'][1])
            running_avgs  = np.zeros((constants.max_gen, 9))
            running_avgsq = np.zeros((constants.max_gen, 9))
            n_s_changes = np.zeros(2)
            gen = 0
            # initialise population
            for j in range(constants.population_size):
                population[j] = ga.initialise_individual(rng, init_type)

            while gen < constants.max_gen:
                avgs.fill(0.0)
                avgsq.fill(0.0)
                nlw_pop.fill(0.0)
                for j in range(constants.population_size):
                    '''
                    setup for calling C version
                    '''
                    n_b = population[j].n_b
                    n_s = population[j].n_s
                    side = (n_b * n_s) + 2
                    n_p   = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.uint32))
                    lp    = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.float64))
                    width = np.ctypeslib.as_ctypes(np.zeros(n_s + 1, dtype=np.float64))
                    n_p[0]   = constants.rc_params[0]
                    lp[0]    = constants.rc_params[1]
                    width[0] = constants.rc_params[2]
                    for k in range(n_s):
                        n_p[k + 1]   = population[j].n_p[k]
                        lp[k + 1]    = population[j].lp[k]
                        width[k + 1] = population[j].w[k]
                    n_eq   = (ctypes.c_double * side)()
                    nu_phi = np.ctypeslib.as_ctypes(np.zeros(3, dtype=np.float64))
                    kp = (ctypes.c_double * len(constants.k_params))(*constants.k_params)

                    la.antenna(l_c, ip_y_c,
                            ctypes.c_double(constants.sig_chl), kp,
                            ctypes.c_double(constants.T),
                            n_p, lp, width,
                            ctypes.c_uint(n_b), ctypes.c_uint(n_s),
                            ctypes.c_uint(len(l)), n_eq, nu_phi)
                    population[j].nu_e  = nu_phi[0]
                    population[j].phi_f = nu_phi[1]
                    avgs[0]  += nu_phi[0]
                    avgsq[0] += nu_phi[0]**2
                    avgs[1]  += nu_phi[1]
                    avgsq[1] += nu_phi[1]**2
                    avgs[2]  += nu_phi[0] * nu_phi[1]
                    avgsq[2] += (nu_phi[0] * nu_phi[1])**2
                    avgs[3]  += np.sum(n_p[1:]) / (len(n_p) - 1) # don't count RC
                    avgsq[3] += np.sum(np.square(n_p[1:])) / (len(n_p) - 1)
                    avgs[4]  += np.sum(lp[1:]) / (len(lp) - 1)
                    avgsq[4] += np.sum(np.square(lp[1:])) / (len(lp) - 1)
                    avgs[5]  += np.sum(width[1:]) / (len(width) - 1)
                    avgsq[5] += np.sum(np.square(width[1:])) / (len(width) - 1)
                    avgs[6]  += n_b
                    avgsq[6] += n_b**2
                    avgs[7]  += n_s
                    avgsq[7] += n_s**2
                    avgs[8]  += nu_phi[2]
                    avgsq[8] += nu_phi[2]**2
                    for k in range(n_s):
                        nlw_pop[k] += 1
                        np_avg[k] += n_p[k + 1]
                        lp_avg[k] += lp[k + 1]
                        w_avg[k]  += width[k + 1]
                        np_avgsq[k] += n_p[k + 1]**2
                        lp_avgsq[k] += lp[k + 1]**2
                        w_avgsq[k]  += width[k + 1]**2

                avgs /= constants.population_size
                avgsq /= constants.population_size
                std_dev = np.sqrt(avgsq - np.square(avgs))
                running_avgs[gen] = avgs
                running_avgsq[gen] = avgsq
                np_avg = np.divide(np_avg, nlw_pop, where=nlw_pop > 0.0)
                lp_avg = np.divide(lp_avg, nlw_pop, where=nlw_pop > 0.0)
                w_avg = np.divide(w_avg, nlw_pop, where=nlw_pop > 0.0)
                np_avgsq = np.divide(np_avgsq, nlw_pop, where=nlw_pop > 0.0)
                lp_avgsq = np.divide(lp_avgsq, nlw_pop, where=nlw_pop > 0.0)
                w_avgsq = np.divide(w_avgsq, nlw_pop, where=nlw_pop > 0.0)
                print("Generation {:4d}: ".format(gen))
                print("================")
                print(f"<ν_e>     = {avgs[0]:10.4n}\t<ν_e^2>       = {avgsq[0]:10.4n}\tσ = {std_dev[0]:10.4n}")
                print(f"<φ_f>     = {avgs[1]:10.4n}\t<φ_f^2>       = {avgsq[1]:10.4n}\tσ = {std_dev[1]:10.4n}")
                print(f"<φ_f ν_e> = {avgs[2]:10.4n}\t<(φ_f ν_e)^2> = {avgsq[2]:10.4n}\tσ = {std_dev[2]:10.4n}")
                print(f"<n_p>     = {avgs[3]:10.4n}\t<n_p^2>       = {avgsq[3]:10.4n}\tσ = {std_dev[3]:10.4n}")
                print(f"<λ_p>     = {avgs[4]:10.4n}\t<λ_p^2>       = {avgsq[4]:10.4n}\tσ = {std_dev[4]:10.4n}")
                print(f"<w>       = {avgs[5]:10.4n}\t<w^2>         = {avgsq[5]:10.4n}\tσ = {std_dev[5]:10.4n}")
                print(f"<n_b>     = {avgs[6]:10.4n}\t<n_b^2>       = {avgsq[6]:10.4n}\tσ = {std_dev[6]:10.4n}")
                print(f"<n_s>     = {avgs[7]:10.4n}\t<n_s^2>       = {avgsq[7]:10.4n}\tσ = {std_dev[7]:10.4n}")
                print(f"<n_eq>    = {avgs[8]:10.4n}\t<n_eq^2>      = {avgsq[8]:10.4n}\tσ = {std_dev[8]:10.4n}")

                survivors, best = ga.selection(rng, population)
                avg_nb_surv = np.sum(np.array([s.n_b
                                        for s in survivors]) / len(survivors))
                avg_ns_surv = np.sum(np.array([s.n_s
                                        for s in survivors]) / len(survivors))
                print("Average n_b, n_s of survivors: ", avg_nb_surv, avg_ns_surv)
                print("\n")

                with open(best_file, "a") as f:
                    f.write(str(best))
                    f.write("\n")
                f.close()
                population = ga.reproduction(rng, survivors, population)
                for j in range(constants.population_size):
                    p = rng.random()
                    if p < constants.mu_rate:
                        population[j] = ga.mutation(rng, population[j], n_s_changes)
                gen += 1

            np.savetxt(avgs_file, np.array(running_avgs))
            np.savetxt(avgsq_file, np.array(running_avgsq))
            np.savetxt(np_file, np_avg)
            np.savetxt(npsq_file, np_avgsq)
            np.savetxt(lp_file, lp_avg)
            np.savetxt(lpsq_file, lp_avgsq)
            np.savetxt(w_file, w_avg)
            np.savetxt(wsq_file, w_avgsq)
            np.savetxt(nlw_pop_file, nlw_pop)
            df['avgs'] = running_avgs
            df['avgsq'] = running_avgsq
            df['np_avg'] = np_avg
            df['np_avgsq'] = np_avgsq
            df['lp_avg'] = lp_avg
            df['lp_avgsq'] = lp_avgsq
            df['w_avg'] = w_avg
            df['w_avgsq'] = w_avgsq
            df['nlw_pop'] = nlw_pop
            df['best'] = best
            
            
            plot_final_best_2d_file = best_prefix + "_r{:1d}_2d.pdf".format(i)
            plot_final_best_3d_file = best_prefix + "_r{:1d}_3d.pdf".format(i)
            plot_nu_phi_file = avgs_prefix + "_r{:1d}_nu_phi.pdf".format(i)
            plots.antenna_plot_2d(best, phoenix_data, plot_final_best_2d_file)
            plots.antenna_plot_3d(best, phoenix_data, plot_final_best_3d_file)
            plots.plot_nu_phi_2(running_avgs[:, 0], running_avgs[:, 1],
                              running_avgs[:, 7], plot_nu_phi_file)
    df = pd.DataFrame(df)
    timestamp = datetime.now().isoformat('_', timespec='minutes')
    df.to_csv("out/" + timestamp + ".csv")

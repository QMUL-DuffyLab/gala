# -*- coding: utf-8 -*-
"""
06/11/2023
@author: callum
"""

import ctypes
from datetime import datetime
from collections import deque
import pandas as pd
import numpy as np
import constants
import genetic_algorithm as ga
import antenna as la
import plots
import stats

if __name__ == "__main__":
    rng = np.random.default_rng()
    df = {}

    # these will be args eventually i guess
    temps = [2300, 2600, 2800, 3300, 3700, 3800, 4300, 4400, 4800, 5800]
    for ts in reversed(temps):
    # for ts in temps:
        print("T = ", ts)
        init_type = 'radiative' # can be radiative or random
        names = ["avg", "avgsq", "np", "npsq", "lp",
                "lpsq", "w", "wsq", "best", "neg"]
        prefs = ["out/{}_{:4d}K".format(p, ts) for p in names]

        spectrum_file = constants.spectrum_prefix \
                        + '{:4d}K'.format(ts) \
                        + constants.spectrum_suffix
        # instead of unpacking and pulling both arrays, pull once
        # and split by column. this stops the arrays being strided,
        # so we can use them in the C code below without messing around
        phoenix_data = np.loadtxt(spectrum_file)
        l    = phoenix_data[:, 0]
        ip_y = phoenix_data[:, 1]

        for run in range(constants.n_runs):
            df['input_file'] = spectrum_file
            population = [None for _ in range(constants.population_size)]
            filenames = {name: pref + "_r{:1d}.dat".format(run)
                    for (name, pref) in zip(names, prefs)}
            with open(filenames['best'], "w") as f:
                pass # start a new file
            f.close()
            with open(filenames['neg'], "w") as f:
                pass # start a new file
            f.close()

            fitnesses = np.zeros(constants.population_size, dtype=np.float64)
            avgs  = np.zeros(9)
            avgsq = np.zeros(9)
            rfm = deque(maxlen=constants.conv_gen)
            np_avg   = np.zeros(constants.bounds['n_s'][1])
            lp_avg   = np.zeros(constants.bounds['n_s'][1])
            np_avgsq = np.zeros(constants.bounds['n_s'][1])
            lp_avgsq = np.zeros(constants.bounds['n_s'][1])
            nlw_pop   = np.zeros(constants.bounds['n_s'][1])
            n_s_changes = np.zeros(2)
            running_avgs = []
            running_avgsq = []
            gen = 0
            # initialise population
            for j in range(constants.population_size):
                population[j] = ga.initialise_individual(rng, init_type)

            while gen < constants.max_gen:
                avgs.fill(0.0)
                avgsq.fill(0.0)
                nlw_pop.fill(0.0)
                fitnesses.fill(0.0)
                fit_max = 0.0
                # initialise in case they all have 0 fitness
                best = population[0]
                for j, p in enumerate(population):
                    nu_phi = la.antenna(l, ip_y, p, False)
                    p.nu_e  = nu_phi[0]
                    # nu_phi[1] is the high intensity result,
                    # nu_phi[2] is the limit at low intensity
                    p.phi_f = nu_phi[2]
                    fitnesses[j] = ga.fitness(p)
                    if (fitnesses[j] > fit_max):
                        fit_max = fitnesses[j]
                        best = population[j]
                    avgs[0]  += nu_phi[0]
                    avgsq[0] += nu_phi[0]**2
                    avgs[1]  += nu_phi[1]
                    avgsq[1] += nu_phi[1]**2
                    avgs[2]  += nu_phi[2]
                    avgsq[2] += nu_phi[2]**2
                    avgs[3]  += fitnesses[j]
                    avgsq[3] += (fitnesses[j])**2
                    avgs[4]  += np.sum(p.n_p) / (p.n_s) # don't count RC
                    avgsq[4] += np.sum(np.square(p.n_p)) / (p.n_s)
                    avgs[5]  += np.sum(p.lp) / (p.n_s)
                    avgsq[5] += np.sum(np.square(p.lp)) / (p.n_s)
                    avgs[6]  += p.n_b
                    avgsq[6] += p.n_b**2
                    avgs[7]  += p.n_s
                    avgsq[7] += p.n_s**2
                    for k in range(p.n_s):
                        nlw_pop[k] += 1
                        np_avg[k]   += p.n_p[k]
                        lp_avg[k]   += p.lp[k]
                        np_avgsq[k] += p.n_p[k]**2
                        lp_avgsq[k] += p.lp[k]**2

                avgs /= constants.population_size
                avgsq /= constants.population_size
                std_dev = np.sqrt(avgsq - np.square(avgs))
                running_avgs.append(avgs)
                running_avgsq.append(avgsq)
                np_avg = np.divide(np_avg, nlw_pop, where=nlw_pop > 0.0)
                lp_avg = np.divide(lp_avg, nlw_pop, where=nlw_pop > 0.0)
                np_avgsq = np.divide(np_avgsq, nlw_pop, where=nlw_pop > 0.0)
                lp_avgsq = np.divide(lp_avgsq, nlw_pop, where=nlw_pop > 0.0)

                if (gen % constants.hist_snapshot == 0):
                    stats.hist(population, gen, run, ts)

                print("Generation {:4d}: ".format(gen))
                print("================")
                print(f"<ν_e>     = {avgs[0]:10.4n}\t<ν_e^2>       = {avgsq[0]:10.4n}\tσ = {std_dev[0]:10.4n}")
                print(f"<φ_e(γ)>  = {avgs[1]:10.4n}\t<φ_e(γ)^2>    = {avgsq[1]:10.4n}\tσ = {std_dev[1]:10.4n}")
                print(f"<φ_e>     = {avgs[2]:10.4n}\t<φ_e^2>       = {avgsq[2]:10.4n}\tσ = {std_dev[2]:10.4n}")
                print(f"f(p)      = {avgs[3]:10.4n}\tf(p)^2        = {avgsq[3]:10.4n}\tσ = {std_dev[3]:10.4n}")
                print(f"<n_p>     = {avgs[4]:10.4n}\t<n_p^2>       = {avgsq[4]:10.4n}\tσ = {std_dev[4]:10.4n}")
                print(f"<λ_p>     = {avgs[5]:10.4n}\t<λ_p^2>       = {avgsq[5]:10.4n}\tσ = {std_dev[5]:10.4n}")
                print(f"<n_b>     = {avgs[6]:10.4n}\t<n_b^2>       = {avgsq[6]:10.4n}\tσ = {std_dev[6]:10.4n}")
                print(f"<n_s>     = {avgs[7]:10.4n}\t<n_s^2>       = {avgsq[7]:10.4n}\tσ = {std_dev[7]:10.4n}")

                # check convergence
                rfm.append(fit_max)
                qs = np.array([np.abs((rfm[i] - rfm[-1]) / rfm[-1])
                      for i in range(len(rfm)- 1)])
                # print(rfm, qs, (qs < constants.conv_per))
                print("convergence trues: {:d}".format(np.count_nonzero(qs)))
                if (gen > constants.conv_gen and
                    (qs < constants.conv_per).all()):
                    print("Fitness converged at gen {}".format(gen))
                    stats.hist(population, gen, run, ts)
                    break

                survivors = ga.selection(rng, population)
                avg_nb_surv = np.sum(np.array([s.n_b
                                        for s in survivors]) / len(survivors))
                avg_ns_surv = np.sum(np.array([s.n_s
                                        for s in survivors]) / len(survivors))
                avg_fit_surv = np.sum(np.array([ga.fitness(s)
                                        for s in survivors]) / len(survivors))
                print("Average n_b, n_s, fitness of survivors: ",
                      avg_nb_surv, avg_ns_surv, avg_fit_surv)
                print("\n")

                with open(filenames['best'], "a") as f:
                    f.write(str(best))
                    f.write("\n")
                f.close()
                population = ga.reproduction(rng, survivors, population)
                for j in range(constants.population_size):
                    p = rng.random()
                    if p < constants.mu_rate:
                        population[j] = ga.mutation(rng, population[j], n_s_changes)
                gen += 1

            running_avgs = np.array(running_avgs)
            running_avgsq = np.array(running_avgsq)
            np.savetxt(filenames['avg'], running_avgs)
            np.savetxt(filenames['avgsq'], running_avgsq)
            np.savetxt(filenames['np'], np_avg)
            np.savetxt(filenames['npsq'], np_avgsq)
            np.savetxt(filenames['lp'], lp_avg)
            np.savetxt(filenames['lpsq'], lp_avgsq)
            plot_final_best_2d_file = prefs[-2] + "_r{:1d}_2d.pdf".format(run)
            plot_final_best_3d_file = prefs[-2] + "_r{:1d}_3d.pdf".format(run)
            plot_nu_phi_file = prefs[0] + "_r{:1d}_nu_phi.pdf".format(run)
            plots.plot_nu_phi_2(running_avgs[:, 0], running_avgs[:, 1],
                              running_avgs[:, 7], plot_nu_phi_file)

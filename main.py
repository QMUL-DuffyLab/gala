# -*- coding: utf-8 -*-
"""
06/11/2023
@author: callum
"""

from collections import deque
import os
import pickle
import zipfile
import numpy as np
import pandas as pd
import constants
import genetic_algorithm as ga
import antenna as la
import plots
import stats
import light

if __name__ == "__main__":
    rng = np.random.default_rng()

    costs = [0.03]
    # various other examples of dicts in light.py
    spectra_dicts = [
          {'type': "filtered", 'kwargs': {'filter': 'red'}},
          ]
    # for d in spectra_dicts:
    #     d["kwargs"]["intensity"] = 50.0
    #     d["kwargs"]["region"] = [400.0, 700.0]
    light.check(spectra_dicts)

    dist_funcs, plot_funcs, scalars = stats.stat_parameters()

    # allocate this so hopefully it doesn't allocate 1 billion times
    nu_phi = np.zeros(3, dtype=np.float64)
    init_type = 'proto' # see ga.new()
    for cost in costs:
        print(f"Cost = {cost}. Building spectra")
        # the zip's an iterator - need to rebuild it each time
        spectra_zip = light.build(spectra_dicts)
        for spectrum, out_name in spectra_zip:
            pigment_list = [*constants.bounds['rc'],
                            *constants.bounds['pigment']]
            overlaps, gammas = la.lookups(spectrum, pigment_list)
            l    = spectrum[:, 0]
            ip_y = spectrum[:, 1]
            print("Spectrum output name: ", out_name)
            outdir = os.path.join(constants.output_dir, f"cost_{cost}")
            print("Output dir: {}".format(outdir))
            # file prefix for all output files for this simulation
            prefix = os.path.join(outdir, out_name)
            os.makedirs(outdir, exist_ok=True)

            do_averages = True
            # list of files to be zipped
            zf = [ [] for _ in range(constants.n_runs)]
            for run in range(constants.n_runs):
                end_run = False
                population = [None for _ in range(constants.population_size)]
                fitnesses = np.zeros(constants.population_size)
                best_file = f"{prefix}_best_{run}.txt"
                with open(best_file, "w") as f:
                    pass # start a new file
                f.close()
                # files to add to zip
                zf[run].append(best_file)
                avgs = []
                rfm = deque(maxlen=constants.conv_gen)
                gen = 0
                gens_since_improvement = 0
                # initialise population
                for j in range(constants.population_size):
                    population[j] = ga.new(rng, init_type)

                fit_max = 0.0
                # initialise in case they all have 0 fitness
                best = population[0]
                while gen < constants.max_gen:
                    for j, p in enumerate(population):
                        nu_phi = la.antenna(l, ip_y, p,
                                            overlaps, gammas, False)
                        '''
                        note - these are set by hand, since if I'm using
                        a non-Python kernel to do the calculations it might
                        not be aware of the dataclass structure and so
                        can't update class members. might be worth adding
                        a function to wrap this though, i guess
                        '''
                        p.nu_e  = nu_phi[0]
                        # nu_phi[1] is the high intensity result,
                        # nu_phi[2] is the limit at low intensity
                        p.phi_e_g = nu_phi[1]
                        p.phi_e = nu_phi[2]
                        p.fitness = ga.fitness(p, cost)
                        fitnesses[j] = p.fitness
                        if (fitnesses[j] > fit_max):
                            fit_max = fitnesses[j]
                            best = ga.copy(population[j])
                            gens_since_improvement = 0
                    # avgs for current generation
                    ca = stats.scalar_stats(population, scalars) 
                    avgs.append(ca)
                    print(f"Run {run}, gen {gen}:")
                    for s in scalars:
                        serr = f"{s}_err"
                        print(f"<{s}> = {ca[s]} +- {ca[serr]}")
                    print("")
                    if (gen % constants.hist_snapshot == 0):
                        # bar charts/histograms of Genome parameters
                        zf[run].extend(stats.hists(population, dist_funcs,
                                             plot_funcs, prefix, run, gen))
                        # average absorption spectrum
                        avg_plot_prefix = f"{prefix}_{run}_{gen}_avg_spectrum"
                        zf[run].extend(plots.plot_average(population,
                                spectrum,
                                avg_plot_prefix,
                                xlim=constants.x_lim,
                                label=r'$ \left<A(\lambda)\right> $'))

                    # check convergence before applying GA
                    rfm.append(ca['fitness'])
                    qs = np.array([np.abs((rfm[i] - rfm[-1]) / rfm[-1])
                          for i in range(len(rfm)- 1)])
                    print("gens since improvement: {:d}".format(
                        gens_since_improvement))
                    print("")
                    if ((gen > constants.conv_gen and
                        (qs < constants.conv_per).all())
                        or gens_since_improvement > constants.conv_gen):
                        print("Converged at gen {}".format(gen))
                        break

                    try:
                        survivors = ga.selection(rng, population, fitnesses, cost)
                    except ValueError:
                        print("Resetting and trying again.")
                        end_run = True
                        do_averages = False
                        break

                    with open(best_file, "a") as f:
                        f.write(str(best).strip('\n'))
                        f.write("\n")
                    f.close()
                    population = ga.reproduction(rng, survivors, population)
                    for j in range(constants.population_size):
                        p = rng.random()
                        if p < constants.mu_rate:
                            population[j] = ga.mutation(rng, population[j])
                    gen += 1
                    gens_since_improvement += 1

                # end of run
                zf[run].extend(stats.hists(population, dist_funcs,
                                     plot_funcs, prefix, run, "final"))
                avg_plot_prefix = f"{prefix}_{run}_final_avg_spectrum"
                zf[run].extend(plots.plot_average(population,
                                spectrum,
                                avg_plot_prefix,
                                xlim=constants.x_lim,
                                label=r'$ \left<A(\lambda)\right> $'))
                df = pd.concat(avgs, axis=1).T # convert the list of Series of scalar stats to one DataFrame
                af = f"{prefix}_{run}_avgs.txt"
                df.to_csv(af, index=False)
                zf[run].append(af)
                zf.extend(plots.plot_running(af, scalars))

                # do pickle stuff and add pickled filename to zf
                pop_file = f"{outdir}/{out_name}_{run}_final_pop.dat"
                with open(pop_file, "wb") as f:
                    pickle.dump(population, f)
                zf[run].append(pop_file)

                try:
                    bestfiles = plots.plot_best(best_file, spectrum)
                    zf[run].extend(bestfiles)
                except AttributeError:
                    pass

            # end of all runs for given cost/spectrum
            # after n_runs, average over runs
            if do_averages:
                avg_files = stats.average_finals(prefix, plot_funcs, spectrum)
            for run in range(constants.n_runs):
                # do zipfile stuff - need to figure out pathnames etc
                # note that these are currently uncompressed
                zipfilename = f"{prefix}_{run}.zip"
                with zipfile.ZipFile(zipfilename, mode="w") as archive:
                    for filename in zf[run]:
                        if os.path.isfile(filename):
                            archive.write(filename,
                                    arcname=os.path.basename(filename))

            # delete the zipped files to save space/clutter
            for run in range(constants.n_runs):
                for filename in zf[run]:
                    if os.path.isfile(filename):
                        os.remove(filename)

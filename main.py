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
import constants
import genetic_algorithm as ga
import antenna as la
import plots
import stats
import light

if __name__ == "__main__":
    rng = np.random.default_rng()

    costs = [0.03, 0.02, 0.01, 0.005]
    # various other examples of dicts in light.py
    spectra_dicts = [
          # {'type': "am15", 'kwargs': {'dataset': "tilt"}},
          # {'type': "marine", 'kwargs': {'depth': 1.0}},
          # {'type': "marine", 'kwargs': {'depth': 2.5}},
          # {'type': "marine", 'kwargs': {'depth': 5.0}},
          # {'type': "filtered", 'kwargs': {'filter': "red"}},
          # {'type': "filtered", 'kwargs': {'filter': "far-red", 'fraction': 0.85}},
          # {'type': "filtered", 'kwargs': {'filter': "far-red", 'fraction': 0.95}},
          {'type': "phoenix", 'kwargs': {'temperature': 2300}},
          {'type': "phoenix", 'kwargs': {'temperature': 2600}},
          {'type': "phoenix", 'kwargs': {'temperature': 2800}},
          {'type': "phoenix", 'kwargs': {'temperature': 3300}},
          {'type': "phoenix", 'kwargs': {'temperature': 3700}},
          {'type': "phoenix", 'kwargs': {'temperature': 3800}},
          ]
    # for d in spectra_dicts:
    #     d["kwargs"]["intensity"] = 50.0
    #     d["kwargs"]["region"] = [400.0, 700.0]
    light.check(spectra_dicts)

    # allocate this so hopefully it doesn't allocate 1 billion times
    nu_phi = np.zeros(3, dtype=np.float64)
    init_type = 'proto' # can be proto (single branch & block) or random
    names = ["avg", "avgsq", "best"]
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
            outdir = os.path.join(constants.output_dir,
                    "cost_{}".format(cost))
            print("Output dir: {}".format(outdir))
            prefs = ["{}/{}_{}".format(outdir,
                p, out_name) for p in names]
            os.makedirs(outdir, exist_ok=True)

            do_averages = True
            do_best_avg = False
            # list of files to be zipped
            zf = [ [] for _ in range(constants.n_runs)]
            for run in range(constants.n_runs):
                end_run = False
                population = [None for _ in range(constants.population_size)]
                filenames = {name: pref + "_r{:1d}.dat".format(run)
                        for (name, pref) in zip(names, prefs)}
                with open(filenames['best'], "w") as f:
                    pass # start a new file
                f.close()
                # files to add to zip
                zf[run].extend(list(filenames.values()))

                fitnesses = np.zeros(constants.population_size, dtype=np.float64)
                avgs  = np.zeros(9)
                avgsq = np.zeros(9)
                rfm = deque(maxlen=constants.conv_gen)
                n_s_changes = np.zeros(2)
                running_avgs = []
                running_avgsq = []
                gen = 0
                gens_since_improvement = 0
                # initialise population
                for j in range(constants.population_size):
                    population[j] = ga.new(rng, init_type)
                            # antenna=constants.Genome(4, 2, 
                            #     [76, 75], [0.0, 0.0], ['apc', 'pc']),
                            # variability=0.0)

                fit_max = 0.0
                # initialise in case they all have 0 fitness
                best = population[0]
                best_pref = os.path.splitext(filenames['best'])[0]
                while gen < constants.max_gen:
                    avgs.fill(0.0)
                    avgsq.fill(0.0)
                    fitnesses.fill(0.0)
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
                        avgs[0]  += nu_phi[0]
                        avgsq[0] += nu_phi[0]**2
                        avgs[1]  += nu_phi[1]
                        avgsq[1] += nu_phi[1]**2
                        avgs[2]  += nu_phi[2]
                        avgsq[2] += nu_phi[2]**2
                        avgs[3]  += fitnesses[j]
                        avgsq[3] += (fitnesses[j])**2
                        avgs[4]  += p.n_b * np.sum(p.n_p) # don't count RC
                        avgsq[4] += (p.n_b * np.sum(p.n_p))**2
                        avgs[5]  += p.n_b * p.n_s
                        avgsq[5] += (p.n_b * p.n_s)**2
                        avgs[6]  += p.n_b
                        avgsq[6] += p.n_b**2
                        avgs[7]  += p.n_s
                        avgsq[7] += p.n_s**2

                    avgs /= constants.population_size
                    avgsq /= constants.population_size
                    std = np.sqrt(avgsq - np.square(avgs))
                    running_avgs.append(np.copy(avgs))
                    running_avgsq.append(np.copy(avgsq))

                    if (gen % constants.hist_snapshot == 0):
                        histfiles = stats.hist(population, gen, 
                                            run, outdir, out_name)
                        zf[run].extend(histfiles)
                        zf[run].extend([os.path.splitext(f)[0] 
                                        + ".pdf" for f in histfiles])
                        plots.hist_plot(*histfiles)
                        avg_plot_prefix = "{}_{:04d}_r{:1d}_spectrum".format(prefs[0], gen, run)
                        zf[run].append(f"{avg_plot_prefix}.dat")
                        zf[run].append(f"{avg_plot_prefix}.pdf")
                        plots.plot_average(population, spectrum,
                                avg_plot_prefix,
                                xlim=constants.x_lim,
                                label=r'$ \left<A(\lambda)\right> $')

                    # f""" looks horrible but keeps it under 80 chars
                    print(f"""{out_name}, run {run:1d}, cost {cost}, gen {gen}:
====================================================
<ν_e>    = {avgs[0]:10.4n}\t<ν_e^2>    = {avgsq[0]:10.4n}\tσ = {std[0]:10.4n}
<φ_e(γ)> = {avgs[1]:10.4n}\t<φ_e(γ)^2> = {avgsq[1]:10.4n}\tσ = {std[1]:10.4n}
<φ_e>    = {avgs[2]:10.4n}\t<φ_e^2>    = {avgsq[2]:10.4n}\tσ = {std[2]:10.4n}
<f(p)>   = {avgs[3]:10.4n}\t<f(p)^2>   = {avgsq[3]:10.4n}\tσ = {std[3]:10.4n}
<n_p>    = {avgs[4]:10.4n}\t<n_p^2>    = {avgsq[4]:10.4n}\tσ = {std[4]:10.4n}
<n_b n_s>= {avgs[5]:10.4n}\t<n_b n_s^2>= {avgsq[5]:10.4n}\tσ = {std[5]:10.4n}
<n_b>    = {avgs[6]:10.4n}\t<n_b^2>    = {avgsq[6]:10.4n}\tσ = {std[6]:10.4n}
<n_s>    = {avgs[7]:10.4n}\t<n_s^2>    = {avgsq[7]:10.4n}\tσ = {std[7]:10.4n}
""")

                    # check convergence
                    rfm.append(np.mean(fitnesses))
                    qs = np.array([np.abs((rfm[i] - rfm[-1]) / rfm[-1])
                          for i in range(len(rfm)- 1)])
                    print("convergence trues: {:d}".format(
                        np.count_nonzero((qs < constants.conv_per))))
                    print("gens since improvement: {:d}".format(
                        gens_since_improvement))
                    print("\n")
                    if ((gen > constants.conv_gen and
                        (qs < constants.conv_per).all())
                        or gens_since_improvement > constants.conv_gen):
                        print("Fitness converged at gen {}".format(gen))
                        histfiles = stats.hist(population, gen, run, 
                                            outdir, out_name)
                        plots.hist_plot(*histfiles)
                        zf[run].extend(histfiles)
                        zf[run].extend([os.path.splitext(f)[0] 
                                        + ".pdf" for f in histfiles])
                        break

                    try:
                        survivors = ga.selection(rng, population, fitnesses, cost)
                    except ValueError:
                        print("Resetting and trying again.")
                        end_run = True
                        do_averages = False
                        break

                    with open(filenames['best'], "a") as f:
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

                if end_run:
                    # for some reason run isn't incremented on continue?
                    # so do it manually
                    run += 1
                    continue

                # if the run ended normally, plot averages
                running_avgs = np.array(running_avgs)
                running_avgsq = np.array(running_avgsq)
                np.savetxt(filenames['avg'], running_avgs)
                np.savetxt(filenames['avgsq'], running_avgsq)
                plot_nu_phi_file = prefs[0] + "_r{:1d}_nu_phi.pdf".format(run)
                zf[run].append(plot_nu_phi_file)
                plots.plot_nu_phi_2(running_avgs[:, 0], running_avgs[:, 2],
                                    running_avgs[:, 3], running_avgs[:, 6],
                                    running_avgs[:, 7], plot_nu_phi_file)

                # do pickle stuff and add pickled filename to zf
                pop_file = f"{outdir}/{out_name}_{run}_final_pop.dat"
                with open(pop_file, "wb") as f:
                    pickle.dump(population, f)
                zf[run].append(pop_file)

                # call julia_plot and antenna_spectra
                try:
                    bestfiles = plots.plot_best(filenames['best'], spectrum)
                    zf[run].extend(bestfiles)
                    print(bestfiles)
                except AttributeError:
                    do_best_avg = False

                avg_out_pref = prefs[0] + "_r{:1d}_spectrum".format(run)
                zf[run].extend([avg_out_pref + ".pdf", avg_out_pref + ".dat"])
                plots.plot_average(population, spectrum,
                        avg_out_pref,
                        xlim=constants.x_lim,
                        label=r'$\left<A(\lambda)\right>$')

            # end of all runs for given cost/spectrum
            # after n_runs, average the best antennae and plot absorption
            if do_averages:
                stats.average_antenna(outdir, spectrum, out_name)
                if do_best_avg:
                    plots.plot_average_best(outdir, spectrum,
                            out_name, cost, xlim=constants.x_lim)
            for run in range(constants.n_runs):
                # do zipfile stuff - need to figure out pathnames etc
                # note that these are currently uncompressed
                zipfilename = f"{outdir}/{out_name}_{run}.zip"
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

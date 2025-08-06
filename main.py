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
import solvers
import plots
import stats
import light
import utils
import genetic_algorithm as ga

if __name__ == "__main__":
    rng = np.random.default_rng()

    cost = 0.01 # temp
    # various other examples of dicts in light.py
    spectra_dicts = []
    for Tstar, Lstar, Rstar in light.phz_stars:
        radii = light.calculate_phz_radii(Tstar, Lstar)
        print(f"Tstar = {Tstar}, radii = {radii}")
        for a in radii:
            spectra_dicts.append(
            {'type': 'stellar', 'kwargs':
             {'Tstar': Tstar, 'Rstar': Rstar,
              'a': a, 'attenuation': 0.0}},
          )
    light.check(spectra_dicts)

    init_kwargs = {'n_b': 1, 'n_s': 1} # see ga.new()
    solver_kwargs = {'diff_ratios': {'ox': 0.0, 'anox': 0.0}}
    spectra_zip = light.build(spectra_dicts)
    for spectrum, out_name in spectra_zip:
        print("Spectrum output name: ", out_name)
        rc_nu_e = {rct: solvers.RC_only(rct, spectrum)[0] # nu_e
                for rct in ga.genome_parameters['rc']['bounds']}
        # lookup tables
        gammas, overlaps = utils.lookups(spectrum)
        # NB!!! diff ratios not considered in the hash table!!!
        hash_table = utils.get_hash_table(out_name)
        rc_dir = "_".join(ga.genome_parameters['rc']['bounds'])
        outdir = os.path.join(constants.output_dir, rc_dir,
        f"cost_{cost}", out_name)
        print(f"Output dir: {outdir}")
        os.makedirs(outdir, exist_ok=True)
        hash_table_finds = 0
        solver_failures = 0

        do_averages = True
        # list of files to be zipped
        zf = [ [] for _ in range(constants.n_runs)]
        for run in range(constants.n_runs):
            end_run = False
            # initialise population
            population = [ga.new(rng, **init_kwargs)
                    for _ in range(constants.population_size)]
            best_file = os.path.join(f"{outdir}", f"run_{run}_best.txt")
            with open(best_file, "w") as f:
                pass # start a new file
            f.close()
            # files to add to zip
            zf[run].append(best_file)
            avgs = []
            rfm = deque(maxlen=constants.conv_gen)
            gen = 0
            gens_since_improvement = 0
            fit_max = 0.0
            # initialise in case they all have 0 fitness
            best = population[0]
            avgs = {k: [] for k in stats.minimal_stats.keys()}
            while gen < constants.max_gen:
                results = {'nu_e': [], 'nu_cyc': [], 'fitness': [],
                        'redox': [], 'recomb': []}
                for j, p in enumerate(population):
                    # NB: solver should return a dict now
                    # this feels horrible to me. but some of the
                    # return values are arrays of different sizes, so
                    # we can't just numpy array the whole thing
                    res = hash_table.get(ga.genome_hash(p))
                    if res == None:
                        res, fail = solvers.antenna_RC(p,
                                spectrum, **solver_kwargs)
                        if not fail:
                            hash_table[ga.genome_hash(p)] = res
                    else:
                        hash_table_finds += 1
                    for k, v in res.items():
                        results[k].append(v)
                    if fail < 0:
                        solver_failures += 1
                    fitness = ga.fitness(p, res['nu_e'],
                            cost, rc_nu_e[p.rc])
                    results['fitness'].append(fitness)
                    if fitness > fit_max:
                        fit_max = fitness
                        best = ga.copy(p)
                        gens_since_improvement = 0
                # avgs for current generation
                df = pd.DataFrame(population)
                for k, v in results.items():
                    df[k] = v

                # outdir here is unused
                stat_dict, stat_files = stats.do_stats(df, spectrum,
                        outdir, **stats.minimal_stats)
                '''
                this is a hack, honestly. separate out the RC counts
                so that we can make them all into a nice dataframe
                later. there's gotta be some way of fixing this in
                stats.py, maybe by messing with the return values
                of counts()
                '''
                for k, v in stat_dict.items():
                    if k == 'rc':
                        if k in avgs:
                            del avgs[k]
                        # rc returns a tuple of arrays (values, counts)
                        (values, counts) = v
                        for value, count in zip(values, counts):
                            if value in avgs:
                                avgs[value].append(count)
                            else:
                                avgs[value] = [count]
                    else:
                        (value, err) = v
                        avgs[k].append(value)
                        if f"{k}_err" in avgs:
                            avgs[f"{k}_err"].append(err)
                        else:
                            avgs[f"{k}_err"] = [err]
                print(f"Run {run}, gen {gen}:")
                for key in stat_dict:
                    print(f"<{key}> = {stat_dict[key]}")
                print(f"solver failures: {solver_failures}")
                print(f"hash table hits: {hash_table_finds}")
                print("")
                if gen % constants.hist_snapshot == 0:
                    # bar charts/histograms of Genome parameters
                    stat_pref = os.path.join(f"{outdir}",
                                             f"run_{run}_gen_{gen}")
                    # they all create output files but nothing to print,
                    # so ignore the tuple returned by do_stats
                    pop_file = f"{stat_pref}_population.csv"
                    df.to_csv(pop_file)
                    _, stat_files = stats.do_stats(df,
                     spectrum, prefix=stat_pref, **stats.big_stats)
                    zf[run].append(pop_file)
                    zf[run].extend(stat_files)

                with open(best_file, "a") as f:
                    f.write(str(best).strip('\n'))
                    f.write("\n")
                f.close()

                # check convergence before applying GA
                rfm.append(stat_dict['fitness'][0])
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
                    population = ga.evolve(rng, population,
                            np.array(results['fitness']), cost)
                except ValueError:
                    print("Selection failed. Resetting")
                    end_run = True
                    do_averages = False
                    raise
                    break

                gen += 1
                gens_since_improvement += 1

            # end of run
            stat_pref = os.path.join(f"{outdir}", f"run_{run}_final")
            output, ofs = stats.do_stats(df,
                spectrum, prefix=stat_pref, **stats.big_stats)
            zf[run].extend(ofs)
            # arrays might be different lengths; make each into Series
            avg_df = pd.DataFrame({k:pd.Series(v) for k, v in avgs.items()})
            af = os.path.join(f"{outdir}", f"run_{run}_avgs.csv")
            avg_df.to_csv(af, index=False)
            zf[run].append(af)

            # do pickle stuff and add pickled filename to zf
            pop_file = os.path.join(f"{outdir}",
                            f"{run}_final_population.csv")
            df.to_csv(pop_file)

            try:
                bestfiles = plots.plot_best(best_file, spectrum)
                zf[run].extend(bestfiles)
            except AttributeError:
                pass

        # end of all runs
        # save hash table
        print(f"Hash table finds: {hash_table_finds}")
        htf = os.path.join("tables", f"{out_name}.pkl")
        with open(htf, "wb") as f:
            pickle.dump(hash_table, f)

        for run in range(constants.n_runs):
            zipfilename = os.path.join(f"{outdir}",
                            f"run_{run}.zip")
            with zipfile.ZipFile(zipfilename,
                    mode="w", compression=zipfile.ZIP_BZIP2) as archive:
                for filename in zf[run]:
                    print(filename)
                    if os.path.isfile(filename):
                        archive.write(filename,
                                arcname=os.path.basename(filename))

        # delete the files we've just zipped up to save space/clutter
        for run in range(constants.n_runs):
            for filename in zf[run]:
                if os.path.isfile(filename):
                    os.remove(filename)

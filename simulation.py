# -*- coding: utf-8 -*-
"""
07/10/2025
@author: callum
"""

from collections import deque
import os
import pickle
import zipfile
import hashlib
import argparse
import numpy as np
import pandas as pd
import constants
import solvers
import stats
import light
import utils
import genetic_algorithm as ga
import rc as rcm

def do_simulation(spectrum_file, cost, rcs, anox_diff_ratio, rng_seed=None):
    spectrum, out_name = light.load_spectrum(spectrum_file)
    print("Spectrum output name: ", out_name)
    outdir = os.path.dirname(spectrum_file)
    print(f"Output dir: {outdir}")

    init_kwargs = {'n_b': 1, 'n_s': 1}
    solver_kwargs = {'diff_ratios': {'ox': 0.0, 'anox': anox_diff_ratio}}
    rc_nu_e = {rct: solvers.RC_only(rct, spectrum)[0] for rct in rcs}
    hash_table = utils.get_hash_table(outdir)
    os.makedirs(outdir, exist_ok=True)
    hash_table_finds = 0
    solver_failures = 0

    if rng_seed is not None:
        seed = rng_seed
    else:
        h = hashlib.shake_128(bytes(spectrum_file, encoding="utf-8"))
        seed = int(h.hexdigest(16), 16) ^ constants.entropy
    # write it out for reproducibility!
    with open(os.path.join(outdir, "seed.txt"), "w") as f:
        f.write(str(seed))

    do_averages = True
    # list of files to be zipped
    zf = [ [] for _ in range(constants.n_runs)]
    for run in range(constants.n_runs):
        sequence = np.random.SeedSequence((seed, run))
        print(f"run = {run}, seed sequence = {sequence}")
        rng = np.random.default_rng(sequence)
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
                    res = solvers.antenna_RC(p,
                            spectrum, **solver_kwargs)
                    hash_table[ga.genome_hash(p)] = res
                else:
                    hash_table_finds += 1
                for k, v in res.items():
                    results[k].append(v)
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
                # make into a series so that if we're not
                # calculating redox, it just fills the redox
                # and recomb columns with NAN missing values
                df[k] = pd.Series(v, dtype=np.float64)

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
                zf[run].append(pop_file)
                
                if do_stats:
                    _, stat_files = stats.do_stats(df,
                     spectrum, prefix=stat_pref, **stats.big_stats)
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

    # end of all runs
    # save hash table
    print(f"Hash table finds: {hash_table_finds}")
    utils.save_hash_table(hash_table, outdir)

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
    return zf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="run GA simulations",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required arguments
    parser.add_argument('-s', '--spectrum_file', type=str, required=True,
            help='Path to file with spectrum data (created by light.py)')
    # optional arguments
    # (spectrum_file, cost, rcs, anox_diff_ratio, rng_seed=None):
    parser.add_argument('-c', '--cost', type=float, default=0.01,
            help="Cost parameter. Default is 0.01")
    parser.add_argument('-rc', '--rc_types', type=str, nargs='+',
            default=ga.genome_parameters['rc']['bounds'],
            choices=rcm.params.keys(),
            help='''
Reaction centre types to use in the simulation. Possible options are listed
in rc.py; default is to take whatever's currently in ga.genome_parameters
for the "rc" parameter. For investigating the effect of substrate availability
you'll want to set this to "anox", either in genetic_algorithm.py or here.
''')
    parser.add_argument('-ad', '--anox_diffusion', type=float,
            default=0.0,
            help='''
Diffusion ratio for anoxygenic oxidation substrate.
For more information on this and the implementation see solvers.py, but
basically it affects the oxidation time for anoxygenic photosynthesis
according to the availability of the substrate.
''')
    parser.add_argument('-rng', '--rng_seed', type=int, default=None,
            help='''
RNG seed to use. Default is None; in this case, a seed will be generated
using constants.entropy XORd with the MD5 hash of the spectrum filename,
with the run number added for each separate run, in order to ensure that
each run for each set of parameters has a different seed, but that they
are reproducible. the XOR of constants.entropy with the MD5 hash will be
saved to a file so you can pass that and reproduce the runs.
''')
    args = parser.parse_args()
    do_simulation(args.spectrum_file, args.cost, args.rc_types,
            args.anox_diffusion, args.rng_seed)

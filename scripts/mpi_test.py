from collections import deque
import os
import pickle
import zipfile
import hashlib
import argparse
import itertools
import numpy as np
import pandas as pd
import constants
import solvers
import stats
import light
import utils
import genetic_algorithm as ga
import rc as rcm
from mpi4py import MPI


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    pop_size = constants.population_size // size
    print(size, rank, pop_size)

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
    parser.add_argument('--stats', action=argparse.BooleanOptionalAction,
            default=True, help='''
If stats is passed, the big set of stats (average shifts per subunit,
histograms of various quantities, etc.) will be done. 
if --no-stats is passed, they will not. This means that no plots will be
and the zip files will be much smaller, only containing the snapshots
of the population in dataframes. there for apocrita mostly.
''')
    args = parser.parse_args()

    spectrum, out_name = light.load_spectrum(args.spectrum_file)
    outdir = os.path.dirname(args.spectrum_file)
    if rank == 0:
        print("Spectrum output name: ", out_name)
        print(f"Output dir: {outdir}")

    init_kwargs = {'n_b': 1, 'n_s': 1}
    solver_kwargs = {'diff_ratios': {'ox': 0.0, 'anox': args.anox_diffusion}}
    rc_nu_e = {rct: solvers.RC_only(rct, spectrum, **solver_kwargs)[0]
            for rct in args.rc_types}
    hash_table = utils.get_hash_table(outdir)
    os.makedirs(outdir, exist_ok=True)
    hash_table_finds = 0
    solver_failures = 0

    if args.rng_seed is not None:
        seed = args.rng_seed
    else:
        h = hashlib.shake_128(bytes(args.spectrum_file, encoding="utf-8"))
        seed = int(h.hexdigest(16), 16) ^ constants.entropy
    # write it out for reproducibility!
    with open(os.path.join(outdir, "seed.txt"), "w") as f:
        f.write(str(seed))

    do_averages = True
    # list of files to be zipped
    zf = [ [] for _ in range(constants.n_runs)]
    for run in range(constants.n_runs):
        sequence = np.random.SeedSequence((seed, size * run + rank))
        print(f"run = {run}, rank {rank}, seed sequence = {sequence}")
        rng = np.random.default_rng(sequence)
        end_run = False
        population = [ga.new(rng, **init_kwargs)
                    for _ in range(pop_size)]
        nu_e = np.zeros(pop_size, dtype=np.float64)
        nu_cyc = np.zeros_like(nu_e)
        fitness = np.zeros_like(nu_e)
        for i in range(size):
            if rank == i:
                print(f"Rank {i} has population length {len(population)}.")
        gen = 0
        # while gen < constants.max_gen:
        while gen < 1:
            for j, p in enumerate(population):
                # NB: solver should return a dict now
                # this feels horrible to me. but some of the
                # return values are arrays of different sizes, so
                # we can't just numpy array the whole thing
                res = hash_table.get(ga.genome_hash(p))
                if res == None:
                    res = solvers.antenna_RC(p,
                            spectrum, **solver_kwargs)
                else:
                    hash_table_finds += 1
                nu_e[j] = res['nu_e']
                nu_cyc[j] = res['nu_cyc']
                fitness[j] = ga.fitness(p, res['nu_e'],
                        args.cost, rc_nu_e[p.rc])
                print(rank, j, nu_e[j], fitness[j])

            # whole population done. wait for all ranks to finish
            print(f"Rank {rank} has finished")
            comm.Barrier()

            if rank == 0:
                all_pop = comm.gather(population, root=0)
                all_nu_e = np.zeros(constants.population_size,
                        dtype=np.float64)
                all_nu_cyc = np.zeros_like(all_nu_e)
                all_fitness = np.zeros_like(all_nu_e)
                all_pop = comm.gather(population, root=0)
                comm.Gather(nu_e, all_nu_e, root=0)
                comm.Gather(nu_cyc, all_nu_cyc, root=0)
                comm.Gather(fitness, all_fitness, root=0)
                results = {'nu_e': all_nu_e,
                         'nu_cyc': all_nu_cyc,
                        'fitness': all_fitness
                        }
                print(len(all_pop))
                print(len(results['nu_e']))
            gen += 1


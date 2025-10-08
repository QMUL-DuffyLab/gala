# -*- coding: utf-8 -*-
"""
06/11/2023
@author: callum
"""
import os
import argparse
import constants
import light
import simulation
# these two are only needed for the rc_types arg below. get rid somehow?
import genetic_algorithm as ga
import rc as rcm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="set up and run GA simulations",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--setup_only', action=argparse.BooleanOptionalAction,
            help='''
If setup_only is given, main.py will just loop over the set of spectrum
parameters it's been given, create the relevant directory structure and
generate the spectrum files, create a list of the spectrum files that
it's generated and then write that to a file and exit. This is for use
on HPC clusters so you can then use an array job to run them all.
''')
    # these parameters are reproduced in simulation.py; i don't really
    # like that as a solution but it works for now and allows you to
    # call simulation.py on its own for a single simulation, which
    # functionality i need for apocrita, it's how the array jobs will work
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
    path = os.path.join(os.getcwd(), constants.output_dir,
            "_".join(args.rc_types), f"cost_{args.cost}",
            f"anox_diffusion_{args.anox_diffusion}")
    os.makedirs(path, exist_ok=True)

    spectra_dicts = []
    for Tstar, Lstar, Rstar in light.phz_stars:
        radii = light.calculate_phz_radii(Tstar, Lstar, n_radii=2)
        print(f"Tstar = {Tstar}, radii = {radii}")
        for a in radii:
            spectra_dicts.append(
            {'type': 'stellar', 'kwargs':
             {'Tstar': Tstar, 'Rstar': Rstar,
              'a': a, 'attenuation': 0.0,
              "output_dir": path}},
          )
    filelist = []
    for sp in spectra_dicts:
        filelist.append(light.spectrum_setup(sp['type'], **sp['kwargs']))
    print("Writing files:")
    with open(os.path.join(path, "list_of_files.txt"), "w") as f:
        for fi in filelist:
            print(fi)
            f.write(f"{fi}\n")

    if not args.setup_only:
        for f in filelist:
            simulation.do_simulation(f, args.cost, args.rc_types,
                args.anox_diffusion, args.rng_seed)

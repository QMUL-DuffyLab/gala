'''
author: @callum
stats.py - some functions to do statistics, both on the
genomes and the outputs. not finished yet. need to think a little more
about how this is done

todo: continue adding write/outfile args for the averaging functions
also note that the way these are written will require
changing how the list of files to zip is generated;
currently i think it adds them based on the return
value of the stat functions. can't remember why i did
it that way, to be honest, but pretty sure that's it
'''

import os
import re
import glob
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
import constants
import plots
import genetic_algorithm as ga
import rc as rcm

def counts(arr, outfile=None):
    '''
    get counts and values of a parameter.
    will work for string or int variables
    '''
    if isinstance(arr[0], np.ndarray):
        np.array(arr).flatten()
    v, c = np.unique(arr, return_counts=True, equal_nan=False)
    if outfile is not None:
        s = pd.Series(c / len(arr), index=v)
        s.to_csv(outfile)
        return outfile
    else:
        return v, c

def avg(arr, outfile=None):
    '''
    return average and error of quantity.
    will work for int/float
    '''
    if isinstance(arr[0], np.ndarray):
        np.array(arr).flatten()
    m, e = np.mean(arr), np.std(arr) / np.sqrt(len(arr))
    if outfile is not None:
        np.savetxt(outfile, np.column_stack((m, e)))
        return outfile
    else:
        return m, e

def element_avg(arr, outfile=None):
    '''
    return element-wise average and error of int/float array
    '''
    m, e = np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(len(arr))
    if outfile is not None:
        np.savetxt(outfile, np.column_stack((m, e)))
        return outfile
    else:
        return m, e

def hist(arr, outfile=None):
    '''
    NB: this will not work. actually, figuring out what kind of
    histogram we want (one for a single int or float, one for a
    flattened array, one per element for an array, what about the
    string variables, and so on) is really hard.
    NB: if it's a genome parameter, pass the subpopulation of genomes
    directly because we need to check if it's a subunit parameter or
    an RC one. otherwise just pass the array
    '''
    if name in ga.genome_parameters.keys():
        hist_arr = np.array([getattr(p, name) for p in arr])
        b = ga.genome_parameters[name]['bounds']
        # if [ga.genome_parameters[name]['size'](p) 
        #         == getattr(p, 'n_s')] for p in arr]:
    else:
        hist_arr = np.array(arr)
        b = [np.min(hist_arr), np.max(hist_arr)]
    if isinstance(b[0], (int, np.int64)):
        # implictly set binwidth to 1
        bins = np.linspace(*b,
        num = np.round(1 + b[1] - b[0]).astype(int))
    else:
        bins = np.linspace(*b)
    try:
        if isinstance(hist_arr[0], np.ndarray):
            # is it a subunit based array?
            h = np.zeros((*hist_arr[0].shape, len(bins)-1))
        h = np.histogram(hist_arr, bins=bins)[0]
    except TypeError:
        print(f"{name}, {b}, {bins}")
        raise
    return bins, h

minimal_stats = {'nu_e': avg,
                 'fitness': avg,
                 'n_b': avg,
                 'n_s': avg,
                 'rc': counts}

# note - to be really general, we could have something like
# big_stats = {'nu_e': {'function': avg, 'split': True, 'split_on': 'rc'}
big_stats = {'split_on': 'rc',
        'rc': counts,
        'nu_e': avg,
        'fitness': avg,
        'redox': element_avg,
        'recomb': element_avg,
        'absorption': plots.plot_average}

def do_stats(population, results, spectrum, prefix=None, **kwargs):
    # note: this assumes that the key you split on is always
    # something in the genome. i think that's reasonable
    output = {}
    for key in kwargs:
        if key in ga.genome_parameters.keys():
            arr = [getattr(p, key) for p in population]
        elif key == 'split_on':
            arr = [getattr(p, kwargs[key]) for p in population]
        elif key == 'absorption':
            arr = population
        else:
            arr = results[key]

        if 'split_on' in kwargs:
            split_key = kwargs['split_on']
            if kwargs[key] == split_key:
                # don't split it on itself, that's stupid
                of = f"{prefix}_{split_key}.txt"
                output[key] = kwargs[kwargs[key]](arr, outfile=of)
            else:
                b = ga.genome_parameters[split_key]['bounds']
                subarr = [[] for i in range(len(b))]
                ofs = []
                for j in range(len(b)):
                    ofs.append(f"{prefix}_{split_key}_{b[j]}_{key}.txt")
                    for i in range(len(arr)):
                        if getattr(population[i], split_key) == b[j]:
                            subarr[j].append(arr[i])
            if key == 'absorption':
                # requires spectrum argument. would be really annoying
                # and require rewriting a load of things to remove that
                output[key] = {b[j]:
                        kwargs[key](subarr[j], spectrum,
                            ofs[j]) for j in range(len(b))}
            elif kwargs[key] != split_key:
                output[key] = {b[j]:
                        kwargs[key](subarr[j],
                            outfile=ofs[j]) for j in range(len(b))}
        else:
            output[key] = kwargs[key](arr, prefix)
    return output
            
def stat_parameters():
    '''
    for an instance of dataclass Genome, return the set of
    distribution functions which should be used for each field
    and the list of scalar parameters to calculate means and errors for.
    also return the plot function which should be used to plot them.

    parameters
    ----------

    outputs
    -------

    dist_functions: dict of functions for calculating distributions
    scalars: list of strings corresponding to scalar variables

    Note that all the functions in dist_functions should take the same
    arguments: (population, name), where name is the name of the parameter.
    '''
    dist_functions = {}
    plot_functions = {}
    scalars = []
    g = ga.Genome() # i think we need to make one to do isinstance below
    for field in dataclasses.fields(ga.Genome):
        attr = getattr(g, field.name)
        if isinstance(attr, np.ndarray):
            dist_functions[field.name] = array_dist
            if field.name == "pigment":
                plot_functions[field.name] = plots.pigment_bar
            else:
                plot_functions[field.name] = plots.plot3d
        elif field.type == int:
            dist_functions[field.name] = int_dist
            plot_functions[field.name] = plots.plot_bar
            scalars.append(field.name)
        elif field.type == float:
            dist_functions[field.name] = float_dist
            plot_functions[field.name] = plots.plot_bar
            scalars.append(field.name)
        elif field.type == str:
            dist_functions[field.name] = string_dist
            plot_functions[field.name] = plots.plot_bar
        else:
            dist_functions[field.name] = None # only other one is the connected param for radial transfer
            plot_functions[field.name] = None
            print(f"Warning: Genome field {field.name} has type {field.type} and no dist/plot function defined")
    return dist_functions, plot_functions, scalars

def scalar_stats(population, scalars):
    '''
    calculate the mean and standard error of scalar Genome parameters.

    parameters
    ----------
    population: a list of Genomes
    scalars: names of scalar parameters (list of strings)

    outputs
    -------
    d: pandas Series. index = scalars, data = (mean, error)
    '''
    d = {}
    for s in scalars:
        arr = np.array([getattr(p, s) for p in population])
        d[s] = np.mean(arr)
        d[f"{s}_err"] = np.std(arr) / np.sqrt(constants.population_size)
    return pd.Series(d)

def array_dist(arr, name, outfile):
    '''
    calculate per-subunit histogram for per-subunit quantities like
    the pigment type and number of pigments.

    parameters
    ----------

    population: a list of Genomes
    name: name of the Genome parameter
    outfile: output filename

    outputs
    -------

    To do: figure a way to deal with the pigment "histogram"
    '''
    ntype = arr[0].dtype
    hist = np.zeros((constants.hist_sub_max, constants.population_size), dtype=ntype)
    for j, p in enumerate(arr):
        for i in range(constants.hist_sub_max):
            if i < len(p):
                hist[i][j] = p[i]
    hl = []
    if ntype == "U10":
        bins = list(ga.genome_parameters[name]['bounds'])
        for i in range(constants.hist_sub_max):
            ap = pd.Series(hist[i])
            # if all genomes are below a certain size, then
            # value_counts above that size will only return '';
            # we don't want to count that, so check for it here
            if set(ap.values) != {""}:
                av = ap.value_counts().reindex(bins, fill_value=0)[bins]
                hl.append(av.to_numpy() / constants.population_size)
            else:
                hl.append(np.zeros(len(bins)))
        hist = pd.DataFrame(np.transpose(hl), index=bins)
        hist.to_csv(outfile)
    else:
        bounds = constants.bounds[name]
        if name in constants.binwidths.keys():
            bw = constants.binwidths[name]
            bins = np.linspace(*bounds,
            num=np.round(1 + (bounds[1] - bounds[0])/bw).astype(int))
        else:
            print(f"Field {name} has no binwidth given in constants.binwidths. Using default 50 bins")
            bins = np.linspace(*bounds) # 50 bins
        for i in range(constants.hist_sub_max):
            h = np.histogram(arr[i], bins=bins)[0]
            hl.append(h / constants.population_size)
        df = pd.DataFrame(np.transpose(hl), index=bins[:-1])
        df.to_csv(outfile)
    return

def hists(population, hist_functions, plot_functions, prefix, run, gen,
        do_plots=True):
    '''
    calculate distributions of parameters based on type.
    for float and array parameters, do histograms, otherwise

    parameters
    ----------
    population: a list of Genomes
    dist_functions: dict of distribution functions for Genome parameters
    plot_functions: dict of plot functions for Genome parameters
    prefix: prefix for output files (directory etc)
    run: run number
    gen: generation number

    outputs
    -------
    outfiles: list of output files generated
    
    '''
    outfiles = []
    '''
    we want to split the stats by RC type; they're doing different kinds of
    photosynthesis and we want to investigate how those perform relative to
    each other, so generate a set of subpopulations and run stats on those
    '''
    n_rc_types = len(constants.bounds["rc"])
    sublists = [[] for i in range(n_rc_types)]
    for i in range(len(population)):
        for j in range(n_rc_types):
            if population[i].rc == constants.bounds["rc"][j]:
                sublists[j].append(population[i])
    subpops = {constants.bounds["rc"][j]:
                      sublists[j] for j in range(n_rc_types)}
    for field in dataclasses.fields(constants.Genome):
        if hist_functions[field.name] is not None:
            for rc_type, subpop in subpops.items():
                outfile = f"{prefix}_{rc_type}_{field.name}_{run}_{gen}.txt"
                hist_functions[field.name](subpop, field.name, outfile)
                outfiles.append(outfile)
                if do_plots:
                    plotfile = plot_functions[field.name](outfile, field.name)
                    outfiles.append(plotfile)
    return outfiles

def average_finals(prefix, plot_functions, spectrum):
    '''
    do the averages over the final histograms for each parameter.

    parameters
    ----------
    prefix: file prefix for input and output files
    plot_functions: the plot functions for plotting final averages
    spectrum: the input spectrum for plotting with average absorption

    outputs
    -------
    outfiles: generated list of all output files
    '''
    outfiles = []
    do_avgs = True
    for field in dataclasses.fields(constants.Genome):
        dfs = []
        for i in range(constants.n_runs):
            f = f"{prefix}_{field.name}_{i}_final.txt"
            if os.path.isfile(f):
                df = pd.read_csv(f, index_col=0)
                dfs.append(df)
            else:
                do_avgs = False
        if do_avgs:
            avg = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
            avg /= constants.n_runs
            of = f"{prefix}_{field.name}_avg_finals.txt"
            outfiles.append(of)
            avg.to_csv(of)
            outfiles.append(plot_functions[field.name](of, field.name))
    # average of final average absorption spectra
    avg_abs = []
    do_avg_abs = True
    for i in range(constants.n_runs):
        f = f"{prefix}_{i}_final_avg_spectrum.txt"
        if os.path.isfile(f):
            avg_abs.append(np.loadtxt(f))
        else:
            do_avg_abs = False
    if do_avg_abs:
        avg = reduce(lambda x, y: x + y, avg_abs)
        avg /= constants.n_runs
        of = f"{prefix}_avg_avg_spectrum.txt"
        outfiles.append(of)
        np.savetxt(of, avg)
        outfiles.append(plots.plot_from_file(of, spectrum))
    return outfiles

if __name__ == "__main__":
    import light
    rng = np.random.default_rng()
    n = constants.population_size
    spectrum, output_prefix = light.spectrum_setup("stellar",
            Tstar=5770, Rstar=6.957E8, a=1.0, attenuation=0.0)
    prefix = os.path.join(constants.output_dir, "tests", "stats",
            output_prefix)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    population = [ga.new(rng, **{'n_b': 1, 'n_s': 1}) for _ in range(n)]
    nu_e = [100.0 * rng.random() for _ in range(n)]
    results = {
        'nu_e': nu_e,
        'fitness': [nu_e[i] * rng.random() for i in range(n)],
        'redox': [rng.random(size=(
            rcm.n_rc[getattr(population[i], 'rc')], 2))
            for i in range(n)],
        'recomb': [rng.random(size=
            rcm.n_rc[getattr(population[i], 'rc')])
            for i in range(n)],
    }
    output = do_stats(population, results, spectrum, prefix=prefix,
            **big_stats)
    print(output)

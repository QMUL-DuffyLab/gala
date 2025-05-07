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

'''
NB: this isn't finished yet. ideally we want to keep track of whichever
genome parameters are important, plus whichever other parameters are important
(nu_e, etc.), and then put them all together and just call the stats functions
once and let this module sort out which averaging functions need to be applied
to which parameter. but i haven't fully sorted out how that's gonna work yet.
'''
things_to_average = ['n_b', 'n_s', 'rho', 'pigment']

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

def int_dist(population, name, outfile):
    '''
    calculate counts for an integer Genome parameter.

    parameters
    ----------
    population: a list of Genomes
    name: name of the Genome parameter
    outfile: output file

    outputs
    -------
    '''
    arr = np.array([getattr(p, name) for p in population])
    v, c = np.unique(arr, return_counts=True, equal_nan=False)
    df = pd.Series(c / constants.population_size, index=v)
    df.to_csv(outfile)
    return

def float_dist(population, name, outfile):
    '''
    calculate a histogram of a float Genome parameter.

    parameters
    ----------
    population: a list of Genomes
    name: name of the Genome parameter
    outfile: output file

    outputs
    -------
    '''
    arr = np.array([getattr(p, name) for p in population])
    bounds = constants.bounds[name]
    if name in constants.binwidths.keys():
        bw = constants.binwidths[name]
        bins = np.linspace(*bounds,
        num=np.round(1 + (bounds[1] - bounds[0])/bw).astype(int))
    else:
        print(f"Field {name} has no binwidth given in constants.binwidths. Using default 50 bins")
        bins = np.linspace(*bounds) # 50 bins
    h = np.histogram(arr, bins=bins)[0]
    df = pd.Series(h / constants.population_size, index=bins[:-1])
    df.to_csv(outfile)
    return

def array_dist(population, name, outfile):
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
    ntype = getattr(population[0], name).dtype
    arr = np.zeros((constants.hist_sub_max, constants.population_size), dtype=ntype)
    for j, p in enumerate(population):
        for i in range(constants.hist_sub_max):
            if i < p.n_s:
                arr[i][j] = getattr(p, name)[i]
    hl = []
    if ntype == "U10":
        bins = list(constants.bounds[name])
        for i in range(constants.hist_sub_max):
            ap = pd.Series(arr[i])
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

def string_dist(population, name, outfile): 
    '''
    calculate a bar chart for string parameters,
    e.g. the name of the RC supersystem of the Genomes.

    parameters
    ----------
    population: a list of Genomes
    name: name of the parameter
    outfile: output file

    outputs
    -------
    '''
    arr = np.array([getattr(p, name) for p in population])
    v, c = np.unique(arr, return_counts=True, equal_nan=False)
    s = pd.Series(c / constants.population_size, index=v)
    s.to_csv(outfile)
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
        if field.name in things_to_average:
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

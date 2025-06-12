'''
author: @callum
stats.py - some functions to do statistics, both on the
genomes and the outputs. not finished yet. need to think a little more

each of the averaging functions should return a tuple of a 'result'
and a list of output files it created. these can be empty, but they should be
there.
then do_files calls the set of functions, runs the statistics we want, and
returns a dict of outputs sorted by key, and a list of output files created
'''

import os
import sys
import re
import glob
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import reduce
import constants
import plots
import genetic_algorithm as ga
import rc as rcm

def unpickle_population(filename):
    return pd.read_pickle(filename)

def split_population(df, split=None):
    if split is not None:
        # get unique values of the split key and make a dict from them
        df_dict = {rct: df[df[split] == rct] for rct in df[split].unique()}
    else:
        df_dict = {'all': df}
    return df_dict

def avg(df, key, **kwargs):
    return (df[key].mean(), df[key].sem()), []

def counts(df, key, **kwargs):
    res = df.value_counts(key)
    return (res.index.to_numpy(), res.to_numpy), []

def hist(df, prefix, key, split=None, **kwargs):
    '''
    use pandas and seaborn to plot per-element
    histograms of array parameters
    '''
    # this is very inefficient. but basically, do one histogram for
    # each element of the given array, knowing that the arrays might be
    # different sizes (e.g. the redox and recombination arrays)
    hmax = np.max([len(np.array(df[key][i]).flatten())
        for i in range(len(df))])
    if key in ga.genome_parameters:
        if ga.genome_parameters[key]['array']:
            if ga.genome_parameters[key]['depends'] == "n_s":
                hmax = constants.hist_sub_max
    if isinstance(np.array(df[key][0]).flatten()[0], int):
        discrete = True
    else:
        discrete = False
    # figure out a layout for the plots
    if hmax % 2 == 0:
        nrows = hmax//2
        ncols = 2
    else:
        nrows = hmax
        ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
            figsize=(12 * ncols, 8 * nrows))
    for i in range(hmax):
        col = np.full(len(df), np.nan)
        for j in range(len(df)):
            v = np.array(df[key][j]).flatten()
            if i < len(v):
                col[j] = v[i]
        # add col to the dataframe so that we can plot it with seaborn
        str_id = f"{key}[{i}]"
        df[str_id] = col
        if hmax == 1:
            curr_ax = axes # otherwise the histplot will fail below
        else:
            curr_ax = axes[i // 2, i % 2]
        sns_kwargs = {'ax': curr_ax, 'data': df, 'x': str_id,
                'discrete': discrete}
        if split is not None:
            sns_kwargs['hue'] = split
            sns_kwargs['multiple'] = 'dodge'
        # sometimes seaborn will get confused trying to make the
        # histogram, which i think is a bin issue, so here try to
        # give it a sensible bin width to work with
        bw = (np.nanmax(col) - np.nanmin(col)) / 50
        if bw > 0.0 and bw != np.nan:
            sns_kwargs['binwidth'] = bw
        sns.histplot(**sns_kwargs)
        # delete the temporary index now we don't need it
        df.drop(columns=str_id, inplace=True)
    if split is not None:
        suffix = f"hist_{key}_splitby_{split}.pdf"
    else:
        suffix = f"hist_{key}.pdf"
    outfile = f"{prefix}_{suffix}"
    plt.savefig(outfile)
    plt.close()
    return (), [outfile]

def absorption(df, spectrum, prefix, **kwargs):
    '''
    convert a dataframe to a population of genomes and plot
    the average absorption of that population, potentially
    split into subpopulations using kwarg split. the argument key
    is only here for consistency with the other functions.
    '''
    outfiles = []
    split = kwargs['split'] if 'split' in kwargs else None
    df_dict = split_population(df, split)
    for rct, subdf in df_dict.items():
        subpop = []
        abs_file = f"{prefix}_{rct}_abs.txt"
        for i in range(len(subdf)):
            row = subdf.iloc[i]
            d = {}
            for index in row.index:
                # NB: this assumes that the genome corresponding to the
                # pickled population is the same as the current one, so
                # if you change the definition of the genome in
                # genetic_algorithm.py and then read in an old population,
                # it'll break. unsure how to get around this. in theory
                # i could save a minimal version of the genome to JSON?
                if index in ga.genome_parameters:
                    d[index] = row[index]
            g = ga.Genome(**d)
            subpop.append(g)
        outfiles.extend(plots.plot_average(subpop, spectrum, abs_file))
    return (), outfiles

def do_stats(df, spectrum, prefix, **kwargs):
    '''
    output_stats
    '''
    spd = {'spectrum': spectrum, 'prefix': prefix}
    output = {} # list of all output files (will be zipped)
    output_files = []
    for k, v in kwargs.items():
        fn = getattr(sys.modules[__name__], v['function'])
        # fn_kwargs = v | spd
        fn_kwargs = {**v, **spd}
        fn_kwargs['key'] = k
        op, ofs = fn(df, **fn_kwargs)
        output[k] = op
        output_files.extend(ofs)
    return output, output_files

minimal_stats = {
        'nu_e':    {'function': 'avg'},
        'fitness': {'function': 'avg'},
        'n_b':     {'function': 'avg'},
        'n_s':     {'function': 'avg'},
        'rc':      {'function': 'counts'},
    }

big_stats = {
        'shift':      {'function': 'hist', 'split': 'rc'},
        'redox':      {'function': 'hist', 'split': 'rc'},
        'nu_e':       {'function': 'hist', 'split': 'rc'},
        'n_b':        {'function': 'hist'},
        'rc':         {'function': 'counts'},
        'absorption': {'function': 'absorption', 'split': 'rc'}
    }

if __name__ == "__main__":
    import light
    import solvers
    import genetic_algorithm as ga

    rng = np.random.default_rng()
    cost = 0.01
    n = constants.population_size
    spectrum, output_prefix = light.spectrum_setup("stellar",
            Tstar=5770, Rstar=6.957E8, a=1.0, attenuation=0.0)
    prefix = os.path.join(constants.output_dir, "tests", "stats",
            output_prefix)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    population = [ga.new(rng, **{'n_b': 1, 'n_s': 1}) for _ in range(n)]
    results = {'nu_e': [], 'nu_cyc': [], 'fitness': [],
               'redox': [], 'recomb': []}
    for p in population:
        res, fail = solvers.antenna_RC(p, spectrum, nnls='scipy')
        res['fitness'] = ga.fitness(p, res['nu_e'], cost)
        for k, v in res.items():
            results[k].append(v)
    df = pd.DataFrame(population)
    for k, v in results.items():
        df[k] = v
    df.to_csv(f"{prefix}_df.csv", sep="\t")
    output, output_files = do_stats(df, spectrum, prefix, **minimal_stats)
    print("Minimal stats:")
    print(f"Output dict: {output}")
    print(f"Output file dict: {output_files}")
    output, output_files = do_stats(df, spectrum, prefix, **big_stats)
    print("Big stats:")
    print(f"Output dict: {output}")
    print(f"Output file dict: {output_files}")

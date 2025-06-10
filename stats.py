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
import seaborn as sns
import pandas as pd
from functools import reduce
import constants
import plots
import genetic_algorithm as ga
import rc as rcm

def counts(arr, **kwargs):
    '''
    get counts and values of a parameter.
    will work for string or int variables
    '''
    # if isinstance(arr[0], np.ndarray):
    #     np.array(arr).flatten()
    v, c = np.unique(arr, return_counts=True, equal_nan=False)
    if "outfile" in kwargs:
        s = pd.Series(c / len(arr), index=v)
        s.to_csv(kwargs['outfile'])
        return (v, c), kwargs['outfile']
    else:
        return (v, c), []

def avg(arr, **kwargs):
    '''
    return average and error of quantity.
    will work for int/float
    '''
    # if isinstance(arr[0], np.ndarray):
    #     np.array(arr).flatten()
    m, e = np.mean(arr), np.std(arr) / np.sqrt(len(arr))
    if "outfile" in kwargs:
        np.savetxt(kwargs['outfile'], np.column_stack((m, e)))
        return (m, e), kwargs['outfile']
    else:
        return (m, e), []

def element_avg(arr, **kwargs):
    '''
    return element-wise average and error of int/float array
    '''
    m, e = np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(len(arr))
    if "outfile" in kwargs:
        np.savetxt(kwargs['outfile'], np.column_stack((m, e)))
        return (m, e), kwargs['outfile']
    else:
        return (m, e), []
    
def hist(filename, key, split=None):
    '''
    use pandas and seaborn to plot per-element
    histograms of array parameters
    '''
    df = pd.read_pickle(filename)
    # this is very inefficient. but basically, do one histogram for
    # each element of the given array, knowing that the arrays might be
    # different sizes (e.g. the redox and recombination arrays)
    hmax = np.max([len(np.array(df[key][i]).flatten())
        for i in range(len(df))])
    if key in ga.genome_parameters:
        if ga.genome_parameters[key]['array']:
            if ga.genome_parameters[key]['depends'] == "n_s":
                hmax = constants.hist_sub_max
    if isinstance(np.array(df[key][0]).flatten()[0], (int, np.int)):
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
        if split is not None:
            sns.histplot(ax=curr_ax, data=df, x=str_id,
                    discrete=discrete, hue=split, multiple='dodge')
        else:
            sns.histplot(ax=curr_ax, data=df, x=str_id, discrete=discrete)
        # delete the temporary index now we don't need it
        df.drop(columns=str_id, inplace=True)
    if split is not None:
        suffix = f"hist_{key}_splitby_{split}.pdf"
    else:
        suffix = f"hist_{key}.pdf"
    plt.savefig(f"{os.path.splitext(filename)[0]}_{suffix}")
    plt.close()

def split_population(population, split_key):
    '''
    for a given key to split on, split the population into a set
    of subpopulations based on their value of that key. note that
    the way this is coded means that at the moment it can basically
    only be RC you split on, but hopefully it's at least a starting
    point if you need something more sophisticated. some notes about
    that are below. returns a dict of subpopulations
    '''
    b = ga.genome_parameters[split_key]['bounds']
    # you could for example make bins and split based on those for
    # numerical parameters, i guess. you'd also have to change the
    # `getattr() == b[j]` line below based on how you're splitting
    subarr = [[] for i in range(len(b))]
    for j in range(len(b)):
        for i in range(constants.population_size):
            if getattr(population[i], split_key) == b[j]:
                subarr[j].append(population[i])
    sd = {b[j]: pd.DataFrame(subarr[j]) for j in range(len(b))}
    return sd

def do_stats(population, results, spectrum, prefix=None, **kwargs):
    # note: this assumes that the key you split on is always
    # something in the genome. i think that's reasonable
    output = {}
    output_files = [] # list of all output files (will be zipped)
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
                ofs = [] # output files for this parameter
                for j in range(len(b)):
                    of = f"{prefix}_{split_key}_{b[j]}_{key}.txt"
                    output_files.append(of)
                    ofs.append(of)
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
            output[key] = kwargs[key](arr)
    return output, output_files

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
        # 'shift': hist,
        'absorption': plots.plot_average}

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

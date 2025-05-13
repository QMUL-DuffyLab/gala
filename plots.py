# -*- coding: utf-8 -*-
"""
29/11/2023
@author: callum

polygon_under_graph and the 3d plot generally stolen from matplotlib
"""
import os
import re
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PolyCollection
import constants
import light
import rc

target_spectra = {
    "PSII": os.path.join("spectra", "psii_wientjies.csv"),
    "BBY": os.path.join("spectra", "bby_spinach.csv"),
    "6803": os.path.join("spectra", "PCC_6803_Abs.txt"),
    "6301": os.path.join("spectra", "SP_6301_Abs.txt"),
    "FRL":  os.path.join("spectra", "frl_cells.csv"),
    "a_marina":  os.path.join("spectra", "a_marina.csv"),
    "marine": os.path.join("spectra", "kolodny_marine_pbs.csv"),
    "red_alga": os.path.join("spectra", "l_glaciale.csv"),
    "a_platensis": os.path.join("spectra",
                    "pbs_a_platensis_appl_sci_2020.csv")
    }

genome_labels = {
        # raw strings for the latex labels for genome parameters
        # don't wrap in $ $, we need them as-is but also with <>
        # for averaging, and you can't do $ <\left<$ ${s}$ $\right>$
        "n_b": r' n_b ',
        "n_s": r' n_s ',
        "n_p": r' n_p ',
        "nu_e": r'  \nu_e ',
        "phi_e": r'  \bar{\phi}_e ',
        "phi_e_g": r'  \phi_e ',
        "fitness": r'  F ',
        "alpha": r'  \alpha ',
        "eta": r'  \eta ',
        "phi": r'  \phi ',
        }

# this is no longer a spectrum colour - what have i done here? fix
def get_spectrum_colour(name):
    cmap = mpl.colormaps["turbo"]
    names = constants.bounds['pigment']
    colours = [cmap(i / float(len(names))) for i in range(len(names))]
    cdict = dict(zip(names, colours))
    for k, v in cdict.items():
        if k in name: # ignore the intensity part of the output name
            return v
    else:
        print("spectrum colour not found")
        return '#99999999'

def get_pigment_colour(name):
    cmap = mpl.colormaps["turbo"]
    names = constants.bounds['pigment']
    colours = [cmap(i / float(len(names))) for i in range(len(names))]
    cdict = dict(zip(names, colours))
    for k, v in cdict.items():
        if name == k: # ignore the intensity part of the output name
            return v
    else:
        print("spectrum colour not found")
        return '#99999999'

def antenna_lines(p, l):
    rcp = rc.params[p.rc]
    n_rc = len(rcp["pigments"])
    rc_n_p = [constants.pigment_data[rc]["n_p"] for rc in rcp["pigments"]]
    n_p = np.array([*rc_n_p, *p.n_p], dtype=np.int32)
    # 0 shift for RCs. shifts stored as integer increments, so
    # multiply by shift_inc here
    shift = np.array([*[0.0 for _ in range(n_rc)], *p.shift],
                     dtype=np.float64)
    shift *= constants.shift_inc
    pigments = np.array([*rcp["pigments"], *p.pigment], dtype='U10')
    lines = np.zeros((p.n_s + n_rc, len(l)))
    pdata = constants.pigment_data
    total = np.zeros_like(l)
    for i in range(p.n_s + n_rc):
        current_pigment = pdata[pigments[i]]['abs']
        for j in range(current_pigment['n_gauss']):
            lines[i] += current_pigment['amp'][j] * np.exp(-1.0
                    * (l - (current_pigment['mu'][j] +
                        (shift[i])))**2
                    / (2.0 * (current_pigment['sigma'][j]**2)))
        lines[i] /= np.sum(lines[i])
        total += lines[i] * n_p[i]
    return lines, total

def antenna_spectra(p, l, ip_y,
        lines_file, total_file, draw_00=True):
    '''
    Draw the total absorption spectrum and the set of subunit
    spectra for the given antenna (usually the fittest one).
    When drawing the individual subunit lines, by default this
    will also plot the 0-0 line without the offset as a dashed vline;
    set draw_00 to False to disable that
    '''
    pd = constants.pigment_data
    lines, total = antenna_lines(p, l)
    lps = [pd[p.pigment[i]]['abs']['mu'][0] for i in range(p.n_s)]
    fig, ax = plt.subplots(figsize=(12,8))
    for i in range(p.n_s):
        # generate Gaussian lineshape for given pigment
        # and draw dotted vline at normal peak wavelength?
        color = 'C{:1d}'.format(i)
        if draw_00:
            ax.axvline(x=lps[i], color=color, ls='--')
        label = f"Subunit {i + 1:1d}: {pd[p.pigment[i]]['abs']['text']}"
        plt.plot(l, lines[i]/np.max(lines[i]), label=label)
    plt.plot(l, ip_y, label="Incident")
    ax.set_xlabel(r'$ \lambda (\text{nm}) $')
    ax.set_ylabel("Intensity (arb. for antenna)")
    lmin = 200.0 if np.min(l) < 200.0 else np.min(l)
    lmax = 1000.0 if np.max(l) > 1000.0 else np.max(l)
    ax.set_xlim([lmin, lmax])
    ax.legend()
    fig.savefig(lines_file)
    plt.close()

    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(l, ip_y, label="Incident", color='0.8')
    # set the peak height of the total antenna spectrum
    # to the same height as the incident spectrum
    norm_total = total / (np.max(total) / np.max(ip_y))
    plt.plot(l, norm_total, label="Total antenna")
    ax.set_xlabel(r'$ \lambda (\text{nm}) $')
    ax.set_ylabel("Intensity (arb. for antenna)")
    ax.set_xlim([lmin, lmax])
    ax.legend()
    fig.savefig(total_file)
    plt.close()
    #return fig, ax

def get_best_from_file(input_file):
    '''
    get parameters for the best antenna and instantiate.
    again, this is incredibly ugly - i've been just
    printing out str(genome) to keep track of the best ones, and
    i figured i could eval them back in or something, but that
    does not work, so read in the final best antenna as a string
    and then parse with regular expressions lol
    '''
    with open(input_file) as f:
        '''
        this is very inefficient, reading in the whole file - if
        the genomes get huge or the runs get really long, might be worth
        looking into reading from end of file. alternatively call sed via
        subprocess or something
        '''
        fstring = ' '.join(f.read().splitlines())
    f.close()
    matches = re.finditer(r'Genome\(*', fstring)
    for m in matches:
        pass # we only want the last one
    best = fstring[m.start():]
    print("best = ", best)
    n_b = int(re.search(r'n_b=(\d+)', best).group(1))
    n_s = int(re.search(r'n_s=(\d+)', best).group(1))
    shiftm = re.search(r"shift=array\(\[\s*([0-9e.\-]+[,\s\]]+)+",
                       best).group(0)
    shifta = re.search(r"\[(.*)\]", shiftm).group(0)
    shift = np.fromstring(shifta[1:-1], sep=',')
    n_pm = re.search(r"n_p=array\(\[\s*([0-9e.\-]+[,\s\]]+)+", best).group(0)
    n_pa = re.search(r"\[(.*)\]", n_pm).group(0)
    n_p = np.fromstring(n_pa[1:-1], sep=',', dtype=int)
    pigm = re.search(r"pigment=array\(\[\s*([a-z_\-']+[,\s\]]+)+", best).group(0)
    piga = re.search(r"\[(.*)\]", pigm).group(0)
    # numpy doesn't know how to fromstring() this so do it manually
    pigment = np.array(piga[1:-1]
                       .replace("'", "")
                       .replace(" ", "")
                       .split(","), dtype='U10')
    return constants.Genome(n_b, n_s, n_p, shift, pigment)

def plot_from_file(infile, spectrum):
    fig, ax = plt.subplots(figsize=(9,9))
    plt.plot(spectrum[:, 0], spectrum[:, 1], label='Incident',
            color='#99999999', ls='--', lw=2.0)
    avg_spectrum = np.loadtxt(infile)
    x = avg_spectrum[:, 0]
    y = avg_spectrum[:, 1]/np.sum(avg_spectrum[:, 1])
    plt.plot(avg_spectrum[:, 0], avg_spectrum[:, 1] / np.max(avg_spectrum[:, 1]),
            color='k', lw=5.0,
            label=r'$ \left<A\left(\lambda\right)\right> $')
    ax.set_xlabel(r'wavelength (nm)')
    ax.set_ylabel("intensity (arb.)")
    ax.set_xlim(constants.x_lim)
    plt.grid(True)
    # plt.legend()
    fig.tight_layout()
    outfile = os.path.splitext(infile)[0] + ".pdf"
    plt.close()
    return outfile

def plot_best(best_file, spectrum):
    '''
    just wraps the get_best and spectrum functions
    to call in one line from main.py
    '''
    p = get_best_from_file(best_file)
    prefix = os.path.splitext(best_file)[0]
    lines_file  = prefix + "_lines.pdf"
    total_file  = prefix + "_total.pdf"

    antenna_spectra(p, spectrum[:, 0],
                         spectrum[:, 1], lines_file, total_file)
    return [lines_file, total_file]

def plot_average(population, spectrum, outfile, **kwargs):
    '''
    plot the average absorption spectrum of the whole population.

    parameters
    ----------
    population: a list of Genomes
    spectrum: the light input
    out_prefix: a prefix for the output files

    outputs
    -------
    datafile: filename of the actual data
    plotfile: filename of the plot
    '''
    l = spectrum[:, 0]
    total = np.zeros_like(l)
    for a in population:
        _, run_tot = antenna_lines(a, l)
        total += run_tot
    total /= len(population)
    norm_total = total / np.sum(total)
    datafile = outfile
    np.savetxt(datafile, np.column_stack((l, norm_total)))
    # normalise peak height to 1 for plot
    norm_total /= np.max(norm_total)
    xs = [l, l]
    ys = [spectrum[:, 1], norm_total]
    if "label" in kwargs:
        labels = ["", kwargs['label']]
    else:
        labels = ["", ""]
    colours = ['0.8', 'C0']
    lws = [2.0, 5.0]
    lsts = ['--', '-']
    peak_wl = 0.0
    draw_peak_wl = False
    if "target" in kwargs:
        tspec = np.loadtxt(target_spectra[kwargs['target']])
        # we want the largest peak in the *visible* spectrum
        t_vis = tspec[(tspec[:, 0] > 500.0) & (tspec[:, 0] < 800.0)]
        arg = np.argmax(t_vis[:, 1])
        peak_wl = t_vis[arg, 0]
        # normalise that peak to 1
        norm_tspec = tspec[:, 1] / t_vis[arg, 1]
        xs.append(tspec[:, 0])
        ys.append(norm_tspec)
        labels.append(kwargs['target'])
        colours.append('C1')
        draw_peak_wl = True

    fig, ax = plt.subplots(figsize=(12,8))
    for x, y, l, c, lw, lst in zip(xs, ys, labels, colours, lws, lsts):
        plt.plot(x, y, label=l, color=c, lw=lw, ls=lst)
    ax.set_xlim(constants.x_lim)
    ax.set_xlabel(r' $ \lambda (\text{nm}) $')
    ax.set_ylabel('intensity')
    plt.grid(True)
    if "label" in kwargs:
        plt.legend()
    fig.tight_layout()

    if draw_peak_wl:
        plt.axvline(peak_wl, ls='--', color='C1')

    if "cost" in kwargs:
        ax.set_title("Cost = " + str(kwargs['cost']))

    plotfile = os.path.splitext(datafile)[0] + ".pdf"
    plt.savefig(plotfile)
    plt.close()
    return [datafile, plotfile]

def plot_running(infile, scalars):
    '''
    plot the running average of scalar parameters with errors.

    parameters
    ----------
    infile: location of the file to read in and plot
    scalars: list of scalars.

    outputs
    -------
    outfiles: location of the PDF files output (one per scalar)
    '''
    df = pd.read_csv(infile)
    outfiles = []
    for s in scalars:
        fig, ax = plt.subplots(figsize=(12,8))
        y = df[s]
        ax.plot(y)
        serr = f"{s}_err"
        ax.fill_between(df.index, y - df[serr], y + df[serr], alpha=0.5)
        ax.set_ylim(constants.bounds[s])
        ax.set_xlabel("Generation")
        if s in genome_labels:
            ax.set_ylabel(fr"$ \left<{genome_labels[s]}\right> $")
        else:
            ax.set_ylabel(fr"$ \left< {s} \right>$")
        dn = os.path.dirname(infile)
        bn = os.path.splitext(os.path.basename(infile))[0]
        outfile = os.path.join(dn, f"{s}_{bn}.pdf")
        print(outfile)
        outfiles.append(outfile)
        fig.savefig(outfile)
        plt.close()
    return outfiles

def plot_bar(infile, name=None):
    '''
    plot a bar chart for int, float and string parameters.

    parameters
    ----------
    infile: location of the file to read in and plot
    name: name of the parameter (optional)

    outputs
    -------
    outfile: location of the PDF file output
    '''
    arr = pd.read_csv(infile).to_numpy()
    labels = arr[:, 0]
    counts = arr[:, 1]
    fig, ax = plt.subplots(figsize=(12, 8))
    if len(labels) > 0:
        ax.bar(labels, counts, width=1.0/len(labels), edgecolor='k')
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Proportion")
    if name is not None:
        if name in genome_labels:
            ax.set_xlabel(fr"$ {genome_labels[name]} $")
        else:
            ax.set_xlabel(name)
    outfile = os.path.splitext(infile)[0] + ".pdf"
    fig.savefig(outfile)
    plt.close()
    return outfile

def plot3d(infile, name=None):
    '''
    make a 3d plot of the histogram of the per-subunit
    quantities like n_p etc. reads in the dataframe from
    infile and outputs a pdf.

    parameters
    ----------
    infile: location of the file to read in
    name: name of the Genome parameter

    outputs
    -------
    outfile: location of the ouput PDF
    '''
    avg = pd.read_csv(infile).to_numpy()
    bins = avg[:, 0]
    props = avg[:, 1:]
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    
    for k in reversed(range(props.shape[1])):
        ys = props[:, k]
        xs = bins
        color = 'C{:1d}'.format(k)
        ax.bar(xs, ys, zs=k, zdir='y', color=color, alpha = 0.7)
     
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.labelpad = 20
    
    ax.set_ylabel("Subunit")
    ax.set_zlabel("Proportion")
    if name is not None:
        if name in genome_labels:
            ax.set_xlabel(fr" $ {genome_labels[name]} $")
        else:
            ax.set_xlabel(name)
    ax.set_zlim([0.0, 1.0])
    ax.set_yticks(np.arange(props.shape[1]))
    ax.set_yticklabels(["{:1d}".format(i) for i in np.arange(1, props.shape[1] + 1)])

    outfile = os.path.splitext(infile)[0] + ".pdf"
    fig.savefig(outfile)
    plt.close()
    return outfile


def pigment_bar(infile, name=None):
    '''
    plot a bar chart showing prevalence of each pigment as a
    function of distance from RC for each subunit in the file

    parameters
    ----------
    infile: location of the input file
    name: not actually used, just there for consistency in the
    call to the plot functions

    outputs
    -------
    outfile: location of the PDF file
    '''
    arr = pd.read_csv(infile).to_numpy()
    names = arr[:, 0]
    labels = {n: constants.pigment_data[n]['name'] for n in names}
    # the array's read in as dtype object because of the string column
    # so convert it to float, otherwise bottom += p fails below
    hists = arr[:, 1:].astype(float)
    props = dict(zip(names, hists))
    # get sub_max from here rather than constants; it might change
    sub_max = hists.shape[1]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlabel("Distance from RC")
    ax.set_ylabel("Prevalence")
    bottom = np.zeros(sub_max)
    for b, p in props.items():
        pp = ax.bar(np.arange(sub_max) + 1, p, 0.8,
                label=labels[b], color=get_pigment_colour(b),
                bottom=bottom, edgecolor='0.6')
        bottom += p
    ax.legend(loc='upper right')
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    # add horizontal space for legend
    ax.set_xlim([0, sub_max + 2])
    ax.set_xticks([2 * i for i in range(sub_max // 2)])
    outfile = os.path.splitext(infile)[0] + ".pdf"
    fig.savefig(outfile)
    plt.close()
    return outfile


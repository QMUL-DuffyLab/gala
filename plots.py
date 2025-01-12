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
from matplotlib.collections import PolyCollection
import constants
import light

target_spectra = {
    # "PSII": os.path.join("spectra", "PSII.csv"),
    "PSII": os.path.join("spectra", "psii_wientjies.csv"),
    # from Jim Barber paper
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

# this is no longer a spectrum colour - what have i done here? fix
def get_spectrum_colour(name):
    cmap = mpl.colormaps["turbo"]
    names = constants.bounds['pigment']
    colours = [cmap(i / float(len(names))) for i in range(len(names))]
    cdict = {n: c for n, c in zip(names, colours)}
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
    cdict = {n: c for n, c in zip(names, colours)}
    for k, v in cdict.items():
        if name == k: # ignore the intensity part of the output name
            return v
    else:
        print("spectrum colour not found")
        return '#99999999'

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def antenna_lines(p, l):
    pd = constants.pigment_data
    lines = np.zeros((p.n_s, len(l)))
    total = np.zeros_like(l)
    for i in range(p.n_s):
        pigment = pd[p.pigment[i]]['abs']
        for j in range(pigment['n_gauss']):
            lines[i] += pigment['amp'][j] * np.exp(-1.0
                    * (l - (pigment['mu'][j] +
                        (p.shift[i] * constants.shift_inc)))**2
                    / (2.0 * (pigment['sigma'][j]**2)))
        lines[i] /= np.sum(lines[i])
        # lines[i] /= np.max(lines[i])
        total += lines[i] * p.n_p[i]
    return lines, total

def plot_nu_phi_2(nu_e, phi_e, fitness, n_b, n_s, filename):
    fig, ax = plt.subplots(nrows=2, figsize=(12,8), sharex=True)
    ax[0].plot(np.arange(len(nu_e)), nu_e, color='C0',
               label=r'$ \left<\nu_e\right> $')
    ax2 = ax[0].twinx()
    ax2.plot(np.arange(len(phi_e)), phi_e, color='C1',
               label=r'$ \left<\varphi_e\right> $')
    ax[0].plot(np.arange(len(fitness)), fitness, color='C2',
               label=r'$ \left< \text{fitness} \right> $')
    ax[1].plot(np.arange(len(n_b)), n_b, label=r'$ \left<n_b\right> $')
    ax[1].plot(np.arange(len(n_s)), n_s, label=r'$ \left<n_s\right> $')
    ax[0].legend(loc='lower left')
    ax[1].legend(loc='upper left')
    ax2.legend(loc='center right')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel(r'$ \left<n_s\right> $')
    ax[0].set_ylabel(r'$ \left<\nu_e\right>, \; \left< \text{fitness} \right> $')
    ax2.set_ylabel(r'$ \left<\varphi_e\right> $')
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def hist_plot(pigment_file, peak_file, n_p_file):
    '''
    gather up the histogram output and make 3D plots of them.
    these don't look that nice really, but it's really annoying trying
    to visualise the data any other way. 
    NB: the code's basically repeated x3 - is there a way to tidy this up?
    '''
    n_s = constants.hist_sub_max

    peak_hist = np.loadtxt(peak_file)
    # ndmin = 1 to ensure a list is returned
    pnames = np.loadtxt(pigment_file, usecols=0, ndmin=1, dtype=str)
    # if there's only one pigment type, this plot's meaningless
    # and also will crash because pprops[:, k] fails below
    if len(pnames) > 1:
        pprops = np.loadtxt(pigment_file,
                usecols=tuple(range(1, n_s + 1)), dtype=float)
        pigment_strings = [constants.pigment_data[p]['abs']['text'] for p in pnames]
        n_pigments = len(pigment_strings)
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(projection='3d')

        for k in reversed(range(n_s)):
            ys = pprops[:, k]
            xs = np.arange(n_pigments)
            color = 'C{:1d}'.format(k)
            ax.bar(xs, ys, zs=k, zdir='y', color=color, alpha = 0.7)
         
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.labelpad = 20

        ax.set_xlabel("Pigment")
        ax.set_ylabel("Subunit")
        ax.set_zlabel("Proportion")
        ax.set_zlim([0.0, 1.0])
        ax.set_xticks(np.arange(n_pigments))
        ax.set_yticks(np.arange(n_s))
        ax.set_yticklabels(["{:1d}".format(i) for i in np.arange(1, n_s + 1)])
        ax.set_xticklabels(pigment_strings)

        fig.savefig(os.path.splitext(pigment_file)[0] + ".pdf")
        plt.close()

    n_p_bins = np.loadtxt(n_p_file, usecols=0)
    props = np.loadtxt(n_p_file,
            usecols=tuple(range(1, n_s + 1)), dtype=float)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')

    for k in reversed(range(n_s)):
        # hide the counts of zero
        ys = props[1:, k]
        xs = n_p_bins[1:]
        color = 'C{:1d}'.format(k)
        ax.bar(xs, ys, zs=k, zdir='y', color=color, alpha = 0.7)
     
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.labelpad = 20

    ax.set_xlabel(r'$ n_{p} $')
    ax.set_ylabel("Subunit")
    ax.set_zlabel("Proportion")
    ax.set_zlim([0.0, 1.0])
    ax.set_yticks(np.arange(n_s))
    ax.set_yticklabels(["{:1d}".format(i) for i in np.arange(1, n_s + 1)])

    fig.savefig(os.path.splitext(n_p_file)[0] + ".pdf")
    plt.close()

    peak_bins = np.loadtxt(peak_file, usecols=0)
    props = np.loadtxt(peak_file,
            usecols=tuple(range(1, n_s + 1)), dtype=float)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')

    for k in reversed(range(n_s)):
        ys = props[:, k]
        xs = peak_bins
        color = 'C{:1d}'.format(k)
        ax.bar(xs, ys, zs=k, zdir='y', color=color, alpha = 0.7)
     
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.labelpad = 20

    ax.set_xlabel(r'$ \lambda_{\text{peak}} (\text{nm}) $')
    ax.set_ylabel("Subunit")
    ax.set_zlabel("Proportion")
    ax.set_zlim([0.0, 1.0])
    ax.set_yticks(np.arange(n_s))
    ax.set_yticklabels(["{:1d}".format(i) for i in np.arange(1, n_s + 1)])
    fig.savefig(os.path.splitext(peak_file)[0] + ".pdf")
    plt.close()

def julia_plot(p, output_file):
    '''
    this is *incredibly* ugly but don't judge pls
    i couldn't find a package to draw an antenna easily in python,
    but turns out getting python and julia to play well together
    on ubuntu isn't trivial, and then importing your own module
    doesn't seem to work and i couldn't find any docs explaining
    why, and it kept throwing errors, so now i'm just doing this
    '''
    lps = [(p.shift[i] * constants.shift_inc) +
           constants.pigment_data[p.pigment[i]]['abs']['mu'][0]
           for i in range(p.n_s)]
    cmd = "julia plot_antenna.jl --n_b {:d} --n_s {:d} ".format(p.n_b, p.n_s)\
    + "--lambdas " + " ".join(str(l) for l in lps)\
    + " --n_ps " + " ".join(str(n) for n in p.n_p)\
    + " --names " + " ".join(p for p in p.pigment)\
    + " --file " + output_file
    print(cmd)
    subprocess.run(cmd.split())

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

def plot_best(best_file, spectrum):
    '''
    just wraps the julia plot and best spectrum functions
    to call in one line from main.py
    '''
    p = get_best_from_file(best_file)
    prefix = os.path.splitext(best_file)[0]
    output_file = prefix + "_antenna.svg" # julia plot outputs svg
    lines_file  = prefix + "_lines.pdf"
    total_file  = prefix + "_total.pdf"

    # julia_plot(p, output_file)
    antenna_spectra(p, spectrum[:, 0],
                         spectrum[:, 1], lines_file, total_file)
    return [output_file, lines_file, total_file]

def plot_lines(xs, ys, labels, colours, **kwargs):
    fig, ax = plt.subplots(figsize=(12,8))
    for x, y, l, c in zip(xs, ys, labels, colours):
        plt.plot(x, y, label=l, color=c)
    if "xlim" in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if "ylim" in kwargs:
        ax.set_ylim(kwargs['ylim'])
    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if "title" in kwargs:
        ax.set_title(kwargs['title'])
    plt.grid()
    plt.legend()
    fig.tight_layout()
    # return the plot in case there's other stuff we need to do with it
    return fig, ax

def plot_average(antennae, spectrum, out_prefix, **kwargs):
    '''
    '''
    print("Plotting average for {}".format(out_prefix))
    l = spectrum[:, 0]
    total = np.zeros_like(l)
    for a in antennae:
        _, run_tot = antenna_lines(a, l)
        total += run_tot
    total /= len(antennae)
    norm_total = total / np.sum(total)
    outfile = out_prefix + ".dat"
    np.savetxt(outfile, np.column_stack((l, norm_total)))
    # normalise peak height to 1 for plot
    norm_total /= np.max(norm_total)
    xs = [l, l]
    ys = [spectrum[:, 1], norm_total]
    if "label" in kwargs:
        labels = ["", kwargs['label']]
    else:
        labels = ["", ""]
    colours = ['0.8', 'C0']
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

    fig, ax = plot_lines(xs, ys, labels, colours,
            xlabel=r' $ \lambda (\text{nm}) $',
            ylabel="Intensity (arbitrary)",
            **kwargs)

    if draw_peak_wl:
        plt.axvline(peak_wl, ls='--', color='C1')

    if "cost" in kwargs:
        ax.set_title("Cost = " + str(kwargs['cost']))

    plt.savefig(out_prefix + ".pdf")
    plt.close()

def plot_average_best(path, spectrum, out_name,
        cost, **kwargs):
    l = spectrum[:, 0]
    prefix = '{}/best_{}'.format(path, out_name)
    suffix = '.dat'
    bests = []
    for i in range(constants.n_runs):
        bf = prefix + "_r{:1d}".format(i) + suffix
        bests.append(get_best_from_file(bf))

    out_prefix = prefix + "_avg_best"
    plot_average(bests, spectrum, out_prefix,
                 label=r'$ \left<\text{best}\right> $', **kwargs)

def pigment_bar(pigments, outfile):
    '''
    plot a bar chart showing prevalence of each pigment as a
    function of distance from RC up to constants.hist_sub_max
    '''
    hists = np.array([row[1:] for row in pigments])
    hists = hists / float(constants.n_runs)
    names = [row[0] for row in pigments]
    labels = {n: constants.pigment_data[n]['name'] for n in names}
    props = {name: hist for name, hist in zip(names, hists)}

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlabel("Distance from RC")
    ax.set_ylabel("Prevalence")
    bottom = np.zeros(len(names))
    for b, p in props.items():
        pp = ax.bar(np.arange(constants.hist_sub_max) + 1, p, 0.8,
                label=labels[b], color=get_pigment_colour(b),
                bottom=bottom, edgecolor='0.6')
        bottom += p
    ax.legend(loc='upper right')
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    # add horizontal space for legend
    ax.set_xlim([0, constants.hist_sub_max + 2])
    ax.set_xticks([2 * i for i in range(constants.hist_sub_max // 2)])
    fig.savefig(outfile)
    plt.close()
    

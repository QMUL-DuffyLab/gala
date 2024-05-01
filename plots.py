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
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import constants
import light
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import Main
# from julia.Main import DrawAntennae

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def gauss(l, lp, w):
    '''
    return a properly unscaled Gaussian to plot alongside spectral irradiance
    '''
    y = np.exp(-1.0*((l - lp)**2)/(2.0*w**2))
    return y

def two_gauss(l, lp1, w1, lp2, w2, a12):
    '''
    return a properly unscaled Gaussian to plot alongside spectral irradiance
    '''
    y = np.exp(-1.0*((l - lp1)**2)/(2.0*w1**2))
    y += a12 * np.exp(-1.0*((l - lp2)**2)/(2.0*w2**2))
    return y

def antenna_lines(p, l):
    pd = constants.pigment_data
    lines = np.zeros((p.n_s, len(l)))
    total = np.zeros_like(l)
    for i in range(p.n_s):
        pigment = pd[p.pigment[i]]
        for j in range(pigment['n_gauss']):
            lines[i] += pigment['amp'][j] * np.exp(-1.0
                    * (l - (pigment['lp'][j] + p.lp[i]))**2
                    / (2.0 * (pigment['w'][j]**2)))
        lines[i] /= np.sum(lines[i])
        # lines[i] /= np.max(lines[i])
        total += lines[i] * p.n_p[i]
    return lines, total

def plot_nu_phi_4(nu_e, phi_e, n_s, filename):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), sharex=True)
    ax[0, 0].plot(np.arange(len(nu_e)), nu_e, label=r'$ \left<\nu_e\right> $')
    ax[0, 1].plot(np.arange(len(phi_e)), phi_e,
               label=r'$ \left<\varphi_e\right> $')
    ax[1, 0].plot(np.arange(len(phi_e)), nu_e * phi_e,
               label=r'$ \left<\nu_e \varphi_e\right> $')
    ax[1, 1].plot(np.arange(len(n_s)), n_s, label=r'$ \left<n_s\right> $')
    for a in ax.flat:
        a.legend()
    plt.legend()
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

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
    ax[1].set_ylim([0., 1.5 * n_s[-1]])
    ax[0].legend(loc='lower left')
    ax[1].legend(loc='upper left')
    ax2.legend(loc='center right')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel(r'$ \left<n_s\right> $')
    ax[0].set_ylabel(r'$ \left<\nu_e\right>, \; \left< \text{fitness} \right> $')
    ax2.set_ylabel(r'$ \left<\varphi_e\right> $')
    plt.legend()
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def hist_plot(pigment_file, peak_file, n_p_file):
    n_s = constants.hist_sub_max

    peak_hist = np.loadtxt(peak_file)
    pnames = np.loadtxt(pigment_file, usecols=0, dtype=str)
    pprops = np.loadtxt(pigment_file,
            usecols=tuple(range(1, n_s + 1)), dtype=float)
    pigment_strings = [constants.pigment_data[p]['text'] for p in pnames]
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

    fig.tight_layout()
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

    fig.tight_layout()
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
    fig.tight_layout()
    fig.savefig(os.path.splitext(peak_file)[0] + ".pdf")
    plt.close()

def plot_antenna(p, output_file):
    '''
    this is *incredibly* ugly but don't judge pls
    i couldn't find a package to draw an antenna easily in python,
    but turns out getting python and julia to play well together
    on ubuntu isn't trivial, and then importing your own module
    doesn't seem to work and i couldn't find any docs explaining
    why, and it kept throwing errors, so now i'm just doing this
    '''
    lps = [p.lp[i] +
           constants.pigment_data[p.pigment[i]]['lp'][0]
           for i in range(p.n_s)]
    cmd = "julia plot_antenna.jl --n_b {:d} --n_s {:d} ".format(p.n_b, p.n_s)\
    + "--lambdas " + " ".join(str(l) for l in lps)\
    + " --n_ps " + " ".join(str(n) for n in p.n_p)\
    + " --names " + " ".join(p for p in p.pigment)\
    + " --file " + output_file
    print(cmd)
    subprocess.run(cmd.split())

def plot_antenna_spectra(p, l, ip_y,
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
    lps = [pd[p.pigment[i]]['lp'][0] for i in range(p.n_s)]
    fig, ax = plt.subplots(figsize=(12,8))
    for i in range(p.n_s):
        # generate Gaussian lineshape for given pigment
        # and draw dotted vline at normal peak wavelength?
        color = 'C{:1d}'.format(i)
        if draw_00:
            ax.axvline(x=lps[i], color=color, ls='--')
        label = "Subunit {:1d}: ".format(i + 1) + pd[p.pigment[i]]['text']
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
    plt.plot(l, ip_y, label="Incident")
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
        for line in f:
            pass
        best = line
    print(best)
    n_b = int(re.search(r'n_b=(\d+)', best).group(1))
    n_s = int(re.search(r'n_s=(\d+)', best).group(1))
    lpm = re.search(r"lp=array\(\[\s*([0-9.\-]+[,\s\]]+)+", best).group(0)
    lpa = re.search(r"\[(.*)\]", lpm).group(0)
    lp = np.fromstring(lpa[1:-1], sep=',')
    n_pm = re.search(r"n_p=array\(\[\s*([0-9.\-]+[,\s\]]+)+", best).group(0)
    n_pa = re.search(r"\[(.*)\]", n_pm).group(0)
    n_p = np.fromstring(n_pa[1:-1], sep=',', dtype=int)
    pigm = re.search(r"pigment=array\(\[\s*([a-z_\-']+[,\s\]]+)+", best).group(0)
    piga = re.search(r"\[(.*)\]", pigm).group(0)
    # numpy doesn't know how to fromstring() this so do it manually
    pigment = np.array(piga[1:-1]
                       .replace("'", "")
                       .replace(" ", "")
                       .split(","), dtype='U10')
    return constants.Genome(n_b, n_s, n_p, lp, pigment)

def plot_best(best_file, spectrum_file):
    p = get_best_from_file(best_file)
    prefix = os.path.splitext(best_file)[0]
    output_file = prefix + "_antenna.pdf"
    lines_file = prefix + "_lines.pdf"
    total_file = prefix + "_total.pdf"

    plot_antenna(p, output_file)
    l, ip_y = np.loadtxt(spectrum_file, unpack=True)
    plot_antenna_spectra(p, l, ip_y, lines_file, total_file)

def plot_average_best(path, spectrum, out_name,
        cost, target="psii", xmin=300.0, xmax=800.0):
    if target == "psii":
        tspec = np.loadtxt("spectra/PSII.csv")
        tname = "PSII"
    elif target == "6803":
        tspec = np.loadtxt("spectra/PCC_6803_Abs.txt")
        tname = "6803"
    elif target == "6301":
        tspec = np.loadtxt("spectra/SP_6301_Abs.txt")
        tname = "6301"
    elif target == "frl":
        tspec = np.loadtxt("spectra/frl_cells.csv")
        tname = "FRL"
    elif target == "marine":
        tspec = np.loadtxt("spectra/kolodny_marine_pbs.csv")
        tname = "marine"
    else:
        tspec = None
    l = spectrum[:, 0]
    prefix = '{}/best_{}'.format(path, out_name)
    suffix = '.dat'
    total = np.zeros_like(l)
    individ = np.zeros((constants.n_runs, len(l)))
    for i in range(constants.n_runs):
        bf = prefix + "_r{:1d}".format(i) + suffix
        print(bf)
        best = get_best_from_file(bf)
        run_lines, run_tot = antenna_lines(best, l)
        individ[i] = run_tot
        total += run_tot

    total /= constants.n_runs
    norm_total = total / (np.max(total))
    outfile = prefix + "_avg_spectrum" + suffix
    np.savetxt(outfile, np.column_stack((l, norm_total)))
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(l, spectrum[:, 1], color='0.8')
    plt.plot(l, norm_total, label=r' $ \left< \text{best} \right> $')
    if tspec is not None:
        # we want the largest peak in the *visible* spectrum
        t_vis = tspec[(tspec[:, 0] > 500.0) & (tspec[:, 0] < 800.0)]
        arg = np.argmax(t_vis[:, 1])
        peak_wl = t_vis[arg, 0]
        # normalise that peak to 1
        norm_tspec = tspec[:, 1] / t_vis[arg, 1]
        plt.plot(tspec[:, 0], norm_tspec, label=tname, color='C1')
        plt.axvline(peak_wl, ls='--', color='C1')
    lmin = xmin if np.min(l) < xmin else np.min(l)
    lmax = xmax if np.max(l) > xmax else np.max(l)
    ax.set_xlabel(r' $ \lambda (\text{nm}) $')
    ax.set_ylabel("Intensity (arbitrary)")
    ax.set_title("Cost = " + str(cost))
    ax.set_xlim([lmin, lmax])
    ax.set_ylim([0.0, 1.2])
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(prefix + "_avg_spectrum.pdf")
    plt.close()
    return outfile

def plot_average_best_by_cost(files, costs, spectrum,
                              out_name, target="psii"):
    if target == "psii":
        tspec = np.loadtxt("spectra/PSII.csv")
        tname = "PSII"
    else:
        tspec = None
    l = spectrum[:, 0]
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(l, spectrum[:, 1], color='0.8')
    for c, f in zip(costs, files):
        d = np.loadtxt(f)
        plt.plot(l, d[:, 1], label="Cost = {}".format(c))
    if tspec is not None:
        plt.plot(tspec[:, 0], tspec[:, 1], label=tname)
    lmin = xmin if np.min(l) < xmin else np.min(l)
    lmax = xmax if np.max(l) > xmax else np.max(l)
    ax.set_xlim([lmin, lmax])
    ax.set_ylim([0.0, 1.2])
    ax.set_xlabel(r' $ \lambda (\text{nm}) $')
    ax.set_ylabel("Intensity (arbitrary)")
    plt.grid(visible=True, axis='both')
    plt.legend(fontsize=20)
    fig.tight_layout()
    plt.savefig(
    "{}{}_avg_spectrum_by_cost.pdf".format(constants.output_dir,
                                            out_name))
    plt.close()

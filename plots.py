# -*- coding: utf-8 -*-
"""
29/11/2023
@author: callum

polygon_under_graph and the 3d plot generally stolen from matplotlib
"""
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import constants
from matplotlib.collections import PolyCollection
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

def antenna_plot_2d(genome, spectrum, filename):
    '''
    take a genome and plot the corresponding set of Gaussians,
    one for each subunit, along with the stellar spectrum.
    spectrum should be given as np.loadtxt(phoenix_data) i.e. two columns
    '''
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlabel(r'$ \lambda (nm) $')
    ax.set_ylabel(r'$ I $')
    plt.plot(spectrum[:, 0], spectrum[:, 1], color='k',
             label=r'$ I_p $')
    cm = plt.colormaps['jet_r'](np.linspace(0, 1, genome.n_s))
    for i in range(genome.n_s):
        a = two_gauss(spectrum[:, 0], genome.lp1[i], genome.w1[i],
                genome.lp2[i], genome.w2[i], genome.a12[i])
        plt.plot(spectrum[:, 0], a * 0.8 * np.max(spectrum[:, 1]),
                 color=cm[i])
    xlim = (0.5 * np.min(genome.lp1), 1.5 * np.max(genome.lp1))
    ax.set_xlim(xlim)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def antenna_plot_3d(genome, spectrum, filename):
    '''
    take a genome and plot the corresponding set of Gaussians,
    one for each subunit, along with the stellar spectrum.
    spectrum should be given as np.loadtxt(phoenix_data) i.e. two columns
    '''
    ax = plt.figure().add_subplot(projection='3d')
    zs = range(1, genome.n_s + 2)
    xlim = (np.min(spectrum[:, 0]), np.max(spectrum[:, 0]))
    # need to cut off the actual spectrum array to use the next line
    # xlim = (0.5 * np.min(genome.lp), 1.5 * np.max(genome.lp))
    cm = plt.colormaps['jet_r'](np.linspace(0, 1, genome.n_s))
    verts = []
    for i in range(genome.n_s):
        verts.append(polygon_under_graph(spectrum[:, 0],
            two_gauss(spectrum[:, 0], genome.lp1[i], genome.w1[i],
                genome.lp2[i], genome.w2[i], genome.a12[i])))
    verts.append(polygon_under_graph(spectrum[:, 0], spectrum[:, 1]))
    cm = np.append(cm, [[0., 0., 0., 1.]], axis=0) # black
    p = PolyCollection(verts, facecolors=cm, alpha=.7)
    ax.add_collection3d(p, zs=zs, zdir='y')
    ax.set(xlim=xlim, ylim=(1, genome.n_s + 1), zlim=(0.0, 5.0),
           xlabel=r'$ \lambda (nm) $', zlabel=r'$ I $')
    # plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_nu_phi_4(nu_e, phi_f, n_s, filename):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), sharex=True)
    ax[0, 0].plot(np.arange(len(nu_e)), nu_e, label=r'$ \left<\nu_e\right> $')
    ax[0, 1].plot(np.arange(len(phi_f)), phi_f,
               label=r'$ \left<\varphi_f\right> $')
    ax[1, 0].plot(np.arange(len(phi_f)), nu_e * phi_f,
               label=r'$ \left<\nu_e \varphi_f\right> $')
    ax[1, 1].plot(np.arange(len(n_s)), n_s, label=r'$ \left<n_s\right> $')
    for a in ax.flat:
        a.legend()
    plt.legend()
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_nu_phi_2(nu_e, phi_f, n_s, filename):
    fig, ax = plt.subplots(nrows=2, figsize=(12,8), sharex=True)
    ax[0].plot(np.arange(len(nu_e)), nu_e, color='C0',
               label=r'$ \left<\nu_e\right> $')
    ax2 = ax[0].twinx()
    ax2.plot(np.arange(len(phi_f)), phi_f, color='C1',
               label=r'$ \left<\varphi_f\right> $')
    ax[0].plot(np.arange(len(phi_f)), nu_e * phi_f, color='C2',
               label=r'$ \left<\nu_e \varphi_f\right> $')
    ax[1].plot(np.arange(len(n_s)), n_s, label=r'$ \left<n_s\right> $')
    ax[1].set_ylim([0., 1.5 * n_s[-1]])
    ax[0].legend(loc='lower left')
    ax2.legend(loc='center right')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel(r'$ \left<n_s\right> $')
    ax[0].set_ylabel(r'$ \left<\nu_e\right>, \; \left<\nu_e\varphi_f\right> $')
    ax2.set_ylabel(r'$ \left<\varphi_f\right> $')
    plt.legend()
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def draw_antenna(p, outfile):
    # Main.eval('include("DrawAntennae.jl")')
    lambdas = [p.lp[i] + constants.pigment_data[p.pigment[i]]['lp'][0] for i in range(p.n_s)]
    names = [constants.pigment_data[pig]['name'] for pig in p.pigment]
    # DrawAntennae.plot(p.n_b, p.n_s, lambdas, p.n_p, names, outfile)


def hist_plot(pigment_file, lp_file):
    n_s = constants.hist_sub_max

    lh = np.loadtxt(lp_file)
    pnames = np.loadtxt(pigment_file, usecols=0, dtype=str)
    pprops = np.loadtxt(pigment_file,
            usecols=tuple(range(1, n_s + 1)), dtype=float)
    pigment_strings = [constants.pigment_data[p]['name'] for p in pnames]
    n_pigments = len(pigment_strings)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')

    for k in reversed(range(n_s)):
        ys = pprops[:, k]
        xs = np.arange(n_pigments)
        color = 'C{:1d}'.format(k)
        ax.bar(xs, ys, zs=k, zdir='y', color=color, alpha = 0.7)
        
        #fig, ax = plt.subplots(figsize=(12,8))
        #ax.bar(xs, ys, color=color)
        #ax.set_ylabel("Proportion")
        #ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))
        #ax.set_ylim([0.0, 1.0])
        #ax.set_xticklabels(yticklabels)
     
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        #axis._axinfo['label']['space_factor'] = 2.5
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

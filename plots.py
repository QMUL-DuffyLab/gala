# -*- coding: utf-8 -*-
"""
29/11/2023
@author: callum

polygon_under_graph and the 3d plot generally stolen from matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
import constants
from matplotlib.collections import PolyCollection

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
    cm = plt.colormaps['jet'](np.linspace(0, 1, genome.n_s))
    for i in range(genome.n_s):
        a = gauss(spectrum[:, 0], genome.lp[i], genome.w[i])
        plt.plot(spectrum[:, 0], a * 0.8 * np.max(spectrum[:, 1]),
                 color=cm[i])
    xlim = (0.5 * np.min(genome.lp), 1.5 * np.max(genome.lp))
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
    cm = plt.colormaps['jet'](np.linspace(0, 1, genome.n_s))
    verts = []
    for i in range(genome.n_s):
        verts.append(polygon_under_graph(spectrum[:, 0], gauss(spectrum[:, 0],
                                               genome.lp[i], genome.w[i])))
    verts.append(polygon_under_graph(spectrum[:, 0], spectrum[:, 1]))
    cm = np.append(cm, [[0., 0., 0., 1.]], axis=0) # black
    p = PolyCollection(verts, facecolors=cm, alpha=.7)
    ax.add_collection3d(p, zs=zs, zdir='y')
    ax.set(xlim=xlim, ylim=(1, genome.n_s + 1), zlim=(0.0, 5.0),
           xlabel=r'$ \lambda (nm) $', zlabel=r'$ I $')
    # plt.legend()
    plt.savefig(filename)
    plt.close()

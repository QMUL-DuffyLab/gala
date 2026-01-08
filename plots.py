# -*- coding: utf-8 -*-
"""
08/01/2026
@author: callum
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constants
import utils
import genetic_algorithm as ga

column_width = 2.0
pad = 0.25 * column_width
lw = 2.0

# plots v2 for RC stuff :)
def plot_lines(p, output_file):
    n_lines = np.sum(p['n_t']) + constants.n_rc + 2
    width = (column_width + pad) * n_lines
    fig, ax = plt.subplots(figsize=(width, 10))
    curr_x = pad
    ax.plot([curr_x, curr_x + column_width],
            [constants.e_donor, constants.e_donor],
            c='k', lw=lw)
    dlabel = r'$ \epsilon_{\text{donor}} $'
    ax.annotate(dlabel, xy=(curr_x + 0.5*column_width,
                            constants.e_donor - 0.1),
                xycoords='data', color='k', va='top', ha='center')
    curr_x += (column_width + pad)
    for rci in range(constants.n_rc):
        ion = -1.0 * p['i'][rci]
        excited = ion + p['dE0'][rci]
        it = f"{rci:1d}"
        ilabel = r'$-I_{' + it +  r'}$'
        elabel = r'$-I_{' + it + r'} + \Delta E_{' + it + '}$'
        ax.plot([curr_x, curr_x + column_width],
                [ion, ion],
                c='k', lw=lw, label=ilabel)
        ax.annotate(ilabel, xy=(curr_x + 0.5*column_width, ion-0.1),
                    xycoords='data', color='k', va='top', ha='center')
        ax.plot([curr_x, curr_x + column_width],
                [excited, excited],
                c='k', lw=lw, label=elabel)
        ax.annotate(elabel, xy=(curr_x + 0.5*column_width, excited + 0.1),
                    xycoords='data', color='k', va='bottom', ha='center')
        # arrow from one to the other
        linex = curr_x + column_width/2.0
        ax.plot([linex, linex],
                [ion + 0.1, excited - 0.1],
                c='r', lw=lw)
        elabel = r'$ \Delta E_{' + it + '}$ = ' + f"{p['dE0'][rci]:4.2f} eV"
        # make arrowhead
        trongle_x = [linex, linex + 0.25, linex - 0.25]
        trongle_y = [excited - 0.1, excited - 0.2, excited - 0.2]
        plt.fill(trongle_x, trongle_y, c='r')
        ax.annotate(elabel, xy=(linex - 0.1, 
                                0.5 * (ion + 0.1 + (excited - 0.1))),
                    xycoords='data', color='r', va='center', ha='right',
                    fontsize='xx-small')
        curr_x += (column_width + pad)
        # traps
        for ti in range(p['n_t'][rci]):
            ax.plot([curr_x, curr_x + column_width],
                    [p['e'][rci][ti], p['e'][rci][ti]],
                    c='k', lw=lw/2.0, alpha=0.6)
            curr_x += (column_width + pad)
    ax.plot([curr_x, curr_x + column_width],
            [constants.e_acceptor, constants.e_acceptor],
            c='k', lw=lw)
    alabel = r'$ \epsilon_{\text{acceptor}} $'
    ax.annotate(alabel, xy=(curr_x + 0.5*column_width,
                            constants.e_acceptor - 0.1),
                xycoords='data', color='k', va='top', ha='center')

    ax.set_xticks([])
    # extend ylims down a bit to accomodate labels
    ymin, ymax = ax.get_ylim()
    ymin -= 0.5
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('E (eV)')
    plt.grid(visible=False)
    # fig.savefig(output_file)
    plt.show()
    plt.close()

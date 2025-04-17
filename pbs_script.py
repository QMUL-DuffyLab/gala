# -*- coding: utf-8 -*-
"""
17/04/2025
@author: callum

"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import constants
import light
import rc
import supersystem
import antenna as la

if __name__ == "__main__":
    '''
    here's a little script to set up a PBS-like antenna-RC system.
    one thing i need to sort out is the hopping rate between the
    PBS subunits and the RCs, which i think is supposed to be longer
    than i've been using for the astro stuff; it's on the to-do list.
    the next block is all the parameters that'll need varying.
    outputs a little file with details about the performance of the
    PBS and a plot of the lineshapes.

    if we need to we can do something cleverer, e.g. looping over
    sets of parameters we want to check and adding results to dataframes
    or something like that, but for now you can at least do some testing
    '''
    colour = "orange"
    intensity = 150.0 # mu_E
    n_b = 5 # number of branches of the antenna
    pigment = ['apc', 'pc', 'pe'] # change this as you like
    n_s = len(pigment)
    # can change this to change relative abundance of pigments
    # obviously it must have the same length as pigment above
    n_p = [70, 70, 70]
    # don't change this - red/blueshifts the pigment absorptions
    shift = [0 for _ in range(n_s)]
    # stoichiometries - [PSII, PSI, PBS]
    # they must sum to 3!
    rho = [1.0, 1.0, 1.0]
    # affinities - [PSII, PSI]
    aff = [2.0, 1.0]

    # the solve functions use a Genome instance internally so
    # create one here. none of this should need changing
    p = constants.Genome(n_b, n_s, n_p, shift,
            pigment, "ox", rho, aff)
    # call light.py to pull the right spectrum and fix intensity
    spectrum, output_prefix = light.spectrum_setup("colour",
            colour=colour, intensity=intensity)
    # solve the whole big matrix
    od = supersystem.solve(spectrum[:, 0], spectrum[:, 1], p, True)

    # 
    side = len(od["p_eq"])
    print(f"alpha (cyc ratio) = {constants.alpha}, rho = {rho}, aff = {aff}")
    print(f"p(0) = {od['p_eq'][0]}")
    print(f"nu_ch2o = {od['nu_ch2o']}")
    print(f"nu_cyc = {od['nu_cyc']}")

    n_rc = len(rc.params["ox"]["pigments"])
    # total photons in
    sg = np.sum(od['gamma'][:n_rc]) + n_b * np.sum(od['gamma'][n_rc:])
    print(f"total excitation rate = {sg} s^-1")
    print(f"'efficiency' = {(od['nu_ch2o'] + od['nu_cyc']) / sg}")
    # output some details of the solution
    with open(f"out/pbs_{colour}_{intensity}_results.dat", "w") as f:
        f.write(f"pigments = {pigment}\n")
        f.write(f"n_p = {n_p}\n")
        f.write(f"alpha = {constants.alpha}, rho = {rho}, affinity = {aff}\n")
        f.write(f"electrons/carbons out (nu_ch2o) = {od['nu_ch2o']}\n")
        f.write(f"nu_cyc = {od['nu_cyc']}\n")
        f.write(f"sum(gamma) = {np.sum(od['gamma'])}\n")
        f.write(f"'efficiency' = {(od['nu_ch2o'] + od['nu_cyc']) / sg}\n")
        for si, pi in zip(od["states"], od["p_eq"]):
            f.write(f"p_eq{si} = {pi}\n")
        f.write(f"kb = {od['k_b']}\n")
        f.write(f"photon input = {od['gamma']}\n")
        f.write(f"p(0) = {od['p_eq'][0]}\n")
    # plot the lineshapes
    names = rc.params["ox"]["pigments"] + pigment
    fig, ax = plt.subplots(nrows=len(names),
            figsize=(12,12), sharex=True)
    for i in range(len(names)):
        ax[i].plot(spectrum[:, 0], od['a_l'][i],
                color='C1', label=f"A ({names[i]})")
        ax[i].plot(spectrum[:, 0], od['e_l'][i],
                color='C0', label=f"F ({names[i]})")
        ax[i].legend()
        ax[i].grid(visible=True)
    fig.supylabel("intensity (arb)", x=0.001)
    ax[0].set_xlim(constants.x_lim)
    ax[-1].set_xlabel("wavelength (nm)")
    fig.savefig(f"out/pbs_lineshapes.pdf")
    plt.close(fig)

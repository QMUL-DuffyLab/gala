# -*- coding: utf-8 -*-
"""
12/03/2024
@author: callum
"""
import numpy as np
import matplotlib.pyplot as plt
import constants
import antenna as la

'''
various helper functions to import or generate spectra
'''

def get_phoenix_spectrum(ts):
    return np.loadtxt(constants.spectrum_prefix
            + constants.phoenix_prefix
            + '{:4d}K'.format(ts)
            + constants.spectrum_suffix)

def get_cwf(mu_e = 1.0):
    '''
    get cool white fluorescent spectrum with given intensity
    in micro Einsteins. assumes it's currently normalised to 1,
    which it is not yet!
    '''
    fluo = np.loadtxt(constants.spectrum_prefix
            + 'anderson_2003_cool_white_fluo.csv')
    return mu_e * fluo

def get_gaussian(l, lp, w, a):
    '''
    return a normalised lineshape made up of Gaussians.
    TO DO: figure out micro einstein normalisation
    and how it should format its output files - should probably
    also make a plot of the incident spectra to go with output
    '''
    return la.gauss(l, lp, w, a)

def get_am15(dataset="tilt"):
    d = np.loadtxt(constants.spectrum_prefix + "ASTMG173.csv",
                   skiprows=2, delimiter=",")
    if dataset == "tilt":
        am15 = np.column_stack((d[:, 0], d[:, 2]))
    elif dataset == "ext":
        am15 = np.column_stack((d[:, 0], d[:, 1]))
    elif dataset == "circum":
        am15 = np.column_stack((d[:, 0], d[:, 3]))
    else:
        raise KeyError("Invalid column key provided to get_am15")
    return am15

def spectrum_setup(spectrum_type, **kwargs):
    if spectrum_type == "phoenix":
        s = get_phoenix_spectrum(kwargs['temperature'])
        out_pref = "{:4d}K".format(kwargs['temperature'])
    elif spectrum_type == "fluo":
        s = get_cwf(kwargs['mu_e'])
        out_pref = "cwf_{:8.3e}_mu_ein".format(kwargs['mu_e'])
    elif spectrum_type == "am15":
        s = get_am15(kwargs['dataset'])
        out_pref = "am15_{}".format(kwargs['dataset'])
    elif spectrum_type == "gauss":
        l = np.arange(kwargs['lmin'], kwargs['lmax'])
        intensity = get_gaussian(l, kwargs['lp'], kwargs['w'], kwargs['a'])
        s = np.column_stack((l, intensity))
        out_pref = "gauss_lp0_{:6.2f}".format(kwargs['lp'][0])
    else:
        raise ValueError("Invalid call to spectrum_setup.")
    return s, out_pref

def build(spectra_dicts):
    '''
    pass a list of dicts to this function to get back
    a tuple of the spectra and output file prefixes.
    '''
    spectra = []
    out_prefs = []
    for sp in spectra_dicts:
        s, out_pref = spectrum_setup(sp['type'], **sp['kwargs'])
        spectra.append(s)
        out_prefs.append(out_pref)
    return zip(spectra, out_prefs)

def check(spectra_dicts):
    z = build(spectra_dicts)
    for spectrum, out_pref in z:
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(spectrum[:, 0], spectrum[:, 1])
        ax.set_xlabel(r'$ \lambda (\text{nm}) $')
        ax.set_ylabel(r'Intensity')
        ax.set_title(out_pref)
        fig.savefig(out_pref + "_test_plot.pdf")
        plt.close()

if __name__ == "__main__":
    sd = [
          {'type': "phoenix", 'kwargs': {'temperature': 5800}},
          {'type': "fluo", 'kwargs': {'mu_e': 10.0}},
          {'type': "phoenix", 'kwargs': {'temperature': 2300}},
          {'type': "fluo", 'kwargs': {'mu_e': 0.01}},
          {'type': "fluo", 'kwargs': {'mu_e': 100.0}},
          {'type': "phoenix", 'kwargs': {'temperature': 4800}},
          {'type': "gauss", 'kwargs': {'lmin': 200.0, 'lmax': 1000.0, 'lp': [600.0, 500.0], 'w': [15.0, 35.0], 'a': [1.0, 0.2]}},
          ]
    check(sd)


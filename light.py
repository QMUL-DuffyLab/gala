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
    '''
    return scaled PHOENIX spectrum representing star of given temperature
    '''
    return np.loadtxt(constants.spectrum_prefix
            + constants.phoenix_prefix
            + '{:4d}K'.format(ts)
            + constants.spectrum_suffix)

def get_cwf(mu_e = 1.0):
    '''
    get cool white fluorescent spectrum with given intensity
    in micro Einsteins - the spectrum in the file's normalised to 1μE
    '''
    l, fluo = np.loadtxt(constants.spectrum_prefix
            + 'anderson_2003_cool_white_fluo.csv', unpack=True)
    return np.column_stack((l, mu_e * fluo))

def get_gaussian(l, lp, w, a):
    '''
    return a normalised lineshape made up of Gaussians.
    TO DO: figure out micro einstein normalisation
    and how it should format its output files - should probably
    also make a plot of the incident spectra to go with output
    '''
    return la.gauss(l, lp, w, a)

def get_am15(dataset="tilt"):
    '''
    return relevant am15 dataset. standard spectrum. taken from
    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
    '''
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

def get_marine(**kwargs):
    '''
    simple function to take am1.5 spectrum and approximate
    light attenuation as a function of wavelength in water.
    taken from doi:10.1038/ismej.2007.59
    note that we (currently) ignore gilvin/tripton/phytoplankton terms
    since we're not trying to model any specific ocean precisely.
    '''
    if 'dataset' in kwargs.keys():
        am15 = get_am15(kwargs['dataset'])
    else:
        am15 = get_am15()
    if 'depth' in kwargs.keys():
        z = kwargs['depth']
    else:
        raise KeyError("Marine spectrum must be given depth parameter")
    water = np.loadtxt(constants.spectrum_prefix +
                       "water_absorption.csv", skiprows=1, delimiter=",")
    water_interp = np.interp(am15[:, 0], water[:, 0], water[:, 1])
    ilz = am15[:, 1] * np.exp(-1.0 * z * water_interp)
    return np.column_stack((am15[:, 0], ilz))

def get_filtered(**kwargs):
    '''
    return a red or far-red filtered AM1.5 spectrum.
    digitised from the red and far-red filters in
    https://dx.doi.org/10.1007/s11120-016-0309-z (Fig. S1)
    '''
    if 'dataset' in kwargs.keys():
        am15 = get_am15(kwargs['dataset'])
    else:
        am15 = get_am15()
    if 'filter' not in kwargs.keys():
        raise KeyError("Filtered spectrum must have filter kwarg")
    if kwargs['filter'] == "red":
        f = np.loadtxt(constants.spectrum_prefix + "filter_red.csv",
                       delimiter=',')
    if kwargs['filter'] == "far-red":
        f = np.loadtxt(constants.spectrum_prefix + "filter_far_red.csv",
                       delimiter=',')
    f_interp = np.interp(am15[:, 0], f[:, 0], f[:, 1])
    transmitted = am15[:, 1] * f_interp
    return np.column_stack((am15[:, 0], transmitted))

def spectrum_setup(spectrum_type, **kwargs):
    '''
    wrap the above functions and return a consistent tuple.
    the actual function call could maybe be streamlined a bit
    but the output prefix is dependent on specific kwargs
    '''
    if spectrum_type == "phoenix":
        s = get_phoenix_spectrum(kwargs['temperature'])
        output_prefix = "{:4d}K".format(kwargs['temperature'])
    elif spectrum_type == "fluo":
        s = get_cwf(kwargs['mu_e'])
        output_prefix = "cwf_{:8.3e}_mu_ein".format(kwargs['mu_e'])
    elif spectrum_type == "am15":
        s = get_am15(kwargs['dataset'])
        output_prefix = "am15_{}".format(kwargs['dataset'])
    elif spectrum_type == "marine":
        s = get_marine(**kwargs)
        output_prefix = "marine_z_{}".format(kwargs['depth'])
    elif spectrum_type == "filtered":
        s = get_filtered(**kwargs)
        output_prefix = kwargs['filter']
    elif spectrum_type == "gauss":
        l = np.arange(kwargs['lmin'], kwargs['lmax'])
        intensity = get_gaussian(l, kwargs['lp'], kwargs['w'], kwargs['a'])
        s = np.column_stack((l, intensity))
        output_prefix = "gauss_lp0_{:6.2f}".format(kwargs['lp'][0])
    else:
        raise ValueError("Invalid call to spectrum_setup.")
    return s, output_prefix

def build(spectra_dicts):
    '''
    pass a list of dicts to this function to get back
    a tuple of the spectra and output file prefixes.
    note that where there are required parameters for the spectrum,
    e.g. temperature for the PHOENIX or depth for a marine spectrum,
    you can't pass them as an array, at least not yet. use one line for
    each value you want to simulate and just copy-paste.
    '''
    spectra = []
    out_prefs = []
    for sp in spectra_dicts:
        s, out_pref = spectrum_setup(sp['type'], **sp['kwargs'])
        spectra.append(s)
        out_prefs.append(out_pref)
    return zip(spectra, out_prefs)

def check(spectra_dicts):
    '''
    take a list of dicts and plot the spectrum that would be used in
    simulations - cuts off at 200 and 1000nm since that's really what
    we're interested in, but only for the plot; the whole spectrum's kept.
    outputs the plots using the output prefix that'll be generated as well.
    '''
    z = build(spectra_dicts)
    for spectrum, out_pref in z:
        np.savetxt(constants.output_dir + out_pref + "_spectrum.dat", spectrum)
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(spectrum[:, 0], spectrum[:, 1])
        smin = np.min(spectrum[:, 0])
        smax = np.max(spectrum[:, 0])
        xmin = 200.0 if smin < 200.0 else smin
        xmax = 1000.0 if smax > 1000.0 else smax
        ax.set_xlabel(r'$ \lambda (\text{nm}) $')
        ax.set_xlim([xmin, xmax])
        ax.set_ylabel(r'Intensity')
        ax.set_title(out_pref)
        fig.savefig(constants.output_dir + out_pref + "_test_plot.pdf")
        plt.close()

if __name__ == "__main__":
    '''
    examples of dicts - run `python light.py` to get spectra/plots
    '''
    sd = [
          {'type': "fluo", 'kwargs': {'mu_e': 100.0}},
          {'type': "filtered", 'kwargs': {'filter': "far-red"}},
          {'type': "phoenix", 'kwargs': {'temperature': 4800}},
          {'type': "am15", 'kwargs': {'dataset': "tilt"}},
          {'type': "marine", 'kwargs': {'depth': 10.0}},
          {'type': "gauss", 'kwargs':
           {'lmin': 200.0, 'lmax': 1000.0, 'lp': [600.0, 500.0],
            'w': [15.0, 35.0], 'a': [1.0, 0.2]}},
          ]
    check(sd)


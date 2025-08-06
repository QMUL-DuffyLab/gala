# -*- coding: utf-8 -*-
"""
12/03/2024
@author: callum

light.py
various helper functions to import or generate spectra.
I've tried to be reasonably consistent here: each different kind of
spectrum has its own function with (hopefully) a fairly obvious name.
Each one takes **kwargs, and the functions check that any required
arguments are there with a try/except block of the form
```
try: 
    required_parameter = kwargs['required_parameter']
except KeyError: 
    print("invalid parameter") # print out relevant stuff here
    raise # re-raise the exception
```
The reason for this is so that I can write one wrapper function,
spectrum_setup, which just takes the function name and the required
arguments and returns the spectrum and output prefix for the files,
which hopefully makes it easier for the user.

TO DO:
- figure out a more consistent way of dealing with optional
arguments or arguments for which there's a choice of options.
- standardise the docstrings
- write a little helper bit in spectrum_setup that returns the list of
possible functions? maybe using inspect.get_members or something, idk yet
"""
import os
import sys
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants
import utils
from scipy.constants import h, c, Avogadro, Boltzmann


# AU in metres for stellar spectra
AU = 149597870700

Rsol = 6.957E8
Msol = 1.989E30
Lsol = 3.828E26
# (Tstar, Lstar/Lsol, Rstar/Rsol) - 4Gyr isochrones from Isabelle Baraffe
phz_stars = [
 # (2343,  -3.58,  0.099),
 (2811,  -3.06,  0.124),
 # (3131,  -2.58,  0.174),
 # (3416,  -1.97,  0.297),
 # (3680,  -1.45,  0.465),
 (4418,  -0.85,  0.646),
 (5297,  -0.29,  0.850),
 # (5697,  -0.03,  0.994),
 # (6053,   0.43,  1.491)
 ]

def calculate_phz_radii(Tstar, Lstar, n_radii=20,
        inner="runaway greenhouse", outer="maximum greenhouse"):
    '''
    for a given stellar temperature and luminosity,
    (cf. Baraffe's isochrones at
    https://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_iso.2mass)
    use the expression and calculated coefficients from Koppaparu et al.
    (doi:10.1088/0004-637X/765/2/131) to get the inner and outer HZ for
    an Earth-like planet. Then divide the HZ up into n_radii slices, and
    return a list of those orbital radii in AU.
    Code adapted from from https://github.com/casshall/phz2
    '''
    #These are the values for Sun system.
    seffsun = [1.776,1.107, 0.356, 0.320, 1.188, 0.99]

    # Other coefficients
    a = [2.136e-4, 1.332e-4, 6.171e-5,
         5.547e-5, 1.433e-4, 1.209e-4]
    b = [2.533e-8, 1.580e-8, 1.698e-9,
         1.526e-9, 1.707e-8, 1.404e-8]
    c = [-1.332e-11, -8.308e-12, -3.198e-12,
         -2.874e-12, -8.968e-12, -7.418e-12]
    d = [-3.097e-15, -1.931e-15, -5.575e-16,
         -5.011e-16, -2.084e-15, -1.713e-15]

    #Initialise the seff arrays and for the HZ
    seff = np.zeros(len(a))
    distance_line = np.zeros(len(a))
    for j in range(len(a)):
        #From Kopparapu equation 2, Tstar = T_effective - 5780K
        Ts = Tstar - 5780
        seff[j] = (seffsun[j] + a[j] * Ts + b[j] * Ts**2
                + c[j]*Ts**3 + d[j]*Ts**4)
        distance_line[j] =  ((10**Lstar) / seff[j])**0.5

    conditions = ["recent Venus", "runaway greenhouse", "maximum greenhouse",
    "early Mars", "runaway greenhouse (5ME)", "runaway greenhouse (0.1ME)"]
    try:
       inner_index = conditions.index(inner)
       inner_HZ = distance_line[inner_index]
    except ValueError:
        print(f"argument 'inner' to light.calculate_phz_radii is incorrect.")
        print(f"passed value: {inner}. Allowed options: {conditions}")
        raise
    try:
       outer_index = conditions.index(outer)
       outer_HZ = distance_line[outer_index]
    except ValueError:
        print(f"argument 'outer' to light.calculate_phz_radii is incorrect.")
        print(f"passed value: {outer}. Allowed options: {conditions}")
        raise
    return np.logspace(np.log10(inner_HZ),
            np.log10(outer_HZ), num=n_radii)

def micromole_in_region(spectrum, lower, upper):
    '''
    calculate the light intensity in micromoles over the wavelength
    range (lower, upper). assumes nm, but since all our spectra are
    given in nm that shouldn't matter. can use this to increase or
    decrease light intensity in a given range as much as we like
    '''
    muM = 0.0
    for row in spectrum:
        if row[0] >= lower and row[0] <= upper:
            e_per_photon = h * c / (row[0] * 1e-9)
            muM += row[1] / e_per_photon
    return 1e6 * (muM / Avogadro)

def phoenix(**kwargs):
    '''
    return scaled PHOENIX spectrum for star of given temperature
    '''
    try:
        T = kwargs["Tstar"]
    except KeyError:
        print("Invalid kwargs in light.phoenix()")
        print("Expected: 'Tstar'")
        print(f"Got: {kwargs}")
        raise
    output_prefix = f"phoenix_{T:4d}K"
    spectrum = np.loadtxt(os.path.join(constants.spectrum_prefix,
            "PHOENIX",
            f"Scaled_Spectrum_PHOENIX_{T:4d}K.dat"))
    return spectrum, output_prefix

def stellar(**kwargs):
    '''
    Generate irradiance for arbitrary star at arbitrary distance.
    kwargs:
    - Tstar:       temperature of star in K
    - Rstar:       radius of star in units of solar radius
    - a:           orbital distance (semi-major axis) in AU
    - attenuation: attenuation factor (should be in [0, 1])
    '''
    try:
        T = kwargs["Tstar"]
        R = kwargs["Rstar"] * Rsol
        a = kwargs["a"]
        att = kwargs["attenuation"]
    except KeyError:
        print("Invalid kwargs in light.stellar()")
        print("Expected: 'Tstar', 'Rstar', 'a', 'attenuation'")
        print(f"Got: {kwargs}")
        raise
    output_prefix = f"stellar_{T:4d}K_a_{a:6.4f}AU"
    a *= AU # convert to metres
    lambdas = np.linspace(*constants.x_lim, constants.nx)
    # spectral irradiance (W m^{-2} nm^{-1})
    lambdas *= 1e-9 # convert to SI
    sp = 1.0E-9 * np.pi * ((2.0 * h * c**2 / lambdas**5.0)
    / (np.exp(h * c / (lambdas * Boltzmann * T)) -1.))
    # irradiance at distance a
    irr = sp * (R / a)**2
    return np.column_stack((lambdas * 1e9, irr)), output_prefix
    
def gaussian(**kwargs):
    '''
    return a normalised lineshape made up of Gaussians.
    intensity kwarg is not used here but I check if it's present
    as with the colour function below, because it should always
    be given for this type of spectrum
    '''
    try:
        mu = kwargs["mu"]
        sigma = kwargs["sigma"]
        a = kwargs["a"]
        intensity = kwargs["intensity"]
    except KeyError:
        print("Invalid kwargs in light.gaussian()")
        print("Expected: 'mu', 'sigma', 'a', 'intensity'")
        print(f"Got: {kwargs}")
        raise
    # this will be a mess if these are lists
    output_prefix = f"gauss_{a}_{mu}_{sigma}_{intensity}"
    lambdas = np.linspace(*constants.x_lim, constants.nx)
    gauss = utils.gauss(lambdas, mu, sigma, a)
    return np.column_stack((lambdas, gauss)), output_prefix

def colour(**kwargs):
    '''
    get one of the specific colour spectra from Samir's experimental
    setup. possible choices are listed below. I'm just reading in the
    100 muE dataset here using pandas and then normalising the
    intensity.
    Note that the intensity normalisation is done outside these
    functions in the wrapper because we can apply it to any spectrum,
    but it should always be given for these! Raise an error if it isn't.
    '''
    try:
        col = kwargs["colour"]
        intensity = kwargs["intensity"]
    except KeyError:
        print("Invalid kwargs in light.colour()")
        print("Expected: 'colour', 'intensity'")
        print(f"Got: {kwargs}")
        raise
    colours = ["UV", "blue", "light_blue", "green", "orange",
            "red", "far_red"]
    if col not in colours:
        raise KeyError("Invalid colour choice in light.colour()")
    output_prefix = f"colour_{col}"
    data = pd.read_csv(os.path.join(constants.spectrum_prefix,
        "100_plotted_dataset.csv"))
    x = data[f"Wavelength"].to_numpy()
    y = data[f"{col}_normalized"].to_numpy()
    y[y < 0.0] = 0.0 # input data has some small negative values
    return np.column_stack((x, y)), output_prefix

def am15(**kwargs):
    '''
    return relevant am15 dataset. standard spectrum. taken from
    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
    '''
    try:
        dataset = kwargs["dataset"]
    except KeyError:
        print("Invalid kwargs in light.am15()")
        print("Expected: 'dataset'")
        print(f"Got: {kwargs}")
        raise
    d = np.loadtxt(os.path.join(constants.spectrum_prefix,
                    "ASTMG173.csv"), skiprows=2, delimiter=",")
    if dataset == "tilt":
        spectrum = np.column_stack((d[:, 0], d[:, 2]))
    elif dataset == "ext":
        spectrum = np.column_stack((d[:, 0], d[:, 1]))
    elif dataset == "circum":
        spectrum = np.column_stack((d[:, 0], d[:, 3]))
    else:
        raise KeyError("Invalid column key provided to am15")
    output_prefix = f"am15_{dataset}"
    return spectrum, output_prefix

def marine(**kwargs):
    '''
    simple function to take am1.5 spectrum and approximate
    light attenuation as a function of wavelength in water.
    taken from doi:10.1038/ismej.2007.59
    note that we (currently) ignore gilvin/tripton/phytoplankton terms
    since we're not trying to model any specific ocean precisely.
    '''
    try:
        depth = kwargs['depth']
    except KeyError:
        print("Invalid kwargs in light.marine()")
        print("Expected: 'depth'")
        print(f"Got: {kwargs}")
        raise
    if 'dataset' in kwargs:
        spectrum, _ = am15(**kwargs)
    else:
        spectrum, _ = am15(dataset="tilt")
    output_prefix = f"marine_depth_{depth}"
    water = np.loadtxt(os.path.join(constants.spectrum_prefix,
                       "water_absorption.csv"), skiprows=1, delimiter=",")
    water_interp = np.interp(spectrum[:, 0], water[:, 0], water[:, 1])
    ilz = spectrum[:, 1] * np.exp(-1.0 * depth * water_interp)
    return np.column_stack((spectrum[:, 0], ilz)), output_prefix

def filtered(**kwargs):
    '''
    return a red or far-red filtered AM1.5 spectrum.
    digitised from the red and far-red filters in
    https://dx.doi.org/10.1007/s11120-016-0309-z (Fig. S1)
    '''
    try:
        fil = kwargs['filter']
    except KeyError:
        print("Invalid kwargs in light.filtered()")
        print("Expected: 'filter'")
        print(f"Got: {kwargs}")
        raise
    if 'dataset' in kwargs:
        spectrum, _ = am15(**kwargs)
    else:
        spectrum, _ = am15(dataset="tilt")
    if fil in ['red', 'far_red']:
        f = np.loadtxt(os.path.join(constants.spectrum_prefix,
            f"filter_{fil}.csv"), delimiter=',')
    else:
        raise KeyError("Invalid filter {fil} provided to light.filtered()")
    f_interp = np.interp(spectrum[:, 0], f[:, 0], f[:, 1])
    output_prefix = f"filtered_{fil}"
    if 'fraction' in kwargs:
        frac = kwargs['fraction']
        output_prefix += f"_fraction_{frac}"
    else:
        frac = 1.
    # (1 - frac) unfiltered + frac * filtered
    transmitted = (((1.0 - frac) * spectrum[:, 1])
            + (frac * spectrum[:, 1] * f_interp))
    return np.column_stack((spectrum[:, 0], transmitted)), output_prefix

def spectrum_setup(spectrum_type, **kwargs):
    '''
    wrap the above functions and return a consistent tuple.
    in principle changing the intensity of the spectrum has nothing
    to do with what kind of spectrum it is, so i have just done the
    intensity normalisation here and added it to the output prefix
    if it's present; i think that's the most consistent way of doing
    things.
    '''
    try:
        # get the corresponding function passed as a str via getattr
        fn = getattr(sys.modules[__name__], spectrum_type)
        spectrum, output_prefix = fn(**kwargs)
    except NameError:
        print("invalid spectrum type passed to light.spectrum_setup().")
        raise

    if "intensity" in kwargs:
        if "region" in kwargs:
            lower, upper = kwargs["region"]
            muM_init = micromole_in_region(spectrum, lower, upper)
            spectrum[:, 1] *= kwargs["intensity"] / muM_init
            output_prefix += f"_{kwargs['intensity']}muE"
        else:
            # assume PAR over [400, 700] nm
            print("spectrum_setup: intensity but no region given. Assuming [400, 700]nm")
            lower, upper = [400.0, 700.0]
            muM_init = micromole_in_region(spectrum, lower, upper)
            spectrum[:, 1] *= kwargs["intensity"] / muM_init
            output_prefix += f"_{kwargs['intensity']}muE"
    return spectrum, output_prefix

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
    zipped = build(spectra_dicts)
    outdir = os.path.join(constants.output_dir, "input_spectra")
    os.makedirs(outdir, exist_ok=True)
    for spectrum, out_pref in zipped:
        np.savetxt(os.path.join(outdir,
                  f"{out_pref}_spectrum.dat"), spectrum)
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(spectrum[:, 0], spectrum[:, 1])
        ax.set_xlabel("wavelength (nm)")
        ax.set_xlim(constants.x_lim)
        ax.set_ylabel(r'intensity')
        ax.set_title(out_pref)
        fig.savefig(os.path.join(outdir,
                    f"{out_pref}_test_plot.pdf"))
        plt.close()

if __name__ == "__main__":
    '''
    examples of dicts - run `python light.py` to get spectra/plots
    '''
    sd = [
          {'type': 'filtered', 'kwargs': {'filter': "red", 'fraction': 0.5}},
          {'type': 'phoenix', 'kwargs': {'Tstar': 4800}},
          {'type': 'stellar', 'kwargs': {'Tstar': 5770, 'Rstar': 6.957E8, 'a': 1.0, 'attenuation': 0.0}},
          {'type': 'am15', 'kwargs': {'dataset': "tilt", "intensity": 50.0, "region": [400.0, 700.0]}},
          {'type': 'marine', 'kwargs': {'depth': 10.0}},
          {'type': 'colour', 'kwargs': {'colour': 'red', 'intensity': 30.0}},
          {'type': 'gaussian', 'kwargs':
           {'mu': [600.0, 500.0],
               'sigma': [15.0, 35.0], 'a': [1.0, 0.2], 'intensity': 100.0}},
          ]

    check(sd)
    # if you only need one you can call spectrum_setup directly like so:
    spectrum, output_prefix = spectrum_setup(colour,
            colour="blue", intensity=50.0)

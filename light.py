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

# (Tstar, Rstar) - 4Gyr isochrones from Isabelle Baraffe
phz_stars = [
 # (2343, 68874300),
 (3131, 121051800),
 # (3416, 206622900),
 (3680, 323500500),
 # (5697, 691525800),
 (6053, 1037288700)
 ]
# semi major axes (orbital radii) of planets from Cass's PHZ paper
phz_radii = [
    # [
    # 1.706046696732635079e-02,
    # 1.771525953042107915e-02,
    # 1.839518348654892513e-02,
    # 1.910120339601702188e-02,
    # 1.983432083962659992e-02,
    # 2.059557583954514659e-02,
    # 2.138604833471279346e-02,
    # 2.220685971287574001e-02,
    # 2.305917440142015579e-02,
    # 2.394420151926349161e-02,
    # 2.486319659214645747e-02,
    # 2.581746333375905048e-02,
    # 2.680835549522756975e-02,
    # 2.783727878558613436e-02,
    # 2.890569286595717768e-02,
    # 3.001511342027015500e-02,
    # 3.116711430545564265e-02,
    # 3.236332978416562450e-02,
    # 3.360545684318684556e-02,
    # 3.489525760083680983e-02,
    # ],
    [
    5.334283503168275181e-02,
    5.529150259545827228e-02,
    5.731135695070926211e-02,
    5.940499862272529547e-02,
    6.157512313660758879e-02,
    6.382452448770795095e-02,
    6.615609873884630010e-02,
    6.857284774893830648e-02,
    7.107788303783336981e-02,
    7.367442979233920775e-02,
    7.636583101859042988e-02,
    7.915555184610743866e-02,
    8.204718398908683596e-02,
    8.504445037066729607e-02,
    8.815120991612462875e-02,
    9.137146252116694345e-02,
    9.470935420172675590e-02,
    9.816918243188020587e-02,
    1.017554016767655556e-01,
    1.054726291276254502e-01,
    ],
    # [
    # 1.073470624067316831e-01,
    # 1.111884324366361476e-01,
    # 1.151672642971283550e-01,
    # 1.192884770027061586e-01,
    # 1.235571655927574924e-01,
    # 1.279786074305380439e-01,
    # 1.325582687275550231e-01,
    # 1.373018113014235486e-01,
    # 1.422150995755497305e-01,
    # 1.473042078292948753e-01,
    # 1.525754277075836718e-01,
    # 1.580352759992404432e-01,
    # 1.636905026936701923e-01,
    # 1.695480993258443592e-01,
    # 1.756153076199086505e-01,
    # 1.818996284420988097e-01,
    # 1.884088310740324179e-01,
    # 1.951509628178418332e-01,
    # 2.021343589450231293e-01,
    # 2.093676530012997250e-01,
    # ],
    [
    1.946624832560100438e-01,
    2.014896596200921042e-01,
    2.085562777930252687e-01,
    2.158707354456428917e-01,
    2.234417247707572163e-01,
    2.312782428125948198e-01,
    2.393896021585042444e-01,
    2.477854420056417317e-01,
    2.564757396157864022e-01,
    2.654708221718972694e-01,
    2.747813790505013176e-01,
    2.844184745244980328e-01,
    2.943935609114737839e-01,
    3.047184921831532778e-01,
    3.154055380521589846e-01,
    3.264673985528201428e-01,
    3.379172191333574271e-01,
    3.497686062773789906e-01,
    3.620356436732513061e-01,
    3.747329089505601063e-01,
    ],
    # [
    # 9.227482934803379333e-01,
    # 9.508250916731063596e-01,
    # 9.797561928240350326e-01,
    # 1.009567591120185570e+00,
    # 1.040286071683211189e+00,
    # 1.071939234635411076e+00,
    # 1.104555519898051230e+00,
    # 1.138164232744234283e+00,
    # 1.172795570129273601e+00,
    # 1.208480647822234078e+00,
    # 1.245251528363010785e+00,
    # 1.283141249870070499e+00,
    # 1.322183855724736734e+00,
    # 1.362414425158687159e+00,
    # 1.403869104772150100e+00,
    # 1.446585141011116127e+00,
    # 1.490600913632745161e+00,
    # 1.535955970189037512e+00,
    # 1.582691061559753143e+00,
    # 1.630848178566503215e+00,
    # ],
    [
    1.533622250641897722e+00,
    1.579549580360156913e+00,
    1.626852294149798261e+00,
    1.675571580587545073e+00,
    1.725749861722928769e+00,
    1.777430830017054264e+00,
    1.830659486387568702e+00,
    1.885482179392962143e+00,
    1.941946645590318266e+00,
    2.000102051101658507e+00,
    2.059999034425071596e+00,
    2.121689750527903762e+00,
    2.185227916260411796e+00,
    2.250668857129409783e+00,
    2.318069555472653143e+00,
    2.387488700075889980e+00,
    2.458986737275799861e+00,
    2.532625923593306183e+00,
    2.608470379943099715e+00,
    2.686586147466567098e+00,
    ]
]

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
    - Rstar:       radius of star
    - a:           orbital distance (semi-major axis) in AU
    - attenuation: attenuation factor (should be in [0, 1])
    '''
    try:
        T = kwargs["Tstar"]
        R = kwargs["Rstar"]
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
    TO DO: figure out micro einstein normalisation
    and how it should format its output files - should probably
    also make a plot of the incident spectra to go with output
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
    outdir = os.path.join(constants.output_dir)
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

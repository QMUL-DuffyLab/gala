import os
import numpy as np
import pandas as pd

basedir = os.path.join("out", "PHZ")
rcs = ["anox", "ox", "exo"]

shift_inc = 10.0 # shift increment in nm

pparams = { # averaged pigment parameters
  "n_gauss": 2,
  "amp": [1.00e+00, 2.00e-01],
  "mu": [6.60e+02, 6.10e+02],
  "sigma": [2.00e+01, 3.0e+01],
}

xlim = [500.0, 1500.0]
x = np.arange(*xlim)

# (Tstar, Lstar, Rstar) for each star used
# from Baraffe - see docstring for calculate_phz_radii below
phz_stars = [
 (2343,  -3.58,  0.099),
 (2811,  -3.06,  0.124),
 (3131,  -2.58,  0.174),
 (3416,  -1.97,  0.297),
 (3680,  -1.45,  0.465),
 (4418,  -0.85,  0.646),
 (5297,  -0.29,  0.850),
 (5697,  -0.03,  0.994),
 (6053,   0.43,  1.491)
 ]

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-1.0 * (x - mu)**2/(2.0 * sigma**2))

def absorption(x, n_pigments, shifts):
    y = np.zeros_like(x)
    for npi, si in zip(n_pigments, shifts):
        for kk in range(pparams['n_gauss']):
            y += npi * gauss(x, pparams['amp'][kk],
                             pparams['mu'][kk] + si, pparams['sigma'][kk])
    return y / np.sum(y)

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

# you shouldn't actually need this but here it is just in case
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

def get_arrays(df):
    '''
    take a combined population dataframe
    and return the n_p and shift arrays
    '''
    n_p = [np.fromstring(string[1:-1], sep=' ')
           for string in df['n_p'].to_numpy()]
    shift = [shift_inc * np.fromstring(string[1:-1], sep=' ')
           for string in df['shift'].to_numpy()]
    return n_p, shift

# right. here's the actual loop to generate stuff

# get the set of radii and the folder names for each star type
filedict = {}
for Ts, Ls, Rs in phz_stars:
    folders = []
    radii = calculate_phz_radii(Ts, Ls)
    for a in radii:
        folders.append(f"stellar_{Ts:4d}K_a_{a:6.4f}AU")
    filedict[Ts] = {
            'radii': radii,
            'folders': folders
            }

# now loop over that set of folders for each RC type, and
# for each folder, read in the converged populations, combine them all,
# and construct the total absorption of the combined population.
# save this absorption (peak normalised to 1) to a separate file so we
# don't have to do this every time
total_abs = np.zeros_like(x)
for rc in rcs:
    for Tstar, subdict in filedict.items():
        outpath = os.path.join(basedir, "absorptions", rc, f"{Tstar:4d}")
        outpaths.append(outpath)
        os.makedirs(outpath, exist_ok=True)
        for folder, radius in zip(subdict['folders'], subdict['radii']):
            path = os.path.join(basedir, rc, "cost_0.01", folder)
            if os.path.isdir(path):
                total_abs = 0.0
                files = [os.path.join(path, f"{ii:1d}_final_population.csv")
                      for ii in range(3)]
                try:
                    dfs = [pd.read_csv(ff) for ff in files]
                    combined_pop = pd.concat(dfs)
                except FileNotFoundError:
                    print(f"Files not found for {path}")
                    raise
                n_p, shift = get_arrays(combined_pop)
                for ii, (n_pi, shifti) in enumerate(zip(n_p, shift)):
                    total_abs += absorption(x, n_pi, shifti)
                total_abs /= np.max(total_abs)
                outfile = os.path.join(outpath,
                                       f"{radius:6.4f}AU_absorption.txt")
                print(outfile)
                np.savetxt(outfile, np.column_stack((x, total_abs)))

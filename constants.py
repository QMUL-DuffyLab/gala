# -*- coding: utf-8 -*-
"""
13/11/2023
@author: callum
"""
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
'''
General stuff
'''
spectrum_prefix = 'PHOENIX/Scaled_Spectrum_PHOENIX_'
spectrum_suffix = '.dat'
T=300.0 # temperature (Kelvin)
n_b_bounds = np.array([1, 5]) # max neighbours for identical spheres
n_s_bounds = np.array([1, 10])
n_p_bounds = np.array([1, 100])
n_individuals = 20
fitness_cutoff = 0.2 # fraction of individuals kept
mutation_width = 0.1 # width of Gaussian/Poisson we draw from for mutation

'''
some rates that I think are going to be fixed
all given in s^{-1}
'''

k_diss=1.0/4.0E-9 #Chl excited state decay rate 
k_trap=1.0/5.0E-12 #PSII trapping rate
k_con=1.0/10.0E-3 #PSII RC turnover rate
k_hop=1.0/10.0E-12 # assume all inter-subunit transfer is around the same
k_lhc_rc=1.0/10.0E-12

k_params  = (k_diss, k_trap, k_con, k_hop, k_lhc_rc)

'''
Spectral parameters - I think these will change
'''
sig_chl = 1E-20 # (approximate!) cross-section of one chlorophyll
sig=1.9E-18 #optical cross-section of the antenna (m^2)
b12=0.56 #ratio of Chlb/Chb peak amplitude 
lp1=650.0#Chlb 650 nm absorption peak (nm)
w1=8.5 #Chlb peak width (nm)
lp2=675.0#Chlb 675 nm absorption peak
w2=9.0 #Chlb width (nm)
lp_rc=680.0 #reaction centre
w_rc=w2 #width is the same as Chla

# check these parameters with chris!
rc_params = (1, sig, lp_rc, w_rc)

lambda_bounds = np.array([200.0, 1400.00])
width_bounds  = np.array([1.0, 500.0])
sigma_bounds  = np.array([sig_chl, sig_chl])

radiative_subunit = [1, sig_chl, 680.0, 10.0]

'''
# dataclass reduces memory requirement, makes code more readable
# dataclass with slots requires python 3.10 or newer
@dataclass(slots=True)
class subunit:
    n_pigments:  int
    sigma:       float
    lambda_peak: float
    width:       float

# NB: the names here must exactly match subunit above if you add anything
@dataclass(slots=True)
class subunit_bounds:
    n_pigments:  npt.NDArray[np.int]
    sigma:       npt.NDArray[np.float64]
    lambda_peak: npt.NDArray[np.float64]
    width:       npt.NDArray[np.float64]

bounds = subunit_bounds(n_p_bounds, sigma_bounds, lambda_bounds, width_bounds)
'''

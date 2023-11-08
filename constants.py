'''
General stuff
'''
spectrum_prefix = 'PHOENIX/Scaled_Spectrum_PHOENIX_'
spectrum_suffix = '.dat'
T=300.0 # temperature (Kelvin)
max_size = 10000 # maximum for n_branches, n_subunits, n_pigments
n_individuals = 100

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

lambda_min = 200.0
lambda_max = 1400.00

width_min = 1.0
width_max = 500.0

radiative_subunit = (1, sig_chl, 680.0, 10.0)

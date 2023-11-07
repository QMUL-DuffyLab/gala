'''
General stuff
'''
spectrum_prefix = 'PHOENIX/Scaled_Spectrum_PHOENIX_'
spectrum_suffix = '.dat'
T=300.0 # temperature (Kelvin)

'''
some rates that I think are going to be fixed
all given in s^{-1}
'''

k_diss=1.0/4.0E-9 #Chl excited state decay rate 
k_trap=1.0/5.0E-12 #PSII trapping rate
k_con=1.0/10.0E-3 #PSII RC turnover rate
k_12=1.0/1.0E-12 #fundamental Chlb-Chla hopping rate
k_2rc=1.0/5.0E-12 #fundanental LHCII-RC hopping rate
k_hop=1.0/5.0E-12
k_lhc_rc=1.0/16.0E-12

'''
Spectral parameters - I think these will change
'''
sig=1.9E-18 #optical cross-section of the antenna (m^2)
b12=0.56 #ratio of Chlb/Chb peak amplitude 
lp1=650.0#Chlb 650 nm absorption peak (nm)
w1=8.5 #Chlb peak width (nm)
lp2=675.0#Chlb 675 nm absorption peak
w2=9.0 #Chlb width (nm)
lp_rc=680.0 #reaction centre
w_rc=w2 #width is the same as Chla

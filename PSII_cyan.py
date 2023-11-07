# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 08:36:44 2023

@author: btw430
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.constants import h as h
from scipy.constants import c as c
from scipy.constants import Boltzmann as kB
import Lattice_antenna as lattice
import PSII_params as PSII

Ts=['2300','2600','2800','3300','3800','4300','4800','5800']
filename='Scaled_Spectrum_PHOENIX_'
colours=['maroon','red','darkorange','gold','darkgreen','darkcyan','darkblue','fuchsia']

'''************************************************************************'''
''' Input paramaters                                                       '''
'''************************************************************************'''

#RC params
N_RC=1
sig_RC=PSII.sig
lp_RC=PSII.lpRC
w_RC=PSII.wRC
RC_params = (N_RC, sig_RC, lp_RC, w_RC)

#antenna params
Nb=5 #5 branches
s1=(PSII.N1, PSII.sig*PSII.B12, PSII.lp1, PSII.w1)
s2=(PSII.N2, PSII.sig, PSII.lp2, PSII.w2)
Branch_params=(Nb,s2,s1)

#Rate parameters
K_hop=1.0/1.0E-12
K_LHC_RC=1.0/30.0E-12
k_params=(PSII.k_diss,PSII.k_trap,PSII.k_con,K_hop,K_LHC_RC)

#temperature
T=300.0

'''************************************************************************'''
'''************************************************************************'''


fin=open(filename+Ts[-1]+'K.txt','r')
l, Ip_y=[],[]
for line in fin:
    line=line.rstrip()
    elements=line.split()
    l.append(float(elements[0]))
    Ip_y.append(float(elements[1]))    
fin.close()
l=np.array(l) #convert to numpy array to pass to functions in Thermo_antenna
Ip_y=np.array(Ip_y)

cyano=lattice.Antenna_branched_funnel(l,Ip_y,Branch_params,RC_params,k_params,T)

TW_Adj_mat=cyano['TW_Adj_mat']
K_b=cyano['K_b']
K_mat=cyano['K_mat']
gamma_vec=cyano['gamma_vec']
N_eq=cyano['N_eq']
gamma_b=cyano['gamma_b']
nu_e=cyano['nu_e']
phi_F=cyano['phi_F']

print(gamma_b)
print(N_eq)
print(nu_e)
print(phi_F)

fout=open('cyano_output.txt','w')
for line in TW_Adj_mat:
    for element in line:
        fout.write(str(element)+'\t')
    fout.write('\n')

fout.write('\n')
for line in K_mat:
    for element in line:
        fout.write(str(element)+'\t')
    fout.write('\n')

fout.write('\n')
for element in gamma_vec:
    fout.write(str(element)+'\n')
    

fout.close()
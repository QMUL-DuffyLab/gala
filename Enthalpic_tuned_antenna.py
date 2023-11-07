# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:53:31 2023

@author: btw430
"""

import numpy as np
import scipy as sp
import seaborn as sns
from scipy.constants import h as h
from scipy.constants import c as c
from scipy.constants import Boltzmann as kB
import Lattice_antenna as lattice
import PSII_params as PSII

import matplotlib.pyplot as plt

import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

#Ts=['2300','2600','2800','3300','3800','4300','4800','5800']
Ts='2300'
filename='Scaled_Spectrum_PHOENIX_'


'''************************************************************************'''
''' General Input Paramaters                                               '''
'''************************************************************************'''

#RC params
N_RC=1
sig_RC=1.6E-18
lp_RC=PSII.lpRC
w_RC=PSII.wRC
RC_params = (N_RC, sig_RC, lp_RC, w_RC)
   
#Fixed antenna parameters
sig=1.6E-18
w=PSII.w2
N_pigment=PSII.N2 #assume 24 pigments per
lp_0=PSII.lp2 #unshifted antenna wave-length

#Rate parameters
k_diss=1.0/4.0E-9
k_trap=1.0/5.0E-12 #PSII trapping rate
k_con=1.0/10.0E-3 #PSII RC turnover rate
K_hop=1.0/5.0E-12
K_LHC_RC=1.0/16.0E-12
k_params=(k_diss,k_trap,k_con,K_hop,K_LHC_RC)

T=300.0 #temperature

Nb=6 #consider a 6-branch antenna

#Branched antennae
#number of subunits of a particular peak wavelength
sub_unit=8
sub_shift=3.0
sub_peaks=[675.0,650.0,615.0,582.0]
peak_colours=['red','darkorange','darkgreen','darkblue','indigo']
       
    
#hex antennae
layer_peaks=[675.0,660.0,653.0,647.0,635.0,614.0,609.0]

layer_colours=['red','darkorange','yellow',
              'darkgreen','darkblue','darkmagenta','magenta']


'''************************************************************************'''
''' Input                                                                  '''
'''************************************************************************'''

fin=open(filename+Ts+'K.txt','r')
l, Ip_y=[],[]
for line in fin:
    line=line.rstrip()
    elements=line.split()
    l.append(float(elements[0]))
    Ip_y.append(float(elements[1]))    
fin.close()
l=np.array(l)
Ip_y=np.array(Ip_y)
Ip_y_plot=Ip_y*50

fin=open('Nb6_branched_nu_e.txt','r')
n_branched, nu_branched=[], []
for line in fin:
    line=line.rstrip()
    elements=line.split()
    n_branched.append(float(elements[0]))
    nu_branched.append(float(elements[1]))
fin.close()

fin=open('Hex_branched_nu_e.txt','r')
n_hex, nu_hex=[], []
for line in fin:
    line=line.rstrip()
    elements=line.split()
    n_hex.append(float(elements[0]))
    nu_hex.append(float(elements[1]))
fin.close()

#optimal branched
plt.plot(l,Ip_y_plot,label='$T_{s} = 2300$ K',color='maroon',
         linewidth=4.0,alpha=0.5)    

branch_structure=[]
for l_num, lp in enumerate(sub_peaks):
    for i in range(sub_unit):
        lp_sub=lp-(float(i*sub_shift))
        branch_structure.append(lp_sub)
        if i==0:
            plt.vlines(lp_sub,-1.0,6.0,linestyle='-',color=peak_colours[l_num],
                       linewidth=2,alpha=0.6,label=str(int(lp))+' nm band')    
        else:
            plt.vlines(lp_sub,-1.0,6.0,linestyle='-',color=peak_colours[l_num],
                       linewidth=2,alpha=0.6)    

plt.xlabel(r'$\lambda$ (nm)',fontsize=14)
plt.ylabel(r'$N_{LHC}$',fontsize=14)
plt.xlim(550.0,800.0)
plt.ylim(-0.05,50.5)
plt.legend(framealpha=1.0,loc='upper right',fontsize=12)
plt.tick_params('x',labelsize=12)
plt.tick_params('y',labelsize=12)
plt.savefig('Enthalpic_tuned_branched_Ts2300K.pdf')
plt.show()

branch=[]
for sub_branch in branch_structure:
    lp=sub_branch
    s_i=(N_pigment,sig,lp,w)
    branch.append(s_i)

Branch_params=[Nb]
for s in branch:
    Branch_params.append(s)
    
cyano=lattice.Antenna_branched_funnel(l, Ip_y, Branch_params, RC_params, k_params, T)
nu_e_branch=cyano['nu_e']
phi_F_branch=cyano['phi_F']
N_LHC_branch=int(sub_unit*len(sub_peaks))
nu_per_prot_branch=nu_e_branch/N_LHC_branch
print('nu_e_opt (branched) = ',nu_e_branch,' s^-1')
print('phi_F (branched) = ',phi_F_branch)
print('N_LHC (branched) = ',N_LHC_branch)
print('nu_e per LHC (branched) = ',nu_per_prot_branch,'\n')


#optimal hex
N_layer_hex=len(layer_peaks)
Hex_list=lattice.Hex_n(N_layer_hex+1) #list of hexagonal numbers
N_LHC_hex=Hex_list[-1]-1

layer=[]
for s in range(N_layer_hex):
    layer.append((N_pigment,sig,layer_peaks[s],w))

G_params=(int(N_LHC_hex),layer)
k_params=(k_diss,K_hop,k_trap,k_con,K_LHC_RC) 
out_hex=lattice.Antenna_hex_funnel(l,Ip_y,G_params,RC_params,k_params,T)   

nu_e_hex=out_hex['nu_e']
phi_F_hex=out_hex['phi_F']
nu_per_prot_hex=nu_e_hex/N_LHC_hex
print('nu_e_opt (hex) = ',nu_e_hex,' s^-1')
print('phi_F (hex) = ',phi_F_hex)
print('N_LHC (hex) = ',N_LHC_hex)
print('nu_e per LHC (hex) = ',nu_per_prot_hex,'\n')


plt.plot(l,Ip_y_plot,label='$T_{s} = 2300$ K',color='maroon',
         linewidth=4.0,alpha=0.5)    

for l_num, lp in enumerate(layer_peaks):
    height=(6*(l_num+1))
    plt.vlines(lp,-1.0,height,linestyle='-',linewidth=2.0,
               color=layer_colours[l_num],alpha=0.6,label=str(int(lp))+' nm layer')    
    
plt.xlabel(r'$\lambda$ (nm)',fontsize=14)
plt.ylabel(r'$N_{LHC}$',fontsize=14)
plt.xlim(550.0,800.0)
plt.ylim(-0.05,50.5)
plt.legend(framealpha=1.0,loc='upper right',fontsize=12)
plt.tick_params('x',labelsize=12)
plt.tick_params('y',labelsize=12)
plt.savefig('Enthalpic_tuned_hex_Ts2300K.pdf')
plt.show()


#Comparison with the acclimation calculations
plt.plot(n_branched,nu_branched,linestyle='-',
         label='$N_{b} = 6, \Delta\lambda_{p} = $'+str(-3.0),color='darkgreen',alpha=0.4)
plt.scatter(n_branched,nu_branched,marker='o',s=40,color='darkgreen',alpha=0.3)  

plt.plot(n_hex,nu_hex,linestyle='-',
         label=r'$hex, \Delta\lambda_{p} = $'+str(-3.0),color='darkblue',alpha=0.4)
plt.scatter(n_hex,nu_hex,marker='x',s=40,color='darkblue',alpha=0.3)  

plt.scatter(N_LHC_branch,nu_e_branch,marker='X',color='darkgreen',
            label='branched tuned',s=200)
plt.scatter(N_LHC_hex,nu_e_hex,marker='X',color='darkblue',
            label='hex tuned',s=200)

plt.ylim(-0.5,85)
plt.xlabel(r'$N_{LHC}$',fontsize=14)
plt.ylabel(r'$\nu_{e} (s^{-1})$',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.tick_params('y',labelsize=12)
plt.legend(loc='lower right',fontsize=12)
plt.savefig('Enthalpic_tuned_comparison.pdf')
plt.show()




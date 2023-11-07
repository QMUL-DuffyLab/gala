# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:47:34 2023

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
k_params_branched=(k_diss,k_trap,k_con,K_hop,K_LHC_RC)
k_params_hex=(k_diss,K_hop,k_trap,k_con,K_LHC_RC)

T=300.0 #temperature

N_LHC_max=200

'''************************************************************************'''
''' Branched Input Paramaters                                              '''
'''************************************************************************'''
#Params_branched[(Nb,dl_max)] #
#Params_branched=[(1,2.0),(6,3.0),(12,5.0),(50,10.0),(100,12.0)]
Params_branched=[(1,0.0),(6,0.0),(1,2.0),(6,3.0)]
Ns_step=10
colours_branched=['maroon','darkgreen','maroon','darkgreen']
marker_branched=['^','o','^','o']

'''************************************************************************'''
''' Hex Input Paramaters                                              '''
'''************************************************************************'''

#Params_hex[dl_max]
Param_hex=[0.0,10.0]
N_LHC_step=20
line_hex=['--','-']

fin=open(filename+Ts+'K.txt','r')
l, Ip_y=[],[]
for line in fin:
    line=line.rstrip()
    elements=line.split()
    l.append(float(elements[0]))
    Ip_y.append(float(elements[1]))    
fin.close()

l=np.array(l) #convert to numpy array to pass to functions in Thermo_antenna
Ip_y=np.array(Ip_y)    

#(1) Branched antenna calculations
for n, b in enumerate(Params_branched):
    Nb, dl=b[0], b[1]

    nu_e_Ns, phi_F_Ns=[],[]        
    N_LHC=[] #N_LHC axis
    
    Ns_max=int(np.ceil(N_LHC_max/Nb))
    for Ns in range(1,Ns_max+1,Ns_step):
        
        #define branch parameters
        branch=[]
        for i in range(Ns+1):
            s_i=(N_pigment,sig,lp_0-(float(i+1)*dl),w)                
            branch.append(s_i)

        Branch_params=[Nb]
        for s in branch:
            Branch_params.append(s)
        
        cyano=lattice.Antenna_branched_funnel(l,Ip_y,Branch_params,RC_params,k_params_branched,T)
        nu_e_Ns.append(cyano['nu_e'])
        phi_F_Ns.append(cyano['phi_F'])
        N_LHC.append(Nb*Ns)

    if dl==0.0:
        plt.plot(N_LHC,nu_e_Ns,color=colours_branched[n],linestyle='-',
                 linewidth=5,alpha=0.2)

    else:
        plt.plot(N_LHC,nu_e_Ns,color=colours_branched[n],linestyle='-',
                 label='$N_{b} = $'+str(Nb)+', $\Delta\lambda_{p} = $'+str(-dl))        
        plt.scatter(N_LHC,nu_e_Ns,color=colours_branched[n],marker=marker_branched[n],s=40)        


#(2) Hex lattice
for n, dl  in enumerate(Param_hex):
    nu_e_LHC, phi_F_LHC=[], []
    N_LHC_axis=[]

    for N_LHC in range(1,N_LHC_max+1,N_LHC_step): #loop over number of LHCs
    
        N_LHC_axis.append(N_LHC)
    
        #Calculate how many layers make up the antenna
        Hex_list=lattice.Hex_n(N_LHC) #list of hexagonal numbers
        layer_cnt=0 #layer counter
        for hex_num in Hex_list:
            if N_LHC>=hex_num:
                layer_cnt=layer_cnt+1
            else:
                break
            
        layer=[]
        for s in range(layer_cnt):
            layer.append((N_pigment,sig,lp_0-(float(s)*dl),w))
    
        G_params=(N_LHC,layer)        
        out_hex=lattice.Antenna_hex_funnel(l,Ip_y,G_params,RC_params,k_params_hex,T)            
        nu_e_LHC.append(out_hex['nu_e'])    
        phi_F_LHC.append(out_hex['phi_F'])
    
    if dl==0.0:
        plt.plot(N_LHC_axis,nu_e_LHC,color='darkblue',linestyle='-',linewidth=5,
                 alpha=0.2)
    else:
        plt.plot(N_LHC_axis,nu_e_LHC,color='darkblue',linestyle='-',linewidth=1,
                 alpha=1.0,label='hex, $\Delta\lambda_{p} = $'+str(dl)+' nm')
        plt.scatter(N_LHC_axis,nu_e_LHC,color='darkblue',marker='x',s=40)

plt.xlim(-5.0,205.0)
plt.ylim(-2,82.0)

plt.xlabel(r'$N_{LHC}$',fontsize=14)
plt.ylabel(r'$\nu_{e} (s^{-1)}$',fontsize=14)
plt.title('$T_{s} $ = 2300 K',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.tick_params('y',labelsize=12)


order=[0,1,2]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in order],
           [labels[i] for i in order],
           fontsize=12)

plt.savefig('Enthalpic_optimum_nu_e_Ts2300K.pdf')
plt.show()


#efficiency (very hacked)

#(1) Branched antenna calculations
fout_Nb6=open('Nb6_branched_nu_e.txt','w')
for n, b in enumerate(Params_branched):
    Nb, dl=b[0], b[1]

    nu_e_Ns, phi_F_Ns=[],[]        
    N_LHC=[] #N_LHC axis
    
    Ns_max=int(np.ceil(N_LHC_max/Nb))
    for Ns in range(1,Ns_max+1,Ns_step):
        
        #define branch parameters
        branch=[]
        for i in range(Ns+1):
            s_i=(N_pigment,sig,lp_0-(float(i+1)*dl),w)                
            branch.append(s_i)

        Branch_params=[Nb]
        for s in branch:
            Branch_params.append(s)
        
        cyano=lattice.Antenna_branched_funnel(l,Ip_y,Branch_params,RC_params,k_params_branched,T)
        nu_e_Ns.append(cyano['nu_e'])
        phi_F_Ns.append(cyano['phi_F'])
        N_LHC.append(Nb*Ns)
        
        if Nb==6 and dl!=0.0:
            fout_Nb6.write(str(Nb*Ns)+'\t'+str(cyano['nu_e'])+'\n')

    if dl==0.0:
        plt.plot(N_LHC,phi_F_Ns,color=colours_branched[n],linestyle='-',
                 linewidth=5,alpha=0.2)

    else:
        plt.plot(N_LHC,phi_F_Ns,color=colours_branched[n],linestyle='-')        
        plt.scatter(N_LHC,phi_F_Ns,color=colours_branched[n],marker=marker_branched[n],s=40) 
        
fout_Nb6.close()        


#(2) Hex lattice
fout_hex=open('Hex_branched_nu_e.txt','w')
for n, dl  in enumerate(Param_hex):
    nu_e_LHC, phi_F_LHC=[], []
    N_LHC_axis=[]

    for N_LHC in range(1,N_LHC_max+1,N_LHC_step): #loop over number of LHCs
    
        N_LHC_axis.append(N_LHC)
    
        #Calculate how many layers make up the antenna
        Hex_list=lattice.Hex_n(N_LHC) #list of hexagonal numbers
        layer_cnt=0 #layer counter
        for hex_num in Hex_list:
            if N_LHC>=hex_num:
                layer_cnt=layer_cnt+1
            else:
                break
            
        layer=[]
        for s in range(layer_cnt):
            layer.append((N_pigment,sig,lp_0-(float(s)*dl),w))
    
        G_params=(N_LHC,layer)        
        out_hex=lattice.Antenna_hex_funnel(l,Ip_y,G_params,RC_params,k_params_hex,T)            
        nu_e_LHC.append(out_hex['nu_e'])    
        phi_F_LHC.append(out_hex['phi_F'])
        
        if dl!=0.0:
            fout_hex.write(str(N_LHC)+'\t'+str(out_hex['nu_e'])+'\n')
    
    if dl==0.0:
        plt.plot(N_LHC_axis,phi_F_LHC,color='darkblue',linestyle='-',linewidth=5,
                 alpha=0.2)
    else:
        plt.plot(N_LHC_axis,phi_F_LHC,color='darkblue',linestyle='-',linewidth=1,
                 alpha=1.0,label='hex, $\Delta\lambda_{p} = $'+str(dl)+' nm')
        plt.scatter(N_LHC_axis,phi_F_LHC,color='darkblue',marker='x',s=40)

fout_hex.close()

plt.xlim(-5.0,205.0)
plt.ylim(-0.05,1.05)

plt.xlabel(r'$N_{LHC}$',fontsize=14)
plt.ylabel(r'$\phi_{e}$',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.tick_params('y',labelsize=12)

'''
order=[0,1,2,4,3,5]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in order],
           [labels[i] for i in order],
           fontsize=11)
'''
plt.savefig('Enthalpic_optimum_phi_F_Ts2300K.pdf')
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:01:23 2023

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
Ts=['2300','2600','2800']
filename='PHOENIX/Scaled_Spectrum_PHOENIX_'
#colours=['maroon','red','darkorange','gold','darkgreen','darkcyan','darkblue','fuchsia']
colours=['maroon','red','darkorange']

'''************************************************************************'''
''' Input paramaters                                                       '''
'''************************************************************************'''

#RC params
N_RC=1
sigRC=1.6E-18
lpRC=PSII.lpRC
wRC=PSII.wRC


#Fixed antenna parameters
sig=1.6E-18
w=PSII.w2
N_pigment=PSII.N2 #assume 24 pigments per
lp_0=PSII.lp2 #unshifted antenna wave-length

T=300.0 #temperature

#Rate parameters
k_diss=1.0/4.0E-9
k_trap=1.0/5.0E-12 #PSII trapping rate
k_con=1.0/10.0E-3 #PSII RC turnover rate
K_hop=1.0/5.0E-12
K_LHC_RC=1.0/16.0E-12

#incremental parameters
shift_step=1.0 #The progressive blue shift in nm
shift_max=20.0

#maximum number of antenna proteins
N_LHC_max=61
N_LHC_step=1

Vmax=50.0

flog=open('Cyano_hex_log.txt','w')

nu_e_Ts, phi_F_Ts=[], [] #data points
for t, Temp in enumerate(Ts):
    fin=open(filename+Temp+'K.dat','r')
    l, Ip_y=[],[]
    for line in fin:
        line=line.rstrip()
        elements=line.split()
        l.append(float(elements[0]))
        Ip_y.append(float(elements[1]))    
    fin.close()

    l=np.array(l) #convert to numpy array to pass to functions in Thermo_antenna
    Ip_y=np.array(Ip_y)    

    #file output
    fout_nu=open('Enthalpic_nu_e_Ts'+str(Temp)+'hex.txt','w')
    fout_phi=open('Enthalpic_phiF_Ts'+str(Temp)+'hex.txt','w')

    fout_nu.write('\t')
    fout_phi.write('\t')

    dl_axis=0.0
    while dl_axis<=shift_max: #write column headings
        fout_nu.write(str(dl_axis)+'\t')
        fout_phi.write(str(dl_axis)+'\t')
        dl_axis=dl_axis+shift_step          
    fout_nu.write('\n')
    fout_phi.write('\n')

    nu_e_LHC, phi_F_LHC=[], []
    for N_LHC in range(1,N_LHC_max+1,N_LHC_step): #loop over number of LHCs

        fout_nu.write(str(N_LHC)+'\t')    
        fout_phi.write(str(N_LHC)+'\t')    

        dl=0.0
        nu_e_dl, phi_F_dl=[], [] #containers 
        while dl<=shift_max: #loop over progressive blue shift

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
            RC_params=(N_RC,sigRC,lpRC,wRC)
            k_params=(k_diss,K_hop,k_trap,k_con,K_LHC_RC)
            
            out_hex=lattice.Antenna_hex_funnel(l,Ip_y,G_params,RC_params,k_params,T)            
            Adj_mat=out_hex['Adj_mat']
            K_mat=out_hex['K_mat']
            tau_mat=out_hex['tau_mat']
            gamma_total=out_hex['gamma_total']
            gamma_vec=out_hex['gamma_vec']
            nu_e=out_hex['nu_e']
            phi_F=out_hex['phi_F']
            
            nu_e_dl.append(nu_e)
            phi_F_dl.append(phi_F)
            
            if dl==0.0 and N_LHC==6:
                flog.write(str(Temp)+' K\ndl = '+str(dl)+', N_LHC = '+str(N_LHC)+'\n\n')
                flog.write('nu_e = '+str(nu_e)+' electrons per second\n\n')
                flog.write('tau_mat\n\t')
                for i in range(N_LHC+2):
                    flog.write(str(i)+'\t')
                flog.write('\n')
                
                for i in range(N_LHC+2):
                    flog.write(str(i)+'\t')
                    for j in range(N_LHC+2):
                        flog.write(str(tau_mat[i][j])+'\t')
                    flog.write('\n')
                
                flog.write('\ngamma (photons per second)\n')
                for i in range(N_LHC+2):
                    if i==0:
                        flog.write('Trap\t'+str(-gamma_vec[i])+'\n')
                    elif i==1:
                        flog.write('RC\t'+str(-gamma_vec[i])+'\n')
                    else:
                        flog.write(str(i)+'\t'+str(-gamma_vec[i])+'\n')
                flog.write('\n')
                
            dl=dl+shift_step            
        
        nu_e_LHC.append(nu_e_dl)
        phi_F_LHC.append(phi_F_dl)
        
    nu_e_Ts.append(nu_e_LHC)
    phi_F_Ts.append(phi_F_LHC)
        
flog.close()

#Plotting and file output
fopt=open('Enthalpic_hex_optimum.txt','w')
fopt.write('Parameters that maximize nu_e\n')
fopt.write('N_LHC_max = '+str(N_LHC_max)+'\n')

dl=np.arange(0,shift_max,shift_step)
for t, Temp in enumerate(Ts):
    
    plt.imshow(np.flipud(nu_e_Ts[t]), cmap='rainbow',
               extent=(0,-shift_max,1,N_LHC),
               vmin=0.0,vmax=Vmax)        
            
    plt.ylabel(r'$N_{LHC}$',fontsize=12)
    plt.xlabel(r'$\Delta\lambda_{p}$ (nm)',fontsize=12)
    plt.title('$hex$',fontsize=12)

    cb=plt.colorbar().set_label(label=r'$\nu_{e}$ $(s^{-1})$',size=12)
    plt.savefig('Enthalpic_nu_e_Ts'+str(Temp)+'K_hex.pdf')
    plt.show()

    #plot efficiency
    plt.imshow(np.flipud(phi_F_Ts[t]), cmap='rainbow',
               extent=(0,-shift_max,1,N_LHC),
               vmin=0.0,vmax=1.0)
        
    plt.ylabel(r'$N_{LHC}$',fontsize=12)
    plt.xlabel(r'$\Delta\lambda_{p}$ (nm)',fontsize=12)
    plt.title('$hex$ = ',fontsize=12)

    cb=plt.colorbar().set_label(label=r'$\phi_{F}$ $(s^{-1})$',size=12)
    plt.savefig('Enthalpic_phiF_Ts'+str(Temp)+'K_hex.pdf')
    plt.show()


    #print out optimal paramaters
    nu_max=np.amax(nu_e_Ts[t])
    dl_max_index=np.where(nu_e_Ts[t]==nu_max)[1][0]
    print('(Ts, dl_max_index, dl_max, nu_e_max) = ('+str(Temp)+
               ', '+str(dl_max_index)+', '+str(dl[dl_max_index])+', '
               +str(nu_max)+')')

    fopt.write('(Ts, dl_max_index, dl_max, nu_e_max) = ('+str(Temp)
               +', '+str(dl_max_index)+', '+str(dl[dl_max_index])+', '
               +str(nu_max)+')\n')

fopt.close()        
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:49:39 2023

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
sig_RC=1.6E-18
lp_RC=PSII.lpRC
w_RC=PSII.wRC
RC_params = (N_RC, sig_RC, lp_RC, w_RC)

#Fixed antenna parameters
sig=1.6E-18
w=PSII.w2
N_pigment=PSII.N2 #assume 24 pigments per
lp_0=PSII.lp2 #unshifted antenna wave-length

#hex funnnel parameters

#Rate parameters
K_hop=1.0/5.0E-12
K_LHC_RC=1.0/16.0E-12
k_params=(PSII.k_diss,PSII.k_trap,PSII.k_con,K_hop,K_LHC_RC)

T=300.0 #temperature

#incremental parameters
shift_step=1.0 #The progressive blue shift in nm
shift_max=20.0

N_LHC=61 #number of LHCs
Nb_vals=[12] #number of branches

Ns_step=1

Vmax=50.0

flog=open('Cyano_branched_log.txt','w')

'''************************************************************************'''
'''************************************************************************'''

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


    nu_e_Nb, phi_F_Nb=[], []
    for Nb in Nb_vals:
        
        Ns_max=int(np.ceil(N_LHC/Nb))
        nu_e_Ns, phi_F_Ns=[], []
        
        #file output
        fout_nu=open('Enthalpic_nu_e_Ts'+str(Temp)+'K_Nb'+str(Nb)+'.txt','w')
        fout_phi=open('Enthalpic_phiF_Ts'+str(Temp)+'K_Nb'+str(Nb)+'.txt','w')
        
        fout_nu.write('\t')
        fout_phi.write('\t')

        dl_axis=0.0
        while dl_axis<=shift_max: #write column headings
            fout_nu.write(str(dl_axis)+'\t')
            fout_phi.write(str(dl_axis)+'\t')
            dl_axis=dl_axis+shift_step          
        fout_nu.write('\n')
        fout_phi.write('\n')

        for Ns in range(1,Ns_max+1,Ns_step):
        
            fout_nu.write(str(Ns*Nb)+'\t')    
            fout_phi.write(str(Ns*Nb)+'\t')    
        
            dl=0.0
            nu_e_dl, phi_F_dl=[], [] #containers 
            while dl<=shift_max: #loop over progressive blue shift

                #Define the branch paramaters
                branch=[]
                for i in range(1,Ns+1):
                    s_i=(N_pigment,sig,lp_0-(float(i+1)*dl),w)                
                    branch.append(s_i)
                
                Branch_params=[Nb]
                for s in branch:
                    Branch_params.append(s)

                cyano=lattice.Antenna_branched_funnel(l,Ip_y,Branch_params,RC_params,k_params,T)
                nu_e_dl.append(cyano['nu_e'])
                phi_F_dl.append(cyano['phi_F'])

                fout_nu.write(str(cyano['nu_e'])+'\t')
                fout_phi.write(str(cyano['phi_F'])+'\t')

                if Nb*Ns==6 and dl==0.0 and Nb==6:
    
                    flog.write(str(Temp)+' K\ndl = '+str(dl)+', N_LHC = '+str(Nb*Ns)+', Nb = '+str(Nb)+', Ns = '+str(Ns)+'\n\n')
                    flog.write('nu_e = '+str(cyano['nu_e'])+' electrons per second\n\n')
                    flog.write('tau_mat\n\t')
                    for i in range((Nb*Ns)+2):
                        flog.write(str(i)+'\t')
                    flog.write('\n')
                    
                    for i in range((Nb*Ns)+2):
                        flog.write(str(i)+'\t')
                        for j in range((Nb*Ns)+2):
                            if cyano['K_mat'][i][j]!=0.0:
                                tau_mat=(1.0/cyano['K_mat'][i][j])/1.0E-12
                            else: 
                                tau_mat=np.inf
                            flog.write(str(tau_mat)+'\t')
                        flog.write('\n')
                    
                    flog.write('\n')

                if Nb*Ns==6 and dl==0.0 and Nb==1:
                    fout_nu.write(str(cyano['nu_e'])+'\t')
                    fout_phi.write(str(cyano['phi_F'])+'\t')
    
                    flog.write(str(Temp)+' K\ndl = '+str(dl)+', N_LHC = '+str(Nb*Ns)+', Nb = '+str(Nb)+', Ns = '+str(Ns)+'\n\n')
                    flog.write('nu_e = '+str(cyano['nu_e'])+' electrons per second\n\n')
                    flog.write('tau_mat\n\t')
                    for i in range((Nb*Ns)+2):
                        flog.write(str(i)+'\t')
                    flog.write('\n')
                    
                    for i in range((Nb*Ns)+2):
                        flog.write(str(i)+'\t')
                        for j in range((Nb*Ns)+2):
                            if cyano['K_mat'][i][j]!=0.0:
                                tau_mat=(1.0/cyano['K_mat'][i][j])/1.0E-12
                            else: 
                                tau_mat=np.inf
                            flog.write(str(tau_mat)+'\t')
                        flog.write('\n')
                    
                    flog.write('\n')


                dl=dl+shift_step            
            
            nu_e_Ns.append(nu_e_dl) #capture the matrix of value for a given Ts and Nb
            phi_F_Ns.append(phi_F_dl)
            
            fout_nu.write('\n')
            fout_phi.write('\n')
            
        nu_e_Ns=np.array(nu_e_Ns)
        phi_F_Ns=np.array(phi_F_Ns)
        
        nu_e_Nb.append(nu_e_Ns)
        phi_F_Nb.append(phi_F_Ns)
        
    nu_e_Ts.append(nu_e_Nb)
    phi_F_Ts.append(phi_F_Nb)
    
    
#Plotting and file output
fopt=open('Enthalpic_branched_optimum.txt','w')
fopt.write('Parameters that maximize nu_e\n')
fopt.write('Nb = '+str(Nb)+' N_LHC_max = '+str(N_LHC)+'\n')

dl=np.arange(0,shift_max,shift_step)
for t, Temp in enumerate(Ts):
    
    for b, Nb in enumerate(Nb_vals):
       
        plt.imshow(np.flipud(nu_e_Ts[t][b]), cmap='rainbow',
                   extent=(0,-shift_max,1,N_LHC),
                   vmin=0.0,vmax=Vmax)
            
        plt.ylabel(r'$N_{LHC}$',fontsize=12)
        plt.xlabel(r'$\Delta\lambda_{p}$ (nm)',fontsize=12)
        plt.title('$N_{b} $ = '+str(Nb),fontsize=12)
    
        cb=plt.colorbar().set_label(label=r'$\nu_{e}$ $(s^{-1})$',size=12)
        plt.savefig('Enthalpic_nu_e_Ts'+str(Temp)+'K_Nb'+str(Nb)+'.pdf')
        plt.show()
        
        #plot efficiency
        plt.imshow(np.flipud(phi_F_Ts[t][b]), cmap='rainbow',
                   extent=(0,-shift_max,1,N_LHC),
                   vmin=0.0,vmax=1.0)
            
        plt.ylabel(r'$N_{LHC}$',fontsize=12)
        plt.xlabel(r'$\Delta\lambda_{p}$ (nm)',fontsize=12)
        plt.title('$N_{b} $ = '+str(Nb),fontsize=12)
    
        cb=plt.colorbar().set_label(label=r'$\phi_{e}$',size=12)
        plt.savefig('Enthalpic_phiF_Ts'+str(Temp)+'K_Nb'+str(Nb)+'.pdf')
        plt.show()

        #print out optimal paramaters
        nu_max=np.amax(nu_e_Ts[t][b])
        dl_max_index=np.where(nu_e_Ts[t][b]==nu_max)[1][0]
        print('(Ts, dl_max_index, dl_max, nu_e_max) = ('+str(Temp)
                   +', '+str(dl_max_index)+', '+str(dl[dl_max_index])
                   +', '+', '+str(nu_max)+')')        
        
        fopt.write('(Ts, dl_max_index, dl_max, nu_e_max) = ('+str(Temp)
                   +', '+str(dl_max_index)+', '+str(dl[dl_max_index])
                   +', '+str(dl[dl_max_index])+', '+str(nu_max)+')\n')

fopt.close()
flog.close()


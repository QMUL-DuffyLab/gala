# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:10:17 2023

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

filename='Scaled_Spectrum_PHOENIX_'
#Ts=['2300','2600','2800','3300','3800','4300','4800','5800']
#colours=['maroon','red','darkorange','gold','darkgreen','darkcyan','darkblue','fuchsia']

Ts=['2300','3300','3800']
colours=['maroon','gold','darkgreen']


N1=18.0
N2=24.0
N_RC=1

k_diss=1.0/4.0E-9
k_hop=1.0/5.0E-12
k_trap=1.0/5.0E-12
k_con=1.0/10.0E-3
K_12=1.0/1.0E-12
K_LHC_RC=1.0/16.0E-12

T=300.0

G_params=PSII.G_params 
G_params[0]=1.6E-18 #retune the absorption rate

k_params = (k_diss,k_hop,k_trap,k_con,K_12,K_LHC_RC) 

N_LHC_max=55
N_LHC_step=4

flog=open('PSII_branched_log.txt','w')

phi_F_N1, nu_e_N1=[],[]
phi_F_N6, nu_e_N6=[],[]
phi_F_N50, nu_e_N50=[],[]
phi_F_hex, nu_e_hex=[],[]

for t, Temp in enumerate(Ts):      

    fin=open(filename+Temp+'K.txt','r')
    l, Ip_y=[],[]
    for line in fin:
        line=line.rstrip()
        elements=line.split()
        l.append(float(elements[0]))
        Ip_y.append(float(elements[1]))    
    fin.close()
    l=np.array(l) #convert to numpy array to pass to functions in Thermo_antenna
    Ip_y=np.array(Ip_y)

    #loop over increasing antenna size
    N=[]
    phi_T_N1, nu_T_N1=[],[]    
    phi_T_N6, nu_T_N6=[],[]    
    phi_T_N50, nu_T_N50=[],[]    
    phi_T_hex, nu_T_hex=[],[]    

    for N_LHC in range(1,N_LHC_max,N_LHC_step):
        
        #single branch calculation
        N_branch=1
        Size_params = (N1, N2, N_RC, N_LHC, N_branch) 
        out_1branch=lattice.Antenna_branched(l,Ip_y,Size_params,G_params,k_params,T)  

        if out_1branch['nu_e']<=100.0:
            nu_T_N1.append(out_1branch['nu_e'])
        else:
            nu_T_N1.append(100.0)
            
        phi_T_N1.append(out_1branch['phi_F'])

        #6 branch calculation
        N_branch=6
        Size_params = (N1, N2, N_RC, N_LHC, N_branch) 
        out_6branch=lattice.Antenna_branched(l,Ip_y,Size_params,G_params,k_params,T)  

        if out_6branch['nu_e']<=100.0:
            nu_T_N6.append(out_6branch['nu_e'])
        else:
            nu_T_N6.append(100.0)

        phi_T_N6.append(out_6branch['phi_F'])

        #50 branch calculation
        N_branch=50
        Size_params = (N1, N2, N_RC, N_LHC, N_branch) 
        out_50branch=lattice.Antenna_branched(l,Ip_y,Size_params,G_params,k_params,T)  

        if out_50branch['nu_e']<=100.0:
            nu_T_N50.append(out_50branch['nu_e'])
        else:
            nu_T_N50.append(100.0)

        phi_T_N50.append(out_50branch['phi_F'])
        
        #hex antenna
        Size_params = (N1, N2, N_RC, N_LHC) 
        out_hex=lattice.Antenna_hex(l,Ip_y,Size_params,G_params,k_params,T)  

        if out_hex['nu_e']<=100.0:
            nu_T_hex.append(out_hex['nu_e'])
        else:
            nu_T_hex.append(100.0)

        phi_T_hex.append(out_hex['phi_F'])
        
        if N_LHC==18:
            flog.write(str(Temp)+' K\n'+'N_LHC = '+str(N_LHC)+', 1 branch\n\n')
            flog.write('nu_e = '+str(out_1branch['nu_e'])+' electrons per second\n\n')
            flog.write('tau_mat\n\t')
            for i in range(N_LHC+2):
                flog.write(str(i)+'\t')
            flog.write('\n')
            
            for i in range(N_LHC+2):
                flog.write(str(i)+'\t')
                for j in range(N_LHC+2):
                    if out_1branch['K_mat'][i][j]!=0.0:
                        tau_mat=(1.0/out_1branch['K_mat'][i][j])/1.0E-12
                    else: 
                        tau_mat=np.inf
                    flog.write(str(tau_mat)+'\t')
                flog.write('\n')
            flog.write('\n')

            flog.write(str(Temp)+' K\n'+'N_LHC = '+str(N_LHC)+', 6 branch\n\n')
            flog.write('nu_e = '+str(out_6branch['nu_e'])+' electrons per second\n\n')
            flog.write('tau_mat\n\t')
            for i in range(N_LHC+2):
                flog.write(str(i)+'\t')
            flog.write('\n')
            
            for i in range(N_LHC+2):
                flog.write(str(i)+'\t')
                for j in range(N_LHC+2):
                    if out_6branch['K_mat'][i][j]!=0.0:
                        tau_mat=(1.0/out_6branch['K_mat'][i][j])/1.0E-12
                    else: 
                        tau_mat=np.inf
                    flog.write(str(tau_mat)+'\t')
                flog.write('\n')
            flog.write('\n')
            
            flog.write(str(Temp)+' K\n'+'N_LHC = '+str(N_LHC)+', hex\n\n')
            flog.write('nu_e = '+str(out_hex['nu_e'])+' electrons per second\n\n')
            flog.write('tau_mat\n\t')
            for i in range(N_LHC+2):
                flog.write(str(i)+'\t')
            flog.write('\n')
            
            for i in range(N_LHC+2):
                flog.write(str(i)+'\t')
                for j in range(N_LHC+2):
                    if out_hex['K_mat'][i][j]!=0.0:
                        tau_mat=(1.0/out_hex['K_mat'][i][j])/1.0E-12
                    else: 
                        tau_mat=np.inf
                    flog.write(str(tau_mat)+'\t')
                flog.write('\n')
            flog.write('\n')
            
            
    nu_e_N1.append(nu_T_N1)
    nu_e_N6.append(nu_T_N6)
    nu_e_N50.append(nu_T_N50)
    nu_e_hex.append(nu_T_hex)

    phi_F_N1.append(phi_T_N1)
    phi_F_N6.append(phi_T_N6)
    phi_F_N50.append(phi_T_N50)
    phi_F_hex.append(phi_T_hex)

flog.close()

#plotting
N=[]
for N_LHC in range(1,N_LHC_max,N_LHC_step):
    N.append(N_LHC)

#All temperatures
for t, Temp in enumerate(Ts):     
#    plt.fill_between(N,nu_e_N1[t],nu_e_N5[t],color=colours[t],alpha=0.2)
#    plt.fill_between(N,nu_e_N5[t],nu_e_N50[t],color=colours[t],alpha=0.1)

    plt.plot(N,nu_e_N1[t],color=colours[t],label=Temp+' K',linestyle='-')
    plt.scatter(N,nu_e_N1[t],color=colours[t],marker='.')
    plt.plot(N,nu_e_N6[t],color=colours[t],linestyle='--')
    plt.scatter(N,nu_e_N6[t],color=colours[t],marker='.')
    plt.plot(N,nu_e_N50[t],color=colours[t],linestyle=':')
    plt.scatter(N,nu_e_N50[t],color=colours[t],marker='.')
    plt.plot(N,nu_e_hex[t],color=colours[t],linestyle='-',linewidth=5.0,alpha=0.4)

plt.vlines(5.0,-5.0,105.0,color='k',linestyle='--',alpha=0.2)
plt.xlabel('$N_{LHC}$',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.ylabel('$\\nu_{e}$ ($s^{-1}$)',fontsize=14)
plt.tick_params('y',labelsize=12)
plt.xlim(-0.5,75)
plt.ylim(-0.5,105.0)
plt.legend(fontsize=12,loc='upper right')
plt.tight_layout()
plt.savefig('PSII_branched_acclimate.pdf')
plt.show()


#Zoomed in look at Ts = 2300 K  
#plt.fill_between(N,nu_e_N1[0],nu_e_N6[0],color=colours[0],alpha=0.2)
#plt.fill_between(N,nu_e_N6[0],nu_e_N50[0],color=colours[0],alpha=0.1)

plt.vlines(5.0,-5.0,105.0,color='k',linestyle='--',alpha=0.2)
plt.plot(N,nu_e_N1[0],color=colours[0],label='$N_{b} = 1$',linestyle='-')
plt.scatter(N,nu_e_N1[0],color=colours[0],marker='.')
plt.plot(N,nu_e_N6[0],color=colours[0],label='$N_{b} = 6$',linestyle='--')
plt.scatter(N,nu_e_N6[0],color=colours[0],marker='.')
plt.plot(N,nu_e_N50[0],color=colours[0],label='$N_{b} = 50$',linestyle=':')
plt.scatter(N,nu_e_N50[0],color=colours[0],marker='.')
plt.plot(N,nu_e_hex[0],color=colours[0],linestyle='-',linewidth=5.0,alpha=0.4,
         label='hex')

plt.xlabel('$N_{LHC}$',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.ylabel('$\\nu_{e}$ ($s^{-1}$)',fontsize=14)
plt.tick_params('y',labelsize=12)
plt.xlim(-0.5,55)
plt.ylim(-0.5,35.0)
plt.legend(fontsize=12,loc='upper left')
plt.tight_layout()
plt.savefig('PSII_branched_acclimate_2300K.pdf')
plt.show()

#phi_F
plt.vlines(5.0,-5.0,105.0,color='k',linestyle='--',alpha=0.2)
plt.plot(N,phi_F_N1[0],color=colours[0],linestyle='-',label='$N_{b}=1$')
plt.scatter(N,phi_F_N1[0],color=colours[0],marker='.')
plt.plot(N,phi_F_N6[0],color=colours[0],linestyle='--',label='$N_{b}=6$')
plt.scatter(N,phi_F_N6[0],color=colours[0],marker='.')
plt.plot(N,phi_F_N50[0],color=colours[0],linestyle=':',label='$N_{b}=50$')
plt.scatter(N,phi_F_N50[0],color=colours[0],marker='.')

plt.plot(N,phi_F_hex[0],color=colours[0],linestyle='-',linewidth=5.0,alpha=0.4,
         label='hex')

#plt.plot(N,nu_e_N5[t],color=colours[t],linestyle='--')
#plt.scatter(N,nu_e_N5[t],color=colours[t],marker='.')
#plt.plot(N,nu_e_N50[t],color=colours[t],linestyle=':')
#plt.scatter(N,nu_e_N50[t],color=colours[t],marker='.')

plt.xlabel('$N_{LHC}$',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.ylabel('$\phi_{e}$',fontsize=14)
plt.tick_params('y',labelsize=12)
plt.legend(fontsize=12,loc='upper right')
plt.xlim(-0.5,55)
plt.ylim(-0.05,1.05)
plt.tight_layout()
plt.savefig('PSII_branched_phi_F.pdf')
plt.show()


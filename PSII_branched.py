# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:05:13 2023

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

N1=18.0
N2=24.0
N_RC=1
N_LHC=6


k_diss=1.0/4.0E-9
k_hop=1.0/5.0E-12
k_trap=1.0/5.0E-12
k_con=1.0/10.0E-3
K_12=1.0/1.0E-12
K_LHC_RC=1.0/18.0E-12

T=300.0

G_params=PSII.G_params 
G_params[0]=1.6E-18 #retune the absorption rate

k_params = (k_diss,k_hop,k_trap,k_con,K_12,K_LHC_RC) 

phi_F_N1, nu_e_N1=[],[]
phi_F_N6, nu_e_N6=[],[]
phi_F_hex, nu_e_hex=[],[]
for Temp in Ts:      
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

    #single branch calculation
    N_branch=1
    Size_params = (N1, N2, N_RC, N_LHC, N_branch) 
    out_1branch=lattice.Antenna_branched(l,Ip_y,Size_params,G_params,k_params,T)  
    
    if out_1branch['nu_e']<=100.0:    
        nu_e_N1.append(out_1branch['nu_e'])
    else:
#        nu_e_N1.append(100.0)
        nu_e_N1.append(out_1branch['nu_e'])

    phi_F_N1.append(out_1branch['phi_F'])

    #6 branch calculation
    N_branch=6
    Size_params = (N1, N2, N_RC, N_LHC, N_branch) 
    out_6branch=lattice.Antenna_branched(l,Ip_y,Size_params,G_params,k_params,T)  

    if out_6branch['nu_e']<=100.0:    
        nu_e_N6.append(out_6branch['nu_e'])
    else:
#        nu_e_N6.append(100.0)
        nu_e_N6.append(out_6branch['nu_e'])
        
    phi_F_N6.append(out_6branch['phi_F'])
    
    #hex calculation
    out_hex=lattice.Antenna_hex(l,Ip_y,Size_params,G_params,k_params,T)
    if out_hex['nu_e']<=100.0:    
        nu_e_hex.append(out_hex['nu_e'])
    else:
#        nu_e_hex.append(100.0)
        nu_e_hex.append(out_hex['nu_e'])
        
    phi_F_hex.append(out_hex['phi_F'])

print('nu_e (N_branch=1, Ts=5800K)', nu_e_N1[-1])
print('phi_F (N_branch=1) = ', phi_F_N1[0])
print('nu_e (N_branch=6, Ts=5800K)', nu_e_N6[-1])
print('phi_F (N_branch=6) = ', phi_F_N6[0])
print('nu_e (hex, Ts=5800K)', nu_e_hex[-1])
print('phi_F (hex) = ', phi_F_hex[0])

print('\ngamma_total = ',out_hex['gamma_total'])
print('gamma1 = ',out_hex['gamma1'])
print('gamma2 = ',out_hex['gamma2'])

#plot the rate and effiiency of PSII for different PS
fig = plt.figure()
plt.scatter(Ts,nu_e_N1,color='maroon',marker='.')
plt.plot(Ts,nu_e_N1,color='maroon',label='$N_{LHC} = 6$, $N_{b} = 1$',linestyle='--')
plt.scatter(Ts,nu_e_N6,color='darkblue',marker='.')
plt.plot(Ts,nu_e_N6,color='darkblue',label='$N_{LHC} = 6$, $N_{b} = 6$',linestyle='--')
plt.plot(Ts,nu_e_hex,color='darkgreen',linewidth=5.0,
         alpha=0.4,label='$N_{LHC} = 6$, $hex$',linestyle='-')
plt.xlabel('$T_{s}$ (K)',fontsize=12)
plt.ylabel('$\\nu_{e}$ ($s^{-1}$)',fontsize=14)
plt.tick_params('x',labelsize=12)
plt.tick_params('y',labelsize=12)
plt.ylim(-5,105)

fig.legend(frameon=True,fontsize=14,loc=(0.16,0.68))
plt.tight_layout()
plt.savefig('PSII_branched.pdf')
plt.show()


#output Adj_mat
Adj_mat=out_hex['Adj_mat']
fout=open('hex_Adj_mat.txt','w')
for line in Adj_mat:
    for element in line:
        fout.write(str(element)+'\t')
    fout.write('\n')
fout.close()

'''
fig.legend(frameon=True,fontsize=14,loc=(0.70,0.19))
plt.tight_layout()

plt.show()

fout=open('chain_lattice_output.txt','w')
for i in range(2*N_LHC+2):
    for j in range(2*N_LHC+2):
        fout.write(str(tau_mat[i][j])+'\t')
    fout.write('\n')

fout.close()

nu_e_list, phi_F_list=[],[]
for Temp in Ts: 
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
    
    out_1branch=lattice.Antenna_branched(l,Ip_y,Size_params,G_params,k_params,300.0)  
    
    nu_e_T=out_1branch['nu_e']
    phi_F_T=out_1branch['phi_F']
    gamma_LHC_T=out_1branch['gamma_LHC']
    gamma_total_T=out_1branch['gamma_total']
    Neq=out_1branch['Neq']
    Adj_mat=out_1branch['Adj_mat']
    K_mat=out_1branch['K_mat']

    nu_e_list.append(nu_e_T)
    phi_F_list.append(phi_F_T)
    
    print('Neq = ',Neq)
    print('gamma_total = ', gamma_total_T)
    
    fout.write(str(Temp)+'K\n K matrix\n')
    fout.write('\t')
    for i in range(N_LHC+2):
        fout.write(str(i)+'\t')
    fout.write('\n')
    for i in range(N_LHC+2):
        fout.write(str(i)+'\t')
        for j in range(N_LHC+2):
            fout.write(str(K_mat[i][j])+'\t')
        fout.write('\n')    

plt.plot(Ts,nu_e_list)
plt.show()
plt.plot(Ts,phi_F_list)
plt.show()

fout.close()    
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:45:39 2023

-This code simulates a lattice model of a single RC surrounded by a modular antenna.
-The antenna is composed of lattice of equivalent 'sites' that absorb and light 
accoridng to the overlap of their absorption profile and the local spectral flux.
-The excitons then migrate through the structure.

@author: C Duffy
"""
import numpy as np
import PSII_params as PSII
from scipy.constants import h as h
from scipy.constants import c as c
from scipy.constants import Boltzmann as kB
import matplotlib.pyplot as plt

'''***************************************************************************
Returns a list of n centred hexagonal numbers.

Input: n = maximum degree

***************************************************************************'''
def Hex_n(n):
    H_n=[]
    for i in range(1,n+1):
        H_n.append(i**3-(i-1)**3)
    return(H_n)


'''***************************************************************************
Returns a normalized Gaussian absorption line.

Input: l = array of wavelengths (nm)
       lp = peak wavelength (nm)
       w = standard deviation (nm) 

***************************************************************************'''
def gauss_abs(l,lp,w):
    exponent=-((l-lp)**2)/(2.0*w**2)
    gauss_y=np.exp(exponent)  
    #normalize
    N=np.trapz(gauss_y,l)
    return(gauss_y/N)    

'''***************************************************************************
Returns the overlap of two functions with a common x-axis

Input: l = array of wavelengths (nm)
       f1 = function 1 y-values for the points in l
       f2 = function 2 y-values for the points in l

***************************************************************************'''
def Olap_Int(l,f1,f2):
    olap=[a*b for a,b in zip(f1,f2)]
    return(np.trapz(olap,l))

'''***************************************************************************
Returns the thermodynamic parameters (enthalpy, entropy and Helmholtz) for the 
transfer of energy between two antenna domains described by a their absorption 
characterisics

Input: lp1 = centre wavelength of domain 1
       lp2 = centre wavelength of domain 2
       f2 = function 2 y-values for the points in l
       N12 = n1/n2 where ni is the number of thermodynamically equivalent 
             states in domain i
       T is the temperature in K
       
***************************************************************************'''
def deltaG(lp1,lp2,N12,T):
    
    lp1=lp1*1.0E-9 #convert to SI units
    lp2=lp2*1.0E-9
    
    H_12=h*c*(lp1-lp2)/(lp1*lp2) #Enthalpy change on moving from domain 1 to domain 2
    H_21=h*c*(lp2-lp1)/(lp1*lp2) #Enthalpy change for the reverse transfer
    
    S_12=kB*np.log(1.0/N12) #Entropy change for transfer from domain 1 to domain 2
    S_21=kB*np.log(N12) #Entropy change for the reverse transfer
    
    G_12=H_12-(T*S_12)
    G_21=H_21-(T*S_21)
    
    return([H_12,H_21],[S_12,S_21],[G_12,G_21])

'''***************************************************************************
Define transfer matrix, K, for a hexagonal antenna-RC configuration.
We assume that the RC sits at the central site and antenna grows  

Inputs: Size_params = (N1, N2, N_RC, N_LHC) 

        G_params = (sig,B12,lp1,w1,lp2,w2) parameters associated with photon 
                    absorption rate.

        k_params = (k_diss,k_hop,k_trap,k_con,K_12,K_LHC_RC) 

***************************************************************************'''

def Antenna_hex(l,Ip_y,Size_params,G_params,k_params,T):
    
    #(1) Spectral parameters
    sig,B12=G_params[0],G_params[1]
    lp1,w1=G_params[2],G_params[3]
    lp2,w2=G_params[4],G_params[5]
    lpRC,wRC=G_params[6],G_params[7] #for later developments involing spectral overlap
    
    #(2) Size parameters
    N1, N2, N_RC, N_LHC=Size_params[0], Size_params[1], Size_params[2], Size_params[3] 
    
    #(3) Rate constants
    k_diss, k_hop=k_params[0], k_params[1] #dissipation and transfer in the antenna
    k_trap, k_con=k_params[2], k_params[3] #irreversible excitation trapping and converion to electrons
    K_12, K_LHC_RC=k_params[4], k_params[5]

    n1n2=N1/N2 #ratio of thermodynamically equivalent states in G1 and G2 
    thermo12=deltaG(lp1,lp2,n1n2,T) #thermodynamic parameters for G1 -> G2
    G_12,G_21=thermo12[2][0],thermo12[2][1]
    
    if G_12==0.0:
        k_12, k_21=K_12, K_12
    elif G_12<0.0:
        k_12, k_21=K_12, K_12*np.exp(-G_21/(kB*T))
    elif G_12>0.0:
        k_12, k_21=K_12*np.exp(-G_12/(kB*T)), K_12
    
    nRCnLHC=N2/N_RC
    thermo2RC=deltaG(lp2,lpRC,nRCnLHC,T) #thermodynamic parameters for G2 -> GRC
    G_LHC_RC,G_RC_LHC=thermo2RC[2][0],thermo2RC[2][1]
    
    if G_LHC_RC==0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC
    elif G_LHC_RC<0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC*np.exp(-G_RC_LHC/(kB*T))
    elif G_LHC_RC>0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*np.exp(-G_LHC_RC/(kB*T)), K_LHC_RC    
    
    #(4) Convert the spectral iradiance into the spectral photon flux
    fp_y=np.zeros(len(Ip_y)) #photon flux
    for i, item in enumerate(Ip_y):
        fp_y[i]=item*((l[i]*1.0E-9)/(h*c)) #factor of 1E-9 since l is in nm.
    
    #(6) Calculate the absorption linshape for each LHC    
    g1_l=gauss_abs(l,lp1,w1)
    g2_l=gauss_abs(l,lp2,w2)
    
    #(7) Calculate gamma1 and gamma2 (photon input rate) per LHC
    gamma1=sig*B12*Olap_Int(l,fp_y,g1_l)
    gamma2=sig*Olap_Int(l,fp_y,g2_l)

    #(8) Build an adjacancy matrix
    #Site 0 is the trap state so canbe visualized as lying out of the plane of the lattice
    #connected only to site 1 which is the RC. 
    #Site 1 is the RC and the centre of the lattice. Always starting directly above the
    #site 1 we add sites in a clockwise direction filling tabulating which previously-placed
    #sites are neighbours of the new one. 
    #We therefore fill out the bottom triangle of the matrix before generating the upper triangle 
    #by transpose
    Adj_mat_upper=np.zeros((N_LHC+2,N_LHC+2)) #Number of LHCs plus an RC and RC trap
    Adj_mat_lower=np.zeros((N_LHC+2,N_LHC+2))
    
    #We first have to generate a list of the numbers that refer to sites belonging to 
    #the 'spokes' of the hexagon
    #These sequence are related to 'centred hexagonal numbers'
    top_spoke=[]
    top_right_spoke=[]
    bottom_right_spoke=[]
    bottom_spoke=[]
    bottom_left_spoke=[]
    top_left_spoke=[]
    for n in range(1,N_LHC):
        top=3*n*n-3*n+2
        top_right=3*n*n-2*n+2
        bottom_right=3*n*n-n+2
        bottom=3*n*n+2
        bottom_left=3*n*n+n+2
        top_left=3*n*n+2*n+2
        
        top_spoke.append(top)
        top_right_spoke.append(top_right)
        bottom_right_spoke.append(bottom_right)
        bottom_spoke.append(bottom)
        bottom_left_spoke.append(bottom_left)
        top_left_spoke.append(top_left)

    for i in range(N_LHC+2):
                    
        if i==1: #site 1 is the RC so the only previous site it is connected to is the trap
            Adj_mat_lower[i][0]=1.0

        elif i>1 and i<=7: #The first concentric layer surrounding the RC
            Adj_mat_lower[i][1]=1.0
            Adj_mat_lower[i][i-1]=1.0
            if i==7: #the last LHC in the first concentric layer
                Adj_mat_lower[i][2]=1.0

        elif i>7: #i.e. one of the outer layers
            if i in top_spoke:
                ind=top_spoke.index(i)
                Adj_mat_lower[i][top_spoke[ind-1]]=1.0
                
            elif i in top_right_spoke:
                ind=top_right_spoke.index(i)
                Adj_mat_lower[i][top_right_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in bottom_right_spoke:
                ind=bottom_right_spoke.index(i)
                Adj_mat_lower[i][bottom_right_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in bottom_spoke:
                ind=bottom_spoke.index(i)
                Adj_mat_lower[i][bottom_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in bottom_left_spoke:
                ind=bottom_left_spoke.index(i)
                Adj_mat_lower[i][bottom_left_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in top_left_spoke:
                ind=top_left_spoke.index(i)
                Adj_mat_lower[i][top_left_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            else:
                #For the rest we need to work out which concentric layer and which 
                #edge of the hexagon i refers to. 
                #First work out the layer using the top_spoke list
                layer=0
                for n, Hn in enumerate(top_spoke):
                    if i>=Hn:
                        layer=layer+1
                    else:
                        break
        
                #Next we have to work out which triangular 'wedge' site i belongs to
                wedge=0 #1=upper right, 2=right, 3=lower right,4=lower left, 5=left, 6=upper left
                
                #a list of the spoke sites in this layer starting 
                spoke_layer=[top_spoke[layer-1],top_right_spoke[layer-1],bottom_right_spoke[layer-1],
                             bottom_spoke[layer-1],bottom_left_spoke[layer-1],top_left_spoke[layer-1]] 
                for n, Hn in enumerate(spoke_layer):
                    if i>Hn:
                        wedge=wedge+1
                    else:
                        break            

                #Adding the neighbouring sites
                if wedge<=5: #have to treat the last wedge separately due to the 
                             #discontinuity in numbering
                    Adj_mat_lower[i][i-((6*(layer-1))+wedge)]=1.0
                    Adj_mat_lower[i][i-((6*(layer-1))+wedge-1)]=1.0
                    Adj_mat_lower[i][i-1]=1.0
                
                else:
                    Adj_mat_lower[i][i-((6*(layer-1))+wedge)]=1.0
                    Adj_mat_lower[i][i-1]=1.0
                    if i+1 in top_spoke:
                        Adj_mat_lower[i][i-(6*layer)+1]=1.0
                        Adj_mat_lower[i][i-(6*layer+6*(layer-1)-1)]=1.0
                    else:
                        Adj_mat_lower[i][i-((6*(layer-1))+wedge-1)]=1.0


    #Finally, having calculated thelower triangle we can calculated the upper
    #triangle via matrix transpose
    Adj_mat_upper=np.transpose(Adj_mat_lower)
    
    Adj_mat=np.zeros((N_LHC+2,N_LHC+2))
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            Adj_mat[i][j]=Adj_mat_lower[i][j]+Adj_mat_upper[i][j]
            
    #(9) Build the 'K' matrix: The dimensions include the trap, the RC, all LHCII
    # and a localized Chl b domain inside each LHCII
    
    K_mat=np.zeros(((2*N_LHC+2,2*N_LHC+2)))        

    #Fill the upper, inter_LHC block first
    #Define off-diagonal elements first. 
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            if i!=j: 
                if i==0: #transfer to the trap is unidirectional from the RC 
                    K_mat[i][j]=Adj_mat[j][i]*k_trap

                elif i==1: #transfer to the RC (excluding transfer back from the trap)
                    if j!=0:
                        K_mat[i][j]=Adj_mat[j][i]*k_LHC_RC

                elif i>=2 and i<=7: #i.e. in first concentric ring around the RC
                    if j==1: 
                        K_mat[i][j]=Adj_mat[j][i]*k_RC_LHC
                    else: 
                        K_mat[i][j]=Adj_mat[j][i]*k_hop
                else:
                    K_mat[i][j]=Adj_mat[j][i]*k_hop

    #Diagonal elements
    K_mat[0][0]=-k_con
    for i in range(1,N_LHC+2):
        for j in range(N_LHC+2):
            if i!=j:
                K_mat[i][i]=K_mat[i][i]-K_mat[j][i]
        
        if i>=2: 
            K_mat[i][i]=K_mat[i][i]-k_diss #dissipation loss to the antenna complexes
            K_mat[i][i]=K_mat[i][i]-k_21 #uphill transfer to Chl b
            
        
    #Now we can fill out the intra-LHC relaxation terms
    for i in range(2,N_LHC+2):
        K_mat[i][i+N_LHC]=k_12
        K_mat[i+N_LHC][i]=k_21
        
    for i in range(N_LHC+2,2*N_LHC+2):
        K_mat[i][i]=K_mat[i][i]-k_12 #relaxation to Chla
        K_mat[i][i]=K_mat[i][i]-k_diss #relaxation directly to the ground state

    #(10) Generating terms
    gamma_vec=np.zeros((2*N_LHC+2))
    for i in range(2,N_LHC+2):
        gamma_vec[i]=-gamma2
    
    for i in range(N_LHC+2,2*N_LHC+2):
        gamma_vec[i]=-gamma1

    #(11) Solve the kinetics
    K_inv=np.linalg.inv(K_mat)
    N_eq=np.zeros((2*N_LHC+2))
    for i in range(2*N_LHC+2):
        for j in range(2*N_LHC+2):
            N_eq[i]=N_eq[i]+K_inv[i][j]*gamma_vec[j]
    
    #(12) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((2*N_LHC+2,2*N_LHC+2))
    for i in range(2*N_LHC+2):
        for j in range(2*N_LHC+2):
            if K_mat[i][j]!=0.0:
                tau_mat[i][j]=(1.0/K_mat[i][j])/1.0E-12
            else: 
                tau_mat[i][j]=np.inf
    
    #(b) Electron output rate
    nu_e=k_con*N_eq[0]
    
    #(c) electron conversion quantum yield
    sum_rate=0.0
    for i in range(2,2*N_LHC+2):
        sum_rate=sum_rate+(k_diss*N_eq[i])
    
    phi_F=nu_e/(nu_e+sum_rate)

    output_dict={
        'Adj_mat': Adj_mat,
        'K_mat': K_mat,
        'tau_mat': tau_mat,
        'N_eq': N_eq,
        'nu_e': nu_e,
        'phi_F': phi_F,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'gamma_LHC': gamma1+gamma2,
        'gamma_total': float(N_LHC)*(gamma1+gamma2),
        'DeltaH_12': [thermo12[0][0],thermo12[0][1]],
        'DeltaS_12': [thermo12[1][0],thermo12[1][1]],
        'DeltaG_12': [thermo12[2][0],thermo12[2][1]],
        'DeltaH_2RC': [thermo2RC[0][0],thermo2RC[0][1]],
        'DeltaS_2RC': [thermo2RC[1][0],thermo2RC[1][1]],
        'DeltaG_2RC': [thermo2RC[2][0],thermo2RC[2][1]],
        'k_12': [k_12, k_21],
        'k_LHC_RC': [k_LHC_RC,k_RC_LHC]
        }
    
    return(output_dict)



'''***************************************************************************
Solve the steady state for a linear antenna-RC configuration.
We assume that the antenna is a series of linear chains of LHCs
radiating from central RC. 

In this version the Chlb -> Chla step is treated explcitly in the topology of
the problem

Inputs: l = array of wavelengths (nm)
        Ip_y = spectral flux values for wavelengths in l
        Size_params = (N1, N2, N_RC, N_LHC, N_branch) 
        G_params=(sig,B12,lp1,w1,lp2,w2,lpRC,wRC) parameters associated with photon 
                    absorption rate.
        k_params = (k_diss,k_hop,k_trap,k_con,K_12,K_LHC_RC) are the rate constants for the 
                    transfer/decay pathways (s^{-1})
        T = temperature in K

***************************************************************************'''

def Antenna_branched(l,Ip_y,Size_params,G_params,k_params,T):

    #(1) Spectral parameters
    sig,B12=G_params[0],G_params[1]
    lp1,w1=G_params[2],G_params[3]
    lp2,w2=G_params[4],G_params[5]
    lpRC,wRC=G_params[6],G_params[7] #for later developments involing spectral overlap
    
    #(2) Size parameters
    N1, N2, N_RC=Size_params[0], Size_params[1], Size_params[2] 
    N_LHC, N_branch=Size_params[3], Size_params[4]
    
    #(3) Rate constants
    k_diss, k_hop=k_params[0], k_params[1] #dissipation and transfer in the antenna
    k_trap, k_con=k_params[2], k_params[3] #irreversible excitation trapping and converion to electrons
    K_12, K_LHC_RC=k_params[4], k_params[5]

    n1n2=N1/N2 #ratio of thermodynamically equivalent states in G1 and G2 
    thermo12=deltaG(lp1,lp2,n1n2,T) #thermodynamic parameters for G1 -> G2
    G_12,G_21=thermo12[2][0],thermo12[2][1]
    
    if G_12==0.0:
        k_12, k_21=K_12, K_12
    elif G_12<0.0:
        k_12, k_21=K_12, K_12*np.exp(-G_21/(kB*T))
    elif G_12>0.0:
        k_12, k_21=K_12*np.exp(-G_12/(kB*T)), K_12
    
    nRCnLHC=N2/N_RC
    thermo2RC=deltaG(lp2,lpRC,nRCnLHC,T) #thermodynamic parameters for G2 -> GRC
    G_LHC_RC,G_RC_LHC=thermo2RC[2][0],thermo2RC[2][1]
    
    if G_LHC_RC==0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC
    elif G_LHC_RC<0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC*np.exp(-G_RC_LHC/(kB*T))
    elif G_LHC_RC>0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*np.exp(-G_LHC_RC/(kB*T)), K_LHC_RC    
    
    #(4) Convert the spectral iradiance into the spectral photon flux
    fp_y=np.zeros(len(Ip_y)) #photon flux
    for i, item in enumerate(Ip_y):
        fp_y[i]=item*((l[i]*1.0E-9)/(h*c)) #factor of 1E-9 since l is in nm.
    
    #(6) Calculate the absorption linshape for each LHC    
    g1_l=gauss_abs(l,lp1,w1)
    g2_l=gauss_abs(l,lp2,w2)
    
    #(7) Calculate gamma1 and gamma2 (photon input rate) per LHC
    gamma1=sig*B12*Olap_Int(l,fp_y,g1_l)
    gamma2=sig*Olap_Int(l,fp_y,g2_l)
    
    #(8) Build an adjacency matrix
    #Each LHC contains two internal sites (Chl b and Chl a) domains, plus the 
    #RC site and the trap site
    Adj_mat=np.zeros((2*N_LHC+2,2*N_LHC+2))

    #First fill out the connections between the LHCs (upper diagonal block)
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            if i!=j: #explicitly exclude diagonal (self-interaction) terms 
                
                if i==0 and j==1: #trap connected only to RC
                    Adj_mat[i][j]=1.0
                    
                elif i==1: 
                    if j==0: #transfer from the RC to the trap
                        Adj_mat[i][j]=1.0
                    
                    #the RC neighbours the first ring of LHCs
                    elif j>=2 and j<=N_branch+1:
                        Adj_mat[i][j]=1.0
            
                #site is an antenna complex in the inner ring
                elif i>=2 and i<=N_branch+1: 
                    if j==1: #they neighbour the RC
                        Adj_mat[i][j]=1.0
                    elif j==i+N_branch: #and the LHC next along the branch
                        Adj_mat[i][j]=1.0

                elif i>N_branch+1: 
                    if j==i-N_branch or j==i+N_branch:
                        Adj_mat[i][j]=1.0
                        
    #Now add the internal connections between the Chl b and Chl a domains in 
    #each LHC
    for i in range(2, N_LHC+2):
            Adj_mat[i][i+N_LHC]=1.0     
            Adj_mat[i+N_LHC][i]=1.0

    #(9) Use the adjancency matrix to assemble the transfer matrix
    K_mat=np.zeros((2*N_LHC+2,2*N_LHC+2))  

    #Lets assemble the inter-subunit diagonal block first
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            #Defining the off-diaganal ('gain') terms first
            if i!=j: 
                if i==0: #the trap can only recevie energy from the RC 
                    K_mat[i][j]=Adj_mat[j][i]*k_trap

                elif i==1: #the RC can receive energy from any coupled LHC
                    if j>0 and j<=N_branch+2: #exclude transferback from the trap
                        K_mat[i][j]=Adj_mat[j][i]*k_LHC_RC

                elif i>=2 and i<=N_branch+1: #one of the LHCs coupled to the RC
                    if j==1: #The RC
                        K_mat[i][j]=Adj_mat[j][i]*k_RC_LHC
                    else:
                        K_mat[i][j]=Adj_mat[j][i]*k_hop
                        
                else:
                    K_mat[i][j]=Adj_mat[j][i]*k_hop

    #Diagonal elements
    K_mat[0][0]=-k_con
    for i in range(1,N_LHC+2):
        for j in range(N_LHC+2):
            if i!=j:
                K_mat[i][i]=K_mat[i][i]-K_mat[j][i]
        
        if i>=2: 
            K_mat[i][i]=K_mat[i][i]-k_diss #dissipation loss to the antenna complexes
            K_mat[i][i]=K_mat[i][i]-k_21 #uphill transfer to Chl b
            
        
    #Now we can fill out the intra-LHC relaxation terms
    for i in range(2,N_LHC+2):
        K_mat[i][i+N_LHC]=k_12
        K_mat[i+N_LHC][i]=k_21
        
    for i in range(N_LHC+2,2*N_LHC+2):
        K_mat[i][i]=K_mat[i][i]-k_12 #relaxation to Chla
        K_mat[i][i]=K_mat[i][i]-k_diss #relaxation directly to the ground state

    #(10) Generating terms
    gamma_vec=np.zeros((2*N_LHC+2))
    for i in range(2,N_LHC+2):
        gamma_vec[i]=-gamma2
    
    for i in range(N_LHC+2,2*N_LHC+2):
        gamma_vec[i]=-gamma1

    #(11) Solve the kinetics
    K_inv=np.linalg.inv(K_mat)
    N_eq=np.zeros((2*N_LHC+2))
    for i in range(2*N_LHC+2):
        for j in range(2*N_LHC+2):
            N_eq[i]=N_eq[i]+K_inv[i][j]*gamma_vec[j]
    
    #(12) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((2*N_LHC+2,2*N_LHC+2))
    for i in range(2*N_LHC+2):
        for j in range(2*N_LHC+2):
            if K_mat[i][j]!=0.0:
                tau_mat[i][j]=(1.0/K_mat[i][j])/1.0E-12
            else: 
                tau_mat[i][j]=np.inf
    
    #(b) Electron output rate
    nu_e=k_con*N_eq[0]
    
    #(c) electron conversion quantum yield
    sum_rate=0.0
    for i in range(2,2*N_LHC+2):
        sum_rate=sum_rate+(k_diss*N_eq[i])
    
    phi_F=nu_e/(nu_e+sum_rate)

    output_dict={
        'Adj_mat': Adj_mat,
        'K_mat': K_mat,
        'tau_mat': tau_mat,
        'N_eq': N_eq,
        'nu_e': nu_e,
        'phi_F': phi_F,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'gamma_LHC': gamma1+gamma2,
        'gamma_total': float(N_LHC)*(gamma1+gamma2),
        'DeltaH_12': [thermo12[0][0],thermo12[0][1]],
        'DeltaS_12': [thermo12[1][0],thermo12[1][1]],
        'DeltaG_12': [thermo12[2][0],thermo12[2][1]],
        'DeltaH_2RC': [thermo2RC[0][0],thermo2RC[0][1]],
        'DeltaS_2RC': [thermo2RC[1][0],thermo2RC[1][1]],
        'DeltaG_2RC': [thermo2RC[2][0],thermo2RC[2][1]],
        'k_12': [k_12, k_21],
        'k_LHC_RC': [k_LHC_RC,k_RC_LHC]
        }
        
    return(output_dict)    
    

'''***************************************************************************
Solve the steady state for a phycobilisome-like antenna.
A series of 'branches' radiate out from a central RC. The subunits contain different 
pools of pigments, with absorption characteristics defined by the user. This is 
a schematic representation of the phycobilisome which has a broad range of pigments, 
covering a large wavelength range. Our hypothesis is that this makes energy transfer 
to the RC highly enthalpically favourable. 

Inputs: l = array of wavelengths (nm)
        Ip_y = spectral flux values for wavelengths in l
        
        Branch_params=(N_b,branch)
            N_b = the number of antenna branches
            branch = (sN, sN-1, ..., s2, s1) where s1 is the furthest most 
                                             subunit from the RC
                si = (Ni, sig_i, lpi, wi) 
        
        RC_params = (N_RC, sig_RC, lp_RC, w_RC)
 
        k_params = (k_diss,k_trap,k_con,K_hop,K_LHC_RC) are the rate constants for the 
                    transfer/decay pathways (s^{-1})
        T = temperature in K

***************************************************************************'''

def Antenna_branched_funnel(l,Ip_y,Branch_params,RC_params,k_params,T):
    
    #(1) Unpack parameters and establish dimensionality of the antenna
    k_diss, k_trap, k_con, K_hop, K_LHC_RC=k_params
    N_RC, sig_RC, lp_RC, w_RC=RC_params
    
    N_b=Branch_params[0] #number of branches
    N_s=len(Branch_params)-1 #number of subunits per branch

    subunits=[]
    for i in range(1,N_s+1):
        subunits.append(Branch_params[i])
    
    #(2) Absorption rates
    fp_y=np.zeros(len(Ip_y)) #photon flux
    for i, item in enumerate(Ip_y):
        fp_y[i]=item*((l[i]*1.0E-9)/(h*c)) #factor of 1E-9 since l is in nm.
                 
    gamma_b=[]
    for i in range(N_s):
        lineshape=gauss_abs(l,subunits[i][2],subunits[i][3])
        gamma_i=subunits[i][1]*Olap_Int(l,fp_y,lineshape)
        gamma_b.append(gamma_i)
    
    #(3) Calculate rate constants
    
    #transfer between RC and an antenna branch
    nRCnLHC=subunits[0][0]/N_RC
    thermoRC=deltaG(subunits[0][2],lp_RC,nRCnLHC,T) #thermodynamic parameters for G2 -> GRC
    G_LHC_RC,G_RC_LHC=thermoRC[2][0],thermoRC[2][1]
    
    if G_LHC_RC==0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC
    elif G_LHC_RC<0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC*np.exp(-G_RC_LHC/(kB*T))
    elif G_LHC_RC>0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*np.exp(-G_LHC_RC/(kB*T)), K_LHC_RC        
        
    #transfer between subunits
    if N_s>1:
        K_b=np.zeros((N_s,N_s))
        for i in range(N_s-1): #working from the inner subunit to the outer one
            n_ratio=subunits[i][0]/subunits[i+1][0]
            thermo=deltaG(subunits[i][2],subunits[i+1][2],n_ratio,T)
            G_out_in, G_in_out=thermo[2][0], thermo[2][1]
        
            if G_out_in==0.0: #i.e. the subunits are identical
                K_b[i][i+1], K_b[i+1][i]=K_hop, K_hop #forward rate        
            elif G_out_in<0.0: #i.e. forward transfer is favoured
                K_b[i][i+1], K_b[i+1][i]=K_hop, K_hop*np.exp(-G_in_out/(kB*T))
            elif G_out_in>0.0: #i.e. forward transfer is limited
                K_b[i][i+1], K_b[i+1][i]=K_hop*np.exp(-G_out_in/(kB*T)), K_hop
            
    #(4) Assemble the Transfer Weighted Adjacency matrix 
    #the numbering of the sub-units has changed relative to the initial branched
    #antnne. Site 0 is the trap, 1 is the RC. 
    #2,3,...,N_s+1 is the first branch
    #N_s+2, N_s+3, ..., 2N_s+1 is the second branch 
    #2N_s+2, 2N_s+3, ... 3Ns+1 is the third branch
    #and so on. 
    
    TW_Adj_mat=np.zeros(((N_b*N_s)+2,(N_b*N_s)+2))

    start_index=np.zeros((N_b)) #list of indices of the starting (RC-adjacent) 
    for i in range(N_b):        #antenna sites of the branches
        start_index[i]=i*N_s+2
           
    end_index=np.zeros((N_b)) #ditto for the ends of the branches
    for i in range(N_b):
        end_index[i]=(i+1)*N_s+1
        
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            #we ignore i==0 as although the trap is connected to the RC, trapping
            #is irreversibile (the rate of de-trapping is 0)
            if i==1: #i.e. transfer from the RC
                if j==0: #to the trap
                    TW_Adj_mat[i][j]=k_trap 
                elif j in start_index: #to the LHCs at the start of the branches
                    TW_Adj_mat[i][j]=k_RC_LHC

            elif i in start_index: #transfer from the inner ring of LHCs
                if j==1: #to the RC 
                    TW_Adj_mat[i][j]=k_LHC_RC
                elif j==i+1: #to the next subunit along the chain
                    if N_s>1:
                        TW_Adj_mat[i][j]=K_b[0][1] 
                    
    #now fill in the nearest-neighbour adjacencies along the branches
    if N_s>1:
        for i in range(1,N_s): #exclude the first subunit which has bee accounted for above
            for j in start_index:
                if i+j in end_index:                
                    TW_Adj_mat[int(j+i)][int(j+i-1)]=K_b[i][i-1]
                else:
                    TW_Adj_mat[int(j+i)][int(j+i+1)]=K_b[i][i+1]
                    TW_Adj_mat[int(j+i)][int(j+i-1)]=K_b[i][i-1]
    
    #(5) Construct the K matrix
    K_mat=np.zeros(((N_b*N_s)+2,(N_b*N_s)+2))
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            if i!=j: #off-diagonal elements first
                K_mat[i][j]=TW_Adj_mat[j][i]
    
    #diagonal elements
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            if i!=j:
                K_mat[i][i]=K_mat[i][i]-K_mat[j][i]
            
    #dissiaption loss
    K_mat[0][0]=K_mat[0][0]-k_con
    for i in range(2,N_b*N_s+2):
        K_mat[i][i]=K_mat[i][i]-k_diss
                   
        
    #(6) The vector of photon inputs
    gamma_vec=np.zeros(N_b*N_s+2)
    for i in range(N_s): #exclude the first subunit which has bee accounted for above
        for j in start_index:
            gamma_vec[int(i+j)]=-gamma_b[i]
        
    #(7) Solve the kinetics
    K_inv=np.linalg.inv(K_mat)
    N_eq=np.zeros((N_b*N_s+2))
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            N_eq[i]=N_eq[i]+K_inv[i][j]*gamma_vec[j]

    #(8) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((N_b*N_s+2,N_b*N_s+2))
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            if K_mat[i][j]!=0.0:
                tau_mat[i][j]=(1.0/K_mat[i][j])/1.0E-12
            else: 
                tau_mat[i][j]=np.inf
    
    #(b) Electron output rate
    nu_e=k_con*N_eq[0]
    
    #(c) electron conversion quantum yield
    sum_rate=0.0
    for i in range(2,N_b*N_s+2):
        sum_rate=sum_rate+(k_diss*N_eq[i])
    
    phi_F=nu_e/(nu_e+sum_rate)
    
    if N_s>1:    
        out_dict={'TW_Adj_mat': TW_Adj_mat,
                 'K_b': K_b,
                 'K_mat': K_mat,
                 'tau_mat': tau_mat,
                 'gamma_b': gamma_b,
                 'gamma_vec': gamma_vec,
                 'N_eq': N_eq,
                 'nu_e': nu_e,
                 'phi_F': phi_F             
            }
    else:
        out_dict={'TW_Adj_mat': TW_Adj_mat,
                 'K_mat': K_mat,
                 'tau_mat': tau_mat,
                 'gamma_b': gamma_b,
                 'gamma_vec': gamma_vec,
                 'N_eq': N_eq,
                 'nu_e': nu_e,
                 'phi_F': phi_F             
            }
        
    return(out_dict)

'''***************************************************************************
Define transfer matrix, K, for a hexagonal antenna-RC configuration.
We assume that the RC sits at the central site and antenna grows  

Inputs: G_params = (N_LHC,layer)
            layer = (layer1,layer2,...,layerN) #absorption properties of the
                                               #concentric antenna layers
                                               #layer1 is the closest to the RC
            layeri = (ni,sigi,lpi,wi)
            
        RC_params = (n_RC, sig_RC, lp_RC, w_RC)


        k_params = (k_diss,k_hop,k_trap,k_con,K_LHC_RC) 

        T = temperature in K
        
***************************************************************************'''

def Antenna_hex_funnel(l,Ip_y,G_params,RC_params,k_params,T):
    
    #(1) Antenna parameters
    N_LHC=G_params[0]
    layer=G_params[1]

    #(2) Absorption rates
    fp_y=np.zeros(len(Ip_y)) #photon flux
    for i, item in enumerate(Ip_y):
        fp_y[i]=item*((l[i]*1.0E-9)/(h*c)) #factor of 1E-9 since l is in nm.

    gamma_layer=[]
    for layeri in layer:
        lineshape_i=gauss_abs(l,layeri[2],layeri[3])
        gamma_layer.append(layeri[1]*Olap_Int(l,fp_y,lineshape_i))

    #(3) RC_params
    N_RC, sigRC, lpRC, wRC=RC_params[0], RC_params[1], RC_params[2], RC_params[3]
         
    #(4) Rate constants
    k_diss, k_hop=k_params[0], k_params[1] #dissipation and transfer in the antenna
    k_trap, k_con=k_params[2], k_params[3] #irreversible excitation trapping and converion to electrons
    K_LHC_RC=k_params[4]
    
    #transfer between the RC and the first antenna layer
    nLHCnRC=layer[0][0]/N_RC
    thermoLHC_RC=deltaG(layer[0][2],lpRC,nLHCnRC,T) #thermodynamic parameters for G2 -> GRC
    G_LHC_RC,G_RC_LHC=thermoLHC_RC[2][0],thermoLHC_RC[2][1]
    
    if G_LHC_RC==0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC
    elif G_LHC_RC<0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC, K_LHC_RC*np.exp(-G_RC_LHC/(kB*T))
    elif G_LHC_RC>0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*np.exp(-G_LHC_RC/(kB*T)), K_LHC_RC    
    
    #Define an antenna hopping rate matrix
    #diagonal elements are intra-layer hopping rates
    #off-diagonal elements are interlayer transfer rates 
    k_antenna=np.zeros((len(layer),len(layer)))
    for i in range(len(layer)):
        for j in range(len(layer)):
            if i==j:
                k_antenna[i][j]=k_hop
            else:
                ni_nj=layer[i][0]/layer[j][0]
                thermo_ij=deltaG(layer[i][2],layer[j][2],ni_nj,T)
                G_ij, G_ji=thermo_ij[2][0], thermo_ij[2][1]
                
                if G_ij==G_ji:
                    k_antenna[i][j], k_antenna[j][i]=k_hop, k_hop
                elif G_ij<0.0: 
                    k_antenna[i][j], k_antenna[j][i]=k_hop, k_hop*np.exp(-G_ji/(kB*T))
                elif G_ij>0.0:
                    k_antenna[i][j], k_antenna[j][i]=k_hop*np.exp(-G_ij/(kB*T)), k_hop

    #(5) Build an adjacancy matrix
    #Site 0 is the trap state so canbe visualized as lying out of the plane of the lattice
    #connected only to site 1 which is the RC. 
    #Site 1 is the RC and the centre of the lattice. Always starting directly above the
    #site 1 we add sites in a clockwise direction filling tabulating which previously-placed
    #sites are neighbours of the new one. 
    #We therefore fill out the bottom triangle of the matrix before generating the upper triangle 
    #by transpose
    Adj_mat_upper=np.zeros((N_LHC+2,N_LHC+2)) #Number of LHCs plus an RC and RC trap
    Adj_mat_lower=np.zeros((N_LHC+2,N_LHC+2))
    
    #We first have to generate a list of the numbers that refer to sites belonging to 
    #the 'spokes' of the hexagon
    #These sequence are related to 'centred hexagonal numbers'
    top_spoke=[]
    top_right_spoke=[]
    bottom_right_spoke=[]
    bottom_spoke=[]
    bottom_left_spoke=[]
    top_left_spoke=[]
    for n in range(1,N_LHC):

        top_spoke.append(3*n*n-3*n+2)
        top_right_spoke.append(3*n*n-2*n+2)
        bottom_right_spoke.append(3*n*n-n+2)
        bottom_spoke.append(3*n*n+2)
        bottom_left_spoke.append(3*n*n+n+2)
        top_left_spoke.append(3*n*n+2*n+2)

    for i in range(N_LHC+2):
                    
        if i==1: #site 1 is the RC so the only previous site it is connected to is the trap
            Adj_mat_lower[i][0]=1.0

        elif i>1 and i<=7: #The first concentric layer surrounding the RC
            Adj_mat_lower[i][1]=1.0
            Adj_mat_lower[i][i-1]=1.0
            if i==7: #the last LHC in the first concentric layer
                Adj_mat_lower[i][2]=1.0

        elif i>7: #i.e. one of the outer layers
            if i in top_spoke:
                ind=top_spoke.index(i)
                Adj_mat_lower[i][top_spoke[ind-1]]=1.0
                
            elif i in top_right_spoke:
                ind=top_right_spoke.index(i)
                Adj_mat_lower[i][top_right_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in bottom_right_spoke:
                ind=bottom_right_spoke.index(i)
                Adj_mat_lower[i][bottom_right_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in bottom_spoke:
                ind=bottom_spoke.index(i)
                Adj_mat_lower[i][bottom_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in bottom_left_spoke:
                ind=bottom_left_spoke.index(i)
                Adj_mat_lower[i][bottom_left_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            elif i in top_left_spoke:
                ind=top_left_spoke.index(i)
                Adj_mat_lower[i][top_left_spoke[ind-1]]=1.0
                Adj_mat_lower[i][i-1]=1.0
                
            else:
                #For the rest we need to work out which concentric layer and which 
                #edge of the hexagon i refers to. 
                #First work out the layer using the top_spoke list
                layer=0
                for n, Hn in enumerate(top_spoke):
                    if i>=Hn:
                        layer=layer+1
                    else:
                        break
        
                #Next we have to work out which triangular 'wedge' site i belongs to
                wedge=0 #1=upper right, 2=right, 3=lower right,4=lower left, 5=left, 6=upper left
                
                #a list of the spoke sites in this layer starting 
                spoke_layer=[top_spoke[layer-1],top_right_spoke[layer-1],bottom_right_spoke[layer-1],
                             bottom_spoke[layer-1],bottom_left_spoke[layer-1],top_left_spoke[layer-1]] 
                for n, Hn in enumerate(spoke_layer):
                    if i>Hn:
                        wedge=wedge+1
                    else:
                        break            

                #Adding the neighbouring sites
                if wedge<=5: #have to treat the last wedge separately due to the 
                             #discontinuity in numbering
                    Adj_mat_lower[i][i-((6*(layer-1))+wedge)]=1.0
                    Adj_mat_lower[i][i-((6*(layer-1))+wedge-1)]=1.0
                    Adj_mat_lower[i][i-1]=1.0
                
                else:
                    Adj_mat_lower[i][i-((6*(layer-1))+wedge)]=1.0
                    Adj_mat_lower[i][i-1]=1.0
                    if i+1 in top_spoke:
                        Adj_mat_lower[i][i-(6*layer)+1]=1.0
                        Adj_mat_lower[i][i-(6*layer+6*(layer-1)-1)]=1.0
                    else:
                        Adj_mat_lower[i][i-((6*(layer-1))+wedge-1)]=1.0


    #Finally, having calculated thelower triangle we can calculated the upper
    #triangle via matrix transpose
    Adj_mat_upper=np.transpose(Adj_mat_lower)
    
    Adj_mat=np.zeros((N_LHC+2,N_LHC+2))
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            Adj_mat[i][j]=Adj_mat_lower[i][j]+Adj_mat_upper[i][j]

    #(6) Build the 'K' matrix: The dimensions include the trap, the RC, all LHCII
    # and a localized Chl b domain inside each LHCII
    
    K_mat=np.zeros(((N_LHC+2,N_LHC+2)))        
    Hex_list=Hex_n(N_LHC) #list of hexagonal numbers for idenifying layers

    #Fill the upper, inter_LHC block first
    #Define off-diagonal elements first. 
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            if i!=j: 
                #first work out which layer the site i is in
                layer_cnt_i=0
                for hex_num in Hex_list:
                    if i-1>=hex_num:
                        layer_cnt_i=layer_cnt_i+1
                    else:
                        break
                
                #thenwork out which layer site j is in
                layer_cnt_j=0
                for hex_num in Hex_list:
                    if j-1>=hex_num:
                        layer_cnt_j=layer_cnt_j+1
                    else:
                        break
 
                if i==0: #transfer to the trap is unidirectional from the RC 
                    K_mat[i][j]=Adj_mat[j][i]*k_trap

                elif i==1: #transfer to the RC
                    if j!=0: #exclude transfer back from the trap
                        K_mat[i][j]=Adj_mat[j][i]*k_LHC_RC

                elif i>=2 and i<=7: #i.e. in first concentric ring around the RC
                    if j==1: #transfer from the RC
                        K_mat[i][j]=Adj_mat[j][i]*k_RC_LHC
                    elif j>1: #transfer from another LHC
                        if layer_cnt_j==layer_cnt_i: #in the same layer
                            K_mat[i][j]=Adj_mat[j][i]*k_antenna[layer_cnt_i-1][layer_cnt_i-1]

                        else: #transfer from another layer
                            K_mat[i][j]=Adj_mat[j][i]*k_antenna[layer_cnt_j-1][layer_cnt_i-1]

                else: #in the outer layers of the antenna
                    if layer_cnt_j==layer_cnt_i: #in the same layer
                        K_mat[i][j]=Adj_mat[j][i]*k_antenna[layer_cnt_i-1][layer_cnt_i-1]
                    else: #in a different layer
                        K_mat[i][j]=Adj_mat[j][i]*k_antenna[layer_cnt_j-1][layer_cnt_i-1]

    #Diagonal elements
    K_mat[0][0]=-k_con
    for i in range(1,N_LHC+2):
        for j in range(N_LHC+2):
            if i!=j:
                K_mat[i][i]=K_mat[i][i]-K_mat[j][i]

        if i>=2: 
            K_mat[i][i]=K_mat[i][i]-k_diss #dissipation loss to the antenna complexes

    #(7) Generating terms
    gamma_vec=np.zeros((N_LHC+2))
    gamma_total=0.0
    for i in range(2,N_LHC+2):
        #which layer is the antenna subunit in?
        layer_cnt_i=0
        for hex_num in Hex_list:
            if i-1>=hex_num:
                layer_cnt_i=layer_cnt_i+1
            else:
                break

        gamma_vec[i]=-gamma_layer[layer_cnt_i-1]
        gamma_total=gamma_total-gamma_vec[i]        

    #(8) Solve the kinetics
    K_inv=np.linalg.inv(K_mat)
    N_eq=np.zeros((N_LHC+2))
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            N_eq[i]=N_eq[i]+K_inv[i][j]*gamma_vec[j]

    #(9) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((N_LHC+2,N_LHC+2))
    for i in range(N_LHC+2):
        for j in range(N_LHC+2):
            if K_mat[i][j]!=0.0:
                tau_mat[i][j]=(1.0/K_mat[i][j])/1.0E-12
            else: 
                tau_mat[i][j]=np.inf

    #(b) Electron output rate
    nu_e=k_con*N_eq[0]

    #(c) Electron conversion quantum yield
    sum_rate=0.0
    for i in range(2,N_LHC+2):
        sum_rate=sum_rate+(k_diss*N_eq[i])
    
    phi_F=nu_e/(nu_e+sum_rate)

    output_dict={
        'Adj_mat': Adj_mat,
        'K_mat': K_mat,
        'tau_mat': tau_mat,
        'N_eq': N_eq,
        'nu_e': nu_e,
        'phi_F': phi_F,
        'gamma_layer': gamma_layer,
        'gamma_total': gamma_total,
        'gamma_vec': gamma_vec,
        }
                               
            
    return(output_dict)




'''***************************************************************************
Solve the steady state for a phycobilisome-like antenna.
A series of 'branches' radiate out from a central RC. The subunits contain different 
pools of pigments, with absorption characteristics defined by the user. This 
is an update of the Antenna_branched_funnel function that accounts for spectral 
overlap in the inter-subunit hopping rate.

The overlap integral is used to calculate 'deviation from perfect overlap'.If 
we consider two identical normalized Gaussian functions then the integrated 
overlap is 1/sqrt(4*pi*w^2). We therefore need to multiply the overlap integral
by some value to (1) make it dimensionless and (2) scale it so the 'perfect overlap'
gives 1. Therefore, we divide the overlap of two Guassians of different mean and 
width by the overlap of two identical Gaussians with wdith equal to the mean width 
of the original 2. 

Inputs: l = array of wavelengths (nm)
        Ip_y = spectral flux values for wavelengths in l
        
        Branch_params=(N_b,branch)
            N_b = the number of antenna branches
            branch = (sN, sN-1, ..., s2, s1) where s1 is the furthest most 
                                             subunit from the RC
                si = (Ni, sig_i, lpi, wi) 
        
        RC_params = (N_RC, sig_RC, lp_RC, w_RC)
 
        k_params = (k_diss,k_trap,k_con,K_hop,K_LHC_RC) are the rate constants for the 
                    transfer/decay pathways (s^{-1})
        T = temperature in K

***************************************************************************'''

def Antenna_branched_overlap(l,Ip_y,Branch_params,RC_params,k_params,T):
    
    #(1) Unpack parameters and establish dimensionality of the antenna
    k_diss, k_trap, k_con, K_hop, K_LHC_RC=k_params
    N_RC, sig_RC, lp_RC, w_RC=RC_params
    
    N_b=Branch_params[0] #number of branches
    N_s=len(Branch_params)-1 #number of subunits per branch

    subunits=[]
    for i in range(1,N_s+1):
        subunits.append(Branch_params[i])
    
    #(2) Absorption rates
    fp_y=np.zeros(len(Ip_y)) #photon flux
    for i, item in enumerate(Ip_y):
        fp_y[i]=item*((l[i]*1.0E-9)/(h*c)) #factor of 1E-9 since l is in nm.
                 
    gamma_b=[]
    lineshape_b=[] #now collect the lineshapes for later use in the overlap calculation
    for i in range(N_s):
        print(i,N_s)
        lineshape=gauss_abs(l,subunits[i][2],subunits[i][3])
        lineshape_b.append(lineshape)
        gamma_i=subunits[i][0]*subunits[i][1]*Olap_Int(l,fp_y,lineshape)
        gamma_b.append(gamma_i)
    
    #(3) Calculate rate constants   
    #(a) Transfer between RC and an antenna branch
    #First calculate the spectral overlap
    gauss_RC=gauss_abs(l,lp_RC,w_RC)
    DE_LHC_RC=Olap_Int(l,gauss_RC,lineshape_b[0])
    
    #rescale this overlap
    mean_w=(w_RC+subunits[0][3])/2.0
    DE_LHC_RC=DE_LHC_RC*np.sqrt(4*np.pi*mean_w)
    
    #thermodynamic factors
    nRCnLHC=subunits[0][0]/N_RC
    thermoRC=deltaG(subunits[0][2],lp_RC,nRCnLHC,T) #thermodynamic parameters for G2 -> GRC
    G_LHC_RC,G_RC_LHC=thermoRC[2][0],thermoRC[2][1]
    
    if G_LHC_RC==0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*DE_LHC_RC, K_LHC_RC*DE_LHC_RC
    elif G_LHC_RC<0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*DE_LHC_RC, K_LHC_RC*np.exp(-G_RC_LHC/(kB*T))*DE_LHC_RC
    elif G_LHC_RC>0.0:
        k_LHC_RC, k_RC_LHC=K_LHC_RC*np.exp(-G_LHC_RC/(kB*T))*DE_LHC_RC, K_LHC_RC*DE_LHC_RC        
        
    #transfer between subunits
    if N_s>1:
        K_b=np.zeros((N_s,N_s))
        for i in range(N_s-1): #working from the inner subunit to the outer one

            #spectral overlap
            DE=Olap_Int(l,lineshape_b[i],lineshape_b[i+1])

            #rescale this overlap
            mean_w=(subunits[i][3]+subunits[i+1][3])/2.0
            DE=DE*np.sqrt(4*np.pi*mean_w)

            #thermodynamic factors
            n_ratio=subunits[i][0]/subunits[i+1][0]
            thermo=deltaG(subunits[i][2],subunits[i+1][2],n_ratio,T)
            G_out_in, G_in_out=thermo[2][0], thermo[2][1]
        
            if G_out_in==0.0: #i.e. the subunits are identical
                K_b[i][i+1], K_b[i+1][i]=K_hop*DE, K_hop*DE #forward rate        
            elif G_out_in<0.0: #i.e. forward transfer is favoured
                K_b[i][i+1], K_b[i+1][i]=K_hop*DE, K_hop*np.exp(-G_in_out/(kB*T))*DE
            elif G_out_in>0.0: #i.e. forward transfer is limited
                K_b[i][i+1], K_b[i+1][i]=K_hop*np.exp(-G_out_in/(kB*T))*DE, K_hop*DE
            
    #(4) Assemble the Transfer Weighted Adjacency matrix 
    #the numbering of the sub-units has changed relative to the initial branched
    #antnne. Site 0 is the trap, 1 is the RC. 
    #2,3,...,N_s+1 is the first branch
    #N_s+2, N_s+3, ..., 2N_s+1 is the second branch 
    #2N_s+2, 2N_s+3, ... 3Ns+1 is the third branch
    #and so on. 
    
    TW_Adj_mat=np.zeros(((N_b*N_s)+2,(N_b*N_s)+2))

    start_index=np.zeros((N_b)) #list of indices of the starting (RC-adjacent) 
    for i in range(N_b):        #antenna sites of the branches
        start_index[i]=i*N_s+2
           
    end_index=np.zeros((N_b)) #ditto for the ends of the branches
    for i in range(N_b):
        end_index[i]=(i+1)*N_s+1
        
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            #we ignore i==0 as although the trap is connected to the RC, trapping
            #is irreversibile (the rate of de-trapping is 0)
            if i==1: #i.e. transfer from the RC
                if j==0: #to the trap
                    TW_Adj_mat[i][j]=k_trap 
                elif j in start_index: #to the LHCs at the start of the branches
                    TW_Adj_mat[i][j]=k_RC_LHC

            elif i in start_index: #transfer from the inner ring of LHCs
                if j==1: #to the RC 
                    TW_Adj_mat[i][j]=k_LHC_RC
                elif j==i+1: #to the next subunit along the chain
                    if N_s>1:
                        TW_Adj_mat[i][j]=K_b[0][1] 
                    
    #now fill in the nearest-neighbour adjacencies along the branches
    if N_s>1:
        for i in range(1,N_s): #exclude the first subunit which has bee accounted for above
            for j in start_index:
                if i+j in end_index:                
                    TW_Adj_mat[int(j+i)][int(j+i-1)]=K_b[i][i-1]
                else:
                    TW_Adj_mat[int(j+i)][int(j+i+1)]=K_b[i][i+1]
                    TW_Adj_mat[int(j+i)][int(j+i-1)]=K_b[i][i-1]
    
    #(5) Construct the K matrix
    K_mat=np.zeros(((N_b*N_s)+2,(N_b*N_s)+2))
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            if i!=j: #off-diagonal elements first
                K_mat[i][j]=TW_Adj_mat[j][i]
    
    #diagonal elements
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            if i!=j:
                K_mat[i][i]=K_mat[i][i]-K_mat[j][i]
            
    #dissiaption loss
    K_mat[0][0]=K_mat[0][0]-k_con
    for i in range(2,N_b*N_s+2):
        K_mat[i][i]=K_mat[i][i]-k_diss
                   
        
    #(6) The vector of photon inputs
    gamma_vec=np.zeros(N_b*N_s+2)
    for i in range(N_s): #exclude the first subunit which has bee accounted for above
        for j in start_index:
            gamma_vec[int(i+j)]=-gamma_b[i]
        
    #(7) Solve the kinetics
    K_inv=np.linalg.inv(K_mat)
    N_eq=np.zeros((N_b*N_s+2))
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            N_eq[i]=N_eq[i]+K_inv[i][j]*gamma_vec[j]

    #(8) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((N_b*N_s+2,N_b*N_s+2))
    for i in range(N_b*N_s+2):
        for j in range(N_b*N_s+2):
            if K_mat[i][j]>=1.0E-12:
                tau_mat[i][j]=(1.0/K_mat[i][j])/1.0E-12
            else: 
                tau_mat[i][j]=np.inf
    
    #(b) Electron output rate
    nu_e=k_con*N_eq[0]
    
    #(c) electron conversion quantum yield
    sum_rate=0.0
    for i in range(2,N_b*N_s+2):
        sum_rate=sum_rate+(k_diss*N_eq[i])
    
    phi_F=nu_e/(nu_e+sum_rate)
    
    if N_s>1:    
        out_dict={'TW_Adj_mat': TW_Adj_mat,
                 'K_b': K_b,
                 'K_mat': K_mat,
                 'tau_mat': tau_mat,
                 'gamma_b': gamma_b,
                 'gamma_vec': gamma_vec,
                 'N_eq': N_eq,
                 'nu_e': nu_e,
                 'phi_F': phi_F             
            }
    else:
        out_dict={'TW_Adj_mat': TW_Adj_mat,
                 'K_mat': K_mat,
                 'tau_mat': tau_mat,
                 'gamma_b': gamma_b,
                 'gamma_vec': gamma_vec,
                 'N_eq': N_eq,
                 'nu_e': nu_e,
                 'phi_F': phi_F             
            }
        
    return(out_dict)





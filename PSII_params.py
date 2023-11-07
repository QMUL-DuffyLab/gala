# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:00:02 2023

Optimum parameters for PSII-LHCII in a 5800K spectrum

@author: btw430
"""

'''***************************************************************************
* Spectral parameters                                                      '''

sig=1.9E-18 #optical cross-section of the antenna (m^2)
B12=0.56 #ratio of Chlb/Chb peak amplitude 
lp1=650.0#Chlb 650 nm absorption peak (nm)
w1=8.5 #Chlb peak width (nm)
lp2=675.0#Chlb 675 nm absorption peak
w2=9.0 #Chlb width (nm)
lpRC=680.0 #reaction centre
wRC=w2 #width is the same as Chla

G_params=[sig,B12,lp1,w1,lp2,w2,lpRC,wRC] #tuple of these parameters to send to the antenna model
'''*************************************************************************'''

'''***************************************************************************
* Dimensions parameters                                                     '''

N_LHC=5.0 #The number of antenna complexes per reaction centre (does not have 
          #to be an integer)
N1=18.0 #number of accessory pigments in a single LHC
N2=24.0 #number of primary pigmets in a single LHCII

Size_params=[N1,N2,N_LHC] #list of these parameters to send to the antenna model
'''*************************************************************************'''

'''***************************************************************************
* Rate Parameters                                                          '''

k_diss=1.0/4.0E-9 #Chl excited state decay rate 
k_trap=1.0/5.0E-12 #PSII trapping rate
k_con=1.0/10.0E-3 #PSII RC turnover rate
K_12=1.0/1.0E-12 #fundamental Chlb-Chla hopping rate
K_2RC=1.0/5.0E-12 #fundanental LHCII-RC hopping rate


k_params=[k_diss,k_trap,k_con,K_12,K_2RC]

'''*************************************************************************'''

T=300.0
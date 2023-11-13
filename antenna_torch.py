import torch
import numpy as np
import constants as constants
from scipy.constants import h as h
from scipy.constants import c as c
from scipy.constants import Boltzmann as kB

def absorption(l, lambda_peak, width):
    g = torch.exp(-(l - lambda_peak)**2/(2.0 * width**2))
    n = torch.trapezoid(g, l)
    return torch.div(g/n)

def overlap(l, f1, f2):
    return torch.trapezoid(torch.mul(f1, f2), l)

def antenna_torch(l,ip_y,Branch_params,RC_params,k_params,T):

    #(1) Unpack parameters and establish dimensionality of the antenna
    k_diss, k_trap, k_con, K_hop, K_LHC_RC = k_params
    N_RC, sig_RC, lp_RC, w_RC = RC_params

    n_b = Branch_params[0] #number of branches
    n_s = len(Branch_params)-1 #number of subunits per branch

    subunits=[]
    for i in range(1, n_s + 1):
        subunits.append(Branch_params[i])

    '''
    2. Absorption rates
    '''
    fp_y = torch.mul(ip_y, l * (1.0E-9/(h*c)))

    gammas = torch.zeros(n_s)
    lineshapes = torch.zeros(n_s)
    for i in range(n_s):
        lineshapes[i] = absorption(l, subunits[i][2], subunits[i][3])
        gammas[i]     = (subunits[i][0]
                         * constants.sig_chl
                         * overlap(l, fp_y, lineshapes[i]))

    '''
    3. Calculate rate constants
    a. Transfer between RC and a branch
    '''
    # First calculate the spectral overlap
    gauss_RC = absorption(l, lp_RC, w_RC)
    DE_LHC_RC = overlap(l, gauss_RC, lineshapes[0])

    # rescale this overlap
    mean_w = (w_RC + subunits[0][3])/2.0
    DE_LHC_RC = DE_LHC_RC * np.sqrt(4 * np.pi * mean_w)

    # thermodynamic factors
    nRCnLHC = subunits[0][0]/N_RC
    # thermodynamic parameters for G2 -> GRC
    thermoRC = deltaG(subunits[0][2], lp_RC, nRCnLHC, T)
    G_LHC_RC, G_RC_LHC = thermoRC[2][0], thermoRC[2][1]

    rate = K_LHC_RC * DE_LHC_RC
    if G_LHC_RC==0.0:
        k_LHC_RC, k_RC_LHC = rate, rate
    elif G_LHC_RC<0.0:
        k_LHC_RC, k_RC_LHC = rate, rate * np.exp(-G_RC_LHC/(kB*T))
    elif G_LHC_RC>0.0:
        k_LHC_RC, k_RC_LHC = rate * np.exp(-G_LHC_RC/(kB*T)), rate

    # transfer between subunits
    if n_s > 1:
        K_b = np.zeros((n_s, n_s))
        for i in range(n_s - 1): # working from inner to outer

            # spectral overlap
            DE = overlap(l, lineshapes[i], lineshapes[i+1])

            # rescale this overlap
            mean_w = (subunits[i][3] + subunits[i+1][3]) / 2.0
            DE = DE * np.sqrt(4 * np.pi * mean_w)

            # thermodynamic factors
            n_ratio = subunits[i][0] / subunits[i+1][0]
            thermo = deltaG(subunits[i][2], subunits[i+1][2], n_ratio, T)
            G_out_in, G_in_out = thermo[2][0], thermo[2][1]

            rate = K_hop * DE
            if G_out_in == 0.0:  # i.e. the subunits are identical
                K_b[i][i+1], K_b[i+1][i] = rate, rate #forward rate
            elif G_out_in < 0.0: # i.e. forward transfer is favoured
                K_b[i][i+1] = rate
                K_b[i+1][i] = rate * np.exp(-G_in_out / (kB * T))
            elif G_out_in > 0.0: # i.e. forward transfer is limited
                K_b[i][i+1] = rate * np.exp(-G_out_in / (kB * T))
                K_b[i+1][i] = rate

    #(4) Assemble the Transfer Weighted Adjacency matrix
    #the numbering of the sub-units has changed relative to the initial branched
    #antnne. Site 0 is the trap, 1 is the RC.
    #2,3,...,N_s+1 is the first branch
    #N_s+2, n_s+3, ..., 2N_s+1 is the second branch
    #2N_s+2, 2N_s+3, ... 3Ns+1 is the third branch
    #and so on.

    TW_Adj_mat = np.zeros((n_b * n_s + 2, n_b * n_s + 2))

    start_index = np.zeros((n_b)) # indices of the starting (RC-adjacent)
    for i in range(n_b):          # antenna sites of the branches
        start_index[i] = i * n_s + 2

    end_index=np.zeros((n_b)) # ditto for the ends of the branches
    for i in range(n_b):
        end_index[i] = (i + 1) * n_s + 1

    for i in range(n_b * n_s + 2):
        for j in range(n_b * n_s + 2):
            #we ignore i==0 as although the trap is connected to the RC, trapping
            #is irreversibile (the rate of de-trapping is 0)
            if i == 1: #i.e. transfer from the RC
                if j == 0: #to the trap
                    TW_Adj_mat[i][j] = k_trap
                elif j in start_index: #to the LHCs at the start of the branches
                    TW_Adj_mat[i][j] = k_RC_LHC

            elif i in start_index: #transfer from the inner ring of LHCs
                if j == 1: #to the RC
                    TW_Adj_mat[i][j] = k_LHC_RC
                elif j == i + 1: #to the next subunit along the chain
                    if n_s > 1:
                        TW_Adj_mat[i][j] = K_b[0][1]

    #now fill in the nearest-neighbour adjacencies along the branches
    if n_s > 1:
        for i in range(1, n_s): #exclude the first subunit which has bee accounted for above
            for j in start_index:
                if i+j in end_index:
                    TW_Adj_mat[int(j+i)][int(j+i-1)] = K_b[i][i-1]
                else:
                    TW_Adj_mat[int(j+i)][int(j+i+1)] = K_b[i][i+1]
                    TW_Adj_mat[int(j+i)][int(j+i-1)] = K_b[i][i-1]

    #(5) Construct the K matrix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K_mat = torch.zeros([n_b * n_s + 2,n_b * n_s + 2],
                        device=device, dtype=torch.float64)
    # K_mat=np.zeros(((n_b*N_s)+2,(n_b*N_s)+2))
    for i in range(n_b * n_s + 2):
        for j in range(n_b * n_s + 2):
            if i != j: #off-diagonal elements first
                K_mat[i][j] = TW_Adj_mat[j][i]

    #diagonal elements
    for i in range(n_b * n_s + 2):
        for j in range(n_b * n_s + 2):
            if i != j:
                K_mat[i][i]=K_mat[i][i]-K_mat[j][i]

    #dissiaption loss
    K_mat[0][0] = K_mat[0][0] - k_con
    for i in range(2, n_b * n_s + 2):
        K_mat[i][i] = K_mat[i][i] - k_diss


    #(6) The vector of photon inputs
    gamma_vec=np.zeros(n_b * n_s + 2)
    for i in range(n_s): #exclude the first subunit which has bee accounted for above
        for j in start_index:
            gamma_vec[int(i+j)] = -gammas[i]

    #(7) Solve the kinetics
    # K_inv=np.linalg.inv(K_mat)
    # N_eq=np.zeros((n_b*N_s+2))
    # for i in range(n_b*N_s+2):
    #     for j in range(n_b*N_s+2):
    #         N_eq[i]=N_eq[i]+K_inv[i][j]*gamma_vec[j]
    gvt = torch.tensor(gamma_vec, device=device, dtype=torch.float64)
    n_eq = torch.linalg.solve(K_mat, gvt)

    #(8) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((n_b * n_s + 2, n_b * n_s + 2))
    for i in range(n_b * n_s + 2):
        for j in range(n_b * n_s + 2):
            if K_mat[i][j] >= 1.0E-12:
                tau_mat[i][j] = (1.0 / K_mat[i][j]) / 1.0E-12
            else:
                tau_mat[i][j] = np.inf

    #(b) Electron output rate
    nu_e=k_con*N_eq[0]

    #(c) electron conversion quantum yield
    sum_rate=0.0
    for i in range(2,n_b*N_s+2):
        sum_rate=sum_rate+(k_diss*N_eq[i])

    phi_F=nu_e/(nu_e+sum_rate)

    if n_s > 1:
        out_dict={'TW_Adj_mat': TW_Adj_mat,
                 'K_b': K_b,
                 'K_mat': K_mat,
                 'tau_mat': tau_mat,
                 'gammas': gammas,
                 'gamma_vec': gamma_vec,
                 'N_eq': N_eq,
                 'nu_e': nu_e,
                 'phi_F': phi_F
            }
    else:
        out_dict={'TW_Adj_mat': TW_Adj_mat,
                 'K_mat': K_mat,
                 'tau_mat': tau_mat,
                 'gammas': gammas,
                 'gamma_vec': gamma_vec,
                 'N_eq': N_eq,
                 'nu_e': nu_e,
                 'phi_F': phi_F
            }

    return(out_dict)

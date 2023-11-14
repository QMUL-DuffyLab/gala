import torch
import numpy as np
import constants as constants
from scipy.constants import h as h
from scipy.constants import c as c
from scipy.constants import Boltzmann as kB

def absorption(l, lambda_peak, width):
    g = torch.exp(-(l - lambda_peak)**2/(2.0 * width**2))
    n = torch.trapezoid(g, l)
    return torch.div(g, n)

def overlap(l, f1, f2):
    '''
    trapezoid rule for the product of f1 and f2 over l.
    note that you have to return this wrapped in float,
    otherwise torch returns it as a tensor, and then later
    on it makes two shallow copies of the tensor for the
    forward and backward rates. alternatively you can do
    k_LHC_RC = rate
    k_RC_LHC = rate.detach().clone()
    later on, but we don't need this as a tensor anyway.
    '''
    return float(torch.trapezoid(torch.mul(f1, f2), l))

def dG(l1, l2, n, T):
    # symmetric - going from 2 -> 1 is just minus this
    h12 = (h * c / 1.0E-9) * (l1 - l2) / (l1 * l2)
    s12 = - kB * np.log(n)
    g12 = h12 - s12 * T
    return g12

def antenna(l,ip_y,Branch_params,RC_params,k_params,T):

    # probably need to copy l and ip_y to the GPU for this
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #(1) Unpack parameters and establish dimensionality of the antenna
    k_diss, k_trap, k_con, K_hop, K_LHC_RC = k_params
    N_RC, sig_RC, lp_RC, w_RC = RC_params
    lt = torch.tensor(l)
    ipyt = torch.tensor(ip_y)

    n_b = Branch_params[0] #number of branches
    subunits = Branch_params[1:]
    n_s = len(subunits)

    # transfer-weighted adjacency matrix.
    # fill in as we go to cut down on the number of loops
    side = n_b * n_s + 2
    TW_Adj_mat = torch.zeros((side, side))
    # likewise with the vector of photon inputs
    gamma_vec = torch.zeros(side, dtype=torch.float32)

    # Site 0 is the trap, 1 is the RC.
    # 2, ..., n_s + 1 is the first branch
    # n_s + 2, ..., 2n_s + 1 is the second branch
    # 2n_s + 2, ..., 3n_s + 1 is the third branch
    # is there a way of striding this to avoid having to do all the indexing?
    # can't think of it yet. so get root and tip indices of each branch
    start_index = torch.zeros(n_b, dtype=torch.int)
    end_index   = torch.zeros(n_b, dtype=torch.int)
    for i in range(n_b):
        start_index[i] = i * n_s + 2
        end_index[i]   = (i + 1) * n_s + 1

    '''
    2. Absorption rates
    '''
    fp_y = torch.mul(ipyt, lt * (1.0E-9/(h*c)))

    gammas = torch.zeros(n_s)
    lineshapes = torch.zeros((n_s, lt.size()[0]))
    for i in range(n_s):
        lineshapes[i] = absorption(lt, subunits[i][2], subunits[i][3])
        gammas[i]     = (subunits[i][0]
                         * constants.sig_chl
                         * overlap(lt, fp_y, lineshapes[i]))
        for j in start_index:
            gamma_vec[i + j] = -gammas[i]

    '''
    3. Calculate rate constants
    a. Transfer between RC and a branch
    '''
    # First calculate the spectral overlap
    gauss_RC  = absorption(lt, lp_RC, w_RC)
    DE_LHC_RC = overlap(lt, gauss_RC, lineshapes[0])

    # rescale this overlap
    mean_w    = (w_RC + subunits[0][3]) / 2.0
    DE_LHC_RC = DE_LHC_RC * np.sqrt(4 * np.pi * mean_w)

    # thermodynamic factors
    nRCnLHC = subunits[0][0] / N_RC
    # thermodynamic parameters for G2 -> GRC
    G_LHC_RC = dG(subunits[0][2], lp_RC, nRCnLHC, T)

    rate = K_LHC_RC * DE_LHC_RC
    k_LHC_RC = rate
    k_RC_LHC = rate
    if G_LHC_RC < 0.0:
        k_RC_LHC *= np.exp(G_LHC_RC/(kB*T))
    elif G_LHC_RC > 0.0:
        k_LHC_RC *= np.exp(-G_LHC_RC/(kB*T))

    # transfer between subunits
    if n_s > 1:
        K_b = torch.zeros((n_s, n_s))
        for i in range(n_s - 1): # working from inner to outer

            # spectral overlap
            DE = overlap(lt, lineshapes[i], lineshapes[i+1])

            # rescale this overlap
            mean_w = (subunits[i][3] + subunits[i + 1][3]) / 2.0
            DE = DE * np.sqrt(4 * np.pi * mean_w)

            # thermodynamic factors
            n_ratio = subunits[i][0] / subunits[i + 1][0]
            G_out_in = dG(subunits[i][2], subunits[i + 1][2], n_ratio, T)

            rate = K_hop * DE
            K_b[i][i + 1], K_b[i + 1][i] = rate, rate #forward rate
            if G_out_in < 0.0: # i.e. forward transfer is favoured
                K_b[i + 1][i] *= np.exp(G_out_in / (kB * T))
            elif G_out_in > 0.0: # i.e. forward transfer is limited
                K_b[i][i + 1] *= np.exp(-G_out_in / (kB * T))

    # matrix elements involving the trap, RC and root of each branch
    # ignore i == 0 - trapping is irreversible
    TW_Adj_mat[1][0] = k_trap # RC -> trap
    for j in start_index:
        TW_Adj_mat[1][j] = k_RC_LHC # RC -> LHC (root)
        TW_Adj_mat[j][1] = k_LHC_RC # LHC (root) -> RC
        print(1, j, TW_Adj_mat[int(1)][j])
        print(j, 1, TW_Adj_mat[int(j)][1])
        if n_s > 1:
            TW_Adj_mat[j][j + 1] = K_b[0][1]
            print(j, i + j, j + 1, TW_Adj_mat[int(j)][int(j+1)])
            # nn adjacencies along the branches
            for i in range(1, n_s):
                # first subunit is accounted for above
                for j in start_index:
                    TW_Adj_mat[i + j][i + j - 1] = K_b[i][i - 1]
                    if (i + j) not in end_index:
                        TW_Adj_mat[i + j][i + j + 1] = K_b[i][i + 1]
        
    # construct the K matrix
    K_mat = torch.zeros([side,side], device=device, dtype=torch.float32)
    # conversion rate of trap
    K_mat[0][0] = K_mat[0][0] - k_con
    for i in range(side):
        if i >= 2:
            K_mat[i][i] = K_mat[i][i] - k_diss
        for j in range(side):
            if i != j:
                K_mat[i][j]  = TW_Adj_mat[j][i]
                K_mat[i][i] -= TW_Adj_mat[j][i]


    # solve the kinetics
    # gvt = gamma_vec.clone().detach()
    n_eq = torch.linalg.solve(K_mat, gamma_vec)

    #(8) Outputs
    #(a) A matrix of lifetimes (in ps) is easier to read than the rate constants
    tau_mat=np.zeros((side, side))
    for i in range(side):
        for j in range(side):
            if K_mat[i][j] >= 1.0E-12:
                tau_mat[i][j] = (1.0 / K_mat[i][j]) / 1.0E-12
            else:
                tau_mat[i][j] = np.inf

    #(b) Electron output rate
    nu_e = k_con * n_eq[0]

    #(c) electron conversion quantum yield
    sum_rate = 0.0
    for i in range(2, side):
        sum_rate = sum_rate + (k_diss * n_eq[i])

    phi_F = nu_e / (nu_e + sum_rate)

    out_dict={'TW_Adj_mat': TW_Adj_mat,
             'K_b': K_b,
             'K_mat': K_mat,
             'tau_mat': tau_mat,
             'gammas': gammas,
             'gamma_vec': gamma_vec,
             'N_eq': n_eq,
             'nu_e': nu_e,
             'phi_F': phi_F
        }
    return(out_dict)

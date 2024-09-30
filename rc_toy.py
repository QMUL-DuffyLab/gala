import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from matplotlib import cm, ticker, colors

# need reasonable numbers for the gammas
# gamma_ox = 1.0e6
gamma_E  = 1.0
k_diss   = 1.0 / 1.0e-9
k_trap   = 1.0 / 10.0e-12
k_o2     = 1.0 / 400.0e-6
k_lin    = 1.0 / 10.0e-3
k_out    = 1.0 / 10.0e-3 # check with chris
# alpha    = 1.0
# k_cyc    = alpha * k_lin

'''
look how clever i am
'''

one_rc = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
          (0, 0, 1), (1, 1, 0), (1, 0, 1)]
# combine one_rc with itself to make the possible states of the supersystem
two_rc = [s1 + s2 for s1 in one_rc for s2 in one_rc]
# assign each combination to an index for use in the array below
indices = {state: i for state, i in zip(two_rc, range(len(two_rc)))}
# assign the array
side = len(two_rc)
twa = np.zeros((side, side))
two_rc = np.array(two_rc)

lindices = []
cycdices = []
for si in two_rc:
    if si[4] == 1 and si[5] == 0:
        # [n_^{ox}, n^{E}_e, 1, 0]
        cycdices.append(indices[tuple(si)])
    if si[1] == 1 and si[2] == 0 and si[4] == 0 and si[5] == 1:
        # [n_^{ox}_e, 1, 0, n^{E}_e, 0, 1]
        lindices.append(indices[tuple(si)])

g_points = np.logspace(-8, 8)
a_points = np.logspace(-6, 6)
ratios = np.zeros((50, 50), dtype=np.float64)

for gi, gamma_ox in enumerate(g_points):
    for ai, alpha in enumerate(a_points):
        k_cyc = alpha * k_lin
        # print(f"gi = {gi}, ai = {ai}, gamma_ox = {gamma_ox}, alpha = {alpha}")
        # keys in the dict here are the population differences between
        # initial and final states, for each type of process; values are rates
        processes = {
                (1, 0, 0, 0, 0, 0): gamma_ox,
                (0, 0, 0, 1, 0, 0): gamma_E,
                (-1, 0, 0, 0, 0, 0): k_diss,
                (0, 0, 0, -1, 0, 0): k_diss,
                (-1, 1, 0, 0, 0, 0): k_trap,
                (0, 0, 0, -1, 1, 0): k_trap,
                (0, 0, -1, 0, 0, 0): k_o2,
                (0, -1, 1, 0, 0, -1): k_lin,
                (0, 0, 0, 0, -1, 1): k_out,
                (0, 0, 0, 0, -1, 0): k_cyc,
                }
        # loop over, check difference, assign rate if necessary. bish bash bosh
        # also add the relevant indices for linear and cyclic flow to lists
        for si in two_rc:
            for sf in two_rc:
                diff = tuple(sf - si)
                if diff in processes:
                    index = indices[tuple(si)], indices[tuple(sf)]
                    twa[index] = processes[diff]

        # set up nnls
        # in theory you should be able to construct this matrix as you go
        # but in practice i always fuck it up somehow. so just do it here
        k = np.zeros((side + 1, side))
        for i in range(side):
            for j in range(side):
                if (i != j):
                    k[i][j]  = twa[j][i]
                    k[i][i] -= twa[i][j]
            k[side][i] = 1.0

        b = np.zeros(side + 1)
        b[-1] = 1.0

        nu_lin = 0.0
        nu_cyc = 0.0
        # nnls
        try:
            p_eq, p_eq_res = nnls(k, b)
            for state, p in zip(two_rc, p_eq):
                if indices[tuple(state)] in lindices:
                    nu_lin += k_lin * p
                if indices[tuple(state)] in cycdices:
                    nu_cyc += k_cyc * p
                # if p > 0.0:
                #     print(f"p_eq({state}) = {p}")
            print(f"alpha = {alpha}, ratio = {nu_cyc / nu_lin}")
            ratios[gi, ai] = nu_cyc / nu_lin
        except RuntimeError:
            p_eq = None
            p_eq_res = None
            nu_cyc = np.nan
            nu_lin = np.nan

np.savetxt("out/rc_toy_nu_ratio.dat", ratios)
fig, ax = plt.subplots(figsize=(14,12))
xx, yy = np.meshgrid(a_points, g_points)
lev_exp = np.arange(np.floor(np.log10(ratios.min()) - 1),
                    np.ceil(np.log10(ratios.max()) + 1))
levs = np.power(10, lev_exp)
cs = ax.contourf(xx, yy, ratios, levs, norm=colors.LogNorm())
ax.grid(visible=False)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'$ \gamma_{ox}/\gamma_{E} $')
ax.set_xlabel(r'$ \alpha (k_{cyc} / k_{lin}) $ ')
fig.colorbar(cs, ticks=levs)
fig.tight_layout()
plt.savefig("out/rc_toy_nu_ratio.pdf")

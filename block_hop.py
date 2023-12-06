# -*- coding: utf-8 -*-
"""
05/12/2023
@author: callum

Calculate average transit time out of a block to a neighbour.
Block size is n x n x h, 3d random walk, run trials times per (n, h).
Closed boundaries in the n x n directions, open in the h direction.
Will be used to dress the bare hopping rate in antenna.c

"""

import numpy as np

dt = 1.0 # ps
tmax = 10000.0
trials = 10000
k_pp = 1.0 / (1.0 * dt) # pigment-pigment hopping
k_bb = 1.0 / (10.0 * k_pp) # block-block hopping
rng = np.random.default_rng()
neighbours = np.zeros((6, 4), dtype=int)
rates = np.zeros(6)
s = np.arange(2, 64, dtype=int)
avg_times = np.zeros((len(s), len(s)), dtype=np.float64)
avg_sq = np.zeros((len(s), len(s)), dtype=np.float64)

for i, n in enumerate(s):
    for j, h in enumerate(s):
        total_t = 0.0
        for trial in range(trials):
            loc = np.zeros(4, dtype=int) # loc[0] is a dummy index basically
            loc[1] = rng.integers(h)
            loc[2] = rng.integers(n)
            loc[3] = rng.integers(n)
            t = 0.0
            while loc[0] == 0 and t < tmax:
                neighbours[0] = np.array([loc[0], loc[1] - 1, loc[2], loc[3]])
                neighbours[1] = np.array([loc[0], loc[1] + 1, loc[2], loc[3]])
                neighbours[2] = np.array([loc[0], loc[1], loc[2] - 1, loc[3]])
                neighbours[3] = np.array([loc[0], loc[1], loc[2] + 1, loc[3]])
                neighbours[4] = np.array([loc[0], loc[1], loc[2], loc[3] - 1])
                neighbours[5] = np.array([loc[0], loc[1], loc[2], loc[3] + 1])
                for k in range(6):
                    rates[k] = k_pp
                    if neighbours[k][1] == h:
                        neighbours[k][0] = 1
                        neighbours[k][1] = 0
                        rates[k] = k_bb
                    if neighbours[k][1] == -1:
                        # neighbours[k][0] = 1
                        # neighbours[k][1] = n - 1
                        # rates[k] = k_bb
                        rates[k] = 0.0 # only one face is connected
                    if neighbours[k][2] == n:
                        rates[k] = 0.0
                    if neighbours[k][2] == -1:
                        rates[k] = 0.0
                    if neighbours[k][3] == n:
                        rates[k] = 0.0
                    if neighbours[k][3] == -1:
                        rates[k] = 0.0
                cr = np.cumsum(rates)
                u = rng.uniform() * cr[-1]
                l = 1
                while u > cr[l]:
                    l += 1
                new = neighbours[l]
                loc = new
                t += (1.0 / cr[-1]) * np.log(1.0 / rng.uniform())
            total_t += t
        avg_times[i][j] = (total_t / trials)
        avg_sq[i][j] = (total_t / trials)**2
        print("n = ", n, " h = ", h,
              " avg = ", avg_times[i][j],
              " avg^2 = ", avg_sq[i][j])
    np.savetxt("out/averages_nh.dat", avg_times)
    np.savetxt("out/averages_sq_nh.dat", avg_times)

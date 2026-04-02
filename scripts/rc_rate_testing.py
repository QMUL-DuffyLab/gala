import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants
import light
import rc
import solvers
import plots
import genetic_algorithm as ga

output_dir = os.path.join("out", "rc_testing")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "results.csv")
# call light.py to pull the right spectrum and fix intensity
# note - this is not used, it's just that solvers.RC_only
# currently requires a spectrum and i haven't bothered to fix that yet
colour = "red"
intensity = 25.0 # mu_E
spectrum_file = light.spectrum_setup("colour",
        colour=colour, intensity=intensity,
        output_dir=output_dir)
spectrum, output_prefix = light.load_spectrum(spectrum_file)

gammas = np.logspace(0, 5, num=6)
rate_intervals = np.logspace(0, 12, num=13)
print(gammas)
print(rate_intervals)
ps_ox_n_p = 100 # this is degenerate with gamma really but we want to make them equal
ps_r_n_p = 100
n_p = [ps_ox_n_p, ps_r_n_p]
# weightings for the fitness function
output_weight = 1.0
cyclic_weight = 0.1
ps_ox_weight = 5.0
ps_r_weight = 10.0

df_list = []
for g0 in gammas:
    for g1 in gammas:
        for k_lin in rate_intervals:
            for k_ox in rate_intervals:
                for k_red in rate_intervals:
                    for k_cyc in rate_intervals:
                        gamma = [g0, g1]
                        print(f"gamma = {gamma}, lin={k_lin}, ox={k_ox}, red={k_red}, cyc={k_cyc}")
                        od = solvers.RC_only('ox', spectrum, gamma=gamma,
                                             lin=k_lin, ox=k_ox, red=k_red,
                                             cyc=k_cyc, n_p=n_p, debug=True)
                        print(f"nu_e = {od['nu_e']}, nu_cyc = {od['nu_cyc']}, redox = {od['redox']}")
                        od['fitness'] = ga.fitness(od['nu_e'], od['nu_cyc'],
                                                   od['redox'],
                                                   xi=output_weight,
                                                   phi=cyclic_weight,
                                                   chi=ps_ox_weight,
                                                   psi=ps_r_weight)
                        d = {}
                        for k in ['nu_e', 'nu_cyc']:
                            d[k] = od[k]
                        d['ps_ox_gamma'] = gamma[0]
                        d['ps_r_gamma']  = gamma[1]
                        d['ps_ox_n_p'] = ps_ox_n_p
                        d['ps_r_n_p']  = ps_r_n_p
                        for rr in od['rates']:
                            d[f'k_{rr}'] = od['rates'][rr]
                        d['ps_ox_oxidised'] = od['redox'][0, 0]
                        d['ps_ox_reduced']  = od['redox'][0, 1]
                        d['ps_r_oxidised']  = od['redox'][1, 0]
                        d['ps_r_reduced']   = od['redox'][1, 1]
                        d['output_weight'] = output_weight
                        d['cyclic_weight'] = cyclic_weight
                        d['ps_ox_weight'] = ps_ox_weight
                        d['ps_r_weight'] = ps_r_weight
                        d['fitness'] = od['fitness']
                        df_list.append(d)
    
df = pd.from_dict(df_list)
df.to_csv(output_file)
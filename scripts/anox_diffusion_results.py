import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import light

basepath = os.path.join("out", "apocrita", "anox", "cost_0.01")
diffs = [0.1, 1.0, 10.0]
n_radii = 20
Ts = np.array([2343, 3416, 5697])
ls = np.zeros(3, dtype=np.float64)
radii = np.zeros((len(Ts), n_radii))
for i, t in enumerate(Ts):
    for star in light.phz_stars:
        if star[0] == t:
            ls[i] = star[1]
    radii[i] = light.calculate_phz_radii(t, ls[i], n_radii=n_radii)
biglist = []
for d in diffs:
    for i, t in enumerate(Ts):
        for a in radii[i]:
            files = [os.path.join(basepath, f"anox_diffusion_{d}",
                    f"stellar_{t:4d}K_a_{a:6.4f}AU", f"{j}_final_population.csv")
                    for j in range(3)]
            try:
                dfs = [pd.read_csv(f) for f in files]
                combined_pop = pd.concat(dfs)
                biglist.append([d, t, a, combined_pop['nu_e'].mean(),
                    combined_pop['nu_e'].std()])
            except FileNotFoundError:
                biglist.append([d, t, a, np.nan, np.nan])
new_df = pd.DataFrame(biglist, columns=['diffusion rate (in terms of k_ox)',
    'stellar temperature (K)', 'semi-major axis (AU)',
    r'$ \left< \nu_e \right> $', r'$ \sigma_{\nu_e} $'])
new_df.to_csv(os.path.join(basepath, "nu_es.csv"))

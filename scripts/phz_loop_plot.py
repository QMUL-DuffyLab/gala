import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import light
from functools import reduce
import seaborn as sns

fig, axes = plt.subplots(figsize=(30, 16), ncols=3, nrows=2, sharey='row')

for i, (Tstar, Rstar) in enumerate(light.phz_stars):
    for rc, col in zip(['ox', 'anox'], ['seagreen', 'mediumorchid']):
        nu_e = np.full((3, 20), np.nan)
        err = np.full((3, 20), np.nan)
        radii = np.zeros(20, dtype=np.float64)
        for j, a in enumerate(light.phz_radii[i]):
            radii[j] = a
            spectrum, output_prefix = light.spectrum_setup("stellar", Tstar=Tstar, Rstar=Rstar, a=a, attenuation=0.0)
            files = [os.path.join("out", "asterion",
                    f"{rc}_only", "cost_0.01", output_prefix,
                    f"{k}_final_population.csv") for k in range(3)]
            try:
                dfs = [pd.read_csv(f) for f in files]
                combined_pop = pd.concat(dfs)
                nu_e[i, j] = combined_pop['nu_e'].mean()
                err[i, j] = combined_pop['nu_e'].sem()
            except FileNotFoundError:
                pass
        axes[1, i].errorbar(radii, nu_e[i, :], yerr=err[i, :], marker='o', lw=5.0, color=col, label=rc)
        axes[0, i].plot(spectrum[:, 0], spectrum[:, 1], lw=2.0, ls='--', color='gray')
        axes[1, i].legend()
        axes[1, i].axhline(y=10.0, ls='--', color='k')
        axes[0, i].set_xlabel("wavelength (nm)")
        axes[1, i].set_xlabel("semi-major axis (AU)")
        axes[0, i].set_title(f"Tstar = {Tstar}K")

plt.grid(visible=True)
axes[1, 0].set_ylabel(r'$ \left< \nu_e \right> $')
axes[0, 0].set_ylabel("intensity")
fig.tight_layout()
plt.show()
plt.close()

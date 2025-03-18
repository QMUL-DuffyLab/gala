# -*- coding: utf-8 -*-
"""
26/02/2025
@author: callum

"""
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import constants
import light
import rc
import antenna as la

ts = [2300, 2600, 2800, 3300, 3700, 3800, 4300, 4400, 4800, 5800] #temps
ds = ["fast", "thermal", "energy_gap", "none"] # detrapping regimes
rcs = ["ox", "frl", "anox", "exo"] # rc types

n_p = 60
n_dirs   = ["n_per_rc", "n_shared"]

for per_rc, n_dir in zip([True, False], n_dirs):
    outpath = os.path.join("out", "rc_only", n_dir)
    os.makedirs(outpath, exist_ok=True)
    gs = np.zeros((len(ts), len(rcs)), dtype=np.float64)
    cs = np.zeros((len(ts), len(rcs), len(ds)), dtype=np.float64)
    for i, t in enumerate(ts):
        for j, r in enumerate(rcs):
            for k, d in enumerate(ds):
                spectrum, out_name = light.spectrum_setup("phoenix",
                        temperature=t)
                res = rc.solve(r, spectrum, d, n_p, per_rc, True)
                cs[i][j][k] = res['nu_ch2o']
            gs[i][j] = res['gamma']

    gamma = xr.DataArray(gs, coords=[ts, rcs],
            dims=["Stellar temperature", "RC type"])
    # give long_name as latex otherwise xarray linebreaks it
    gamma.attrs["long_name"] = r'$ \text{Excitation rate} $'
    gamma.attrs["units"] = r'$ \gamma s^{-1} $'
    gamma.to_netcdf(os.path.join(outpath, "gamma.nc"))
    carbon = xr.DataArray(cs, coords=[ts, rcs, ds],
            dims=["Stellar temperature", "RC type", "detrap"])
    carbon.attrs["long_name"] = r'$ \text{Carbon reduced} $ '
    carbon.attrs["units"] = r'$ s^{-1} $'
    carbon.to_netcdf(os.path.join(outpath, "carbon.nc"))
    
    # plots
    gamma.plot.line(hue="RC type", lw=5.0, marker='o', ms=10.0)
    ax = plt.gca()
    ax.set_xticks([2500, 3000, 3500, 4000, 4500, 5000, 5500])
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "gamma.pdf"))
    plt.close()

    for d in ds:
        print(carbon.sel(detrap=d))
        carbon.sel(detrap=d).plot.line(hue="RC type", lw=5.0, marker='o', ms=10.0)
        ax = plt.gca()
        ax.set_xticks([2500, 3000, 3500, 4000, 4500, 5000, 5500])
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f"carbon_{d}_detrap.pdf"))
        plt.close()

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

n_p = 60

intensities = [1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 10000]
tau_diffs = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
rc_redox = ["PS_ox_ox", "PS_ox_red", "PS_r_ox", "PS_r_red"]
rc_names = ["PS_ox", "PS_r"]

redox = np.zeros((len(intensities), len(tau_diffs), 4), dtype=np.float64)
rec = np.zeros((len(intensities), len(tau_diffs), 2), dtype=np.float64)
nu_e = np.zeros((len(intensities), len(tau_diffs)), dtype=np.float64)
eff = np.zeros_like(nu_e)
gamma = np.zeros(len(intensities), dtype=np.float64)
outpath = os.path.join("out", "ox_intensity")
os.makedirs(outpath, exist_ok=True)
for i, mu_e in enumerate(intensities):
    for j, td in enumerate(tau_diffs):
        spectrum, out_name = light.spectrum_setup("am15",
                dataset="tilt", intensity=mu_e, region=[400.0, 700.0])
        res = rc.solve("ox", spectrum, "none", td, n_p, True, True)
        redox[i][j] = res['redox'].flatten()
        print(f"intensity = {mu_e}")
        print(f"gamma = {res['gamma']}")
        print(f"tau_ox = {res['tau_ox']}")
        print(f"nu_e = {res['nu_ch2o']}")
        print(f"redox = {res['redox']}")
        print(f"recomb = {res['recomb']}")
        print()
        rec[i][j] = res['recomb']
        nu_e[i][j] = res['nu_ch2o']
        gamma[i] = res['gamma']
        eff[i][j] = nu_e[i][j] / gamma[i]
        
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(intensities, gamma, lw=5.0, marker='o', ms=10.0)
ax.set_xscale('log')
ax.set_xticks(intensities)
ax.set_xlabel(r'intensity $ \mu E $')
ax.set_yscale('log')
ax.set_yticks([0.01, 0.1, 1.0, 10.0, 100.0])
ax.set_ylabel(r'excitation rate $ \gamma s^{-1} $')
plt.grid(visible=True)
fig.tight_layout()
fig.savefig(os.path.join(outpath, "gamma_intensity.pdf"))
plt.close()

np.savetxt(os.path.join(outpath, "gamma.txt"), gamma)
redox_xr = xr.DataArray(redox,
        coords=[intensities, tau_diffs, rc_redox],
        dims=["intensity", "tau_diff", "rc_redox"])
# give long_name as latex otherwise xarray linebreaks it
redox_xr.attrs["long_name"] = r'$ \text{Redox states} $'
redox_xr.attrs["units"] = r'$ probabilities $'
redox_xr.to_netcdf(os.path.join(outpath, "redoxes.nc"))
recomb_xr = xr.DataArray(rec,
        coords=[intensities, tau_diffs, rc_names],
        dims=["intensity", "tau_diff", "rc"])
recomb_xr.attrs["long_name"] = r'$ \text{Recombination losses} $ '
recomb_xr.attrs["units"] = r'$ s^{-1} $'
recomb_xr.to_netcdf(os.path.join(outpath, "recomb.nc"))
nu_e_xr = xr.DataArray(nu_e,
        coords=[intensities, tau_diffs],
        dims=["intensity", "tau_diff"])
nu_e_xr.attrs["long_name"] = r'$ \text{Electron output} $ '
nu_e_xr.attrs["units"] = r'$ s^{-1} $'
nu_e_xr.to_netcdf(os.path.join(outpath, "nu_e.nc"))
eff_xr = xr.DataArray(eff,
       coords=[intensities, tau_diffs],
       dims=["intensity", "tau_diff"])
eff_xr.attrs["long_name"] = r'$ \text{efficiency} $ '
eff_xr.attrs["units"] = r'$ \% $'
eff_xr.to_netcdf(os.path.join(outpath, "efficiency.nc"))

for mu_e in intensities:
    # plots
    mu_xr = redox_xr.sel(intensity=mu_e)
    mu_xr.plot.line(hue="rc_redox", lw=5.0, marker='o', ms=10.0)
    ax = plt.gca()
    ax.set_xticks(tau_diffs)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"redox_intensity_{mu_e}.pdf"))
    plt.close()

    nu_xr = nu_e_xr.sel(intensity=mu_e)
    nu_xr.plot.line(lw=5.0, marker='o', ms=10.0)
    ax = plt.gca()
    ax.set_xticks(tau_diffs)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"nu_e_intensity_{mu_e}.pdf"))
    plt.close()

for td in tau_diffs:
    tau_xr = redox_xr.sel(tau_diff=td)
    tau_xr.plot.line(hue="rc_redox", lw=5.0, marker='o', ms=10.0)
    ax = plt.gca()
    ax.set_xticks(intensities)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"redox_diffusion_{td}.pdf"))
    plt.close()

ds = xr.Dataset(
        data_vars=dict(
        redox=(["intensity", "tau_diff"], redox),
        recomb=(["intensity", "tau_diff"], rec),
        nu_e=(["intensity", "tau_diff"], nu_e),
        gamma=(["intensity"], gamma),
        efficiency=(["intensity", "tau_diff"], eff),
        ),
        coords=dict(
            intensity=intensities,
            tau_diff=tau_diffs,
            ),
        attrs=dict(description="RC parameter dataset"))
ds.to_netcdf(os.path.join(outpath, "whole_dataset.nc"))

def stellar_temp_rc_comp():
    '''
    RC types as function of stellar temperature
    '''
    ts = [2300, 2600, 2800, 3300, 3700, 3800, 4300, 4400, 4800, 5800] #temps
    ds = ["fast", "thermal", "energy_gap", "none"] # detrapping regimes
    rcs = ["ox", "frl", "anox", "exo"] # rc types

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
                    res = rc.solve(r, spectrum, d, 0.0, n_p, per_rc, True)
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
            

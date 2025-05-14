# -*- coding: utf-8 -*-
"""
9/4/24
@author: callum
i keep writing different tests in different files and deleting them
for various reasons. so just keep them all here instead
"""
import ctypes
import numpy as np
import genetic_algorithm as ga
import utils
import solvers
import constants
import light
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

def test_connected_kmat():
    '''
    generate a pair of identical antennae with one connected and one
    not. solve the equations and print out all the debug info for both
    to check that the rate matrix is reasonable etc.
    '''
    spectrum_dict = [{'type': "am15", 'kwargs': {'dataset': "tilt"}}]
    spectrum, out_name = zip(*light.build(spectrum_dict))
    # zip(*x) returns a 1-tuple rather than just the array
    l = spectrum[0][:, 0]
    ip_y = spectrum[0][:, 1]
    rng = np.random.default_rng()

    c = ga.new(rng)
    c.connected = True
    u = ga.copy(c)
    u.connected = False
    cd = solvers.antenna_only(l, ip_y, c, debug=True)
    ud = solvers.antenna_only(l, ip_y, u, debug=True)

    return [cd, ud]

def test_connected_fitness(n_trials=1000):
    '''
    test the connected property of antennae by creating pairs of
    identical antennae with one connected and one not. keep track
    of the results and return two (n_trials, 3) arrays
    '''
    connected_res = np.zeros((n_trials, 3), dtype=np.float64)
    unconnected_res = np.zeros((n_trials, 3), dtype=np.float64)
    spectrum_dict = [{'type': "am15", 'kwargs': {'dataset': "tilt"}}]
    spectrum, out_name = zip(*light.build(spectrum_dict))
    l = spectrum[0][:, 0]
    ip_y = spectrum[0][:, 1]
    rng = np.random.default_rng()

    for i in range(n_trials):
        c = ga.new(rng)
        c.connected = True
        u = ga.copy(c)
        u.connected = False
        cr = solvers.antenna_only(l, ip_y, c)
        ur = solvers.antenna_only(l, ip_y, u)
        connected_res[i] = cr
        unconnected_res[i] = ur
        diff = np.abs(cr - ur)
        print(str.format(("Iteration {:d}: Δν_e = {:10.4e}, "
        + "Δϕ_e(γ) = {:10.6e}, Δϕ_e = {:10.6e}"), i, *diff))

    return [connected_res, unconnected_res]

def test_antenna_fortran():
    spectrum_dict = [{'type': "am15", 'kwargs': {'dataset': "tilt"}}]
    spectrum, out_name = zip(*light.build(spectrum_dict))
    l = spectrum[0][:, 0].astype(ctypes.c_double)
    ip_y = spectrum[0][:, 1].astype(ctypes.c_double)
    # set up fortran stuff
    libantenna = ctypes.CDLL("./lib/libantenna.so")
    libantenna.fitness_calc.argtypes = [ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  np.ctypeslib.ndpointer(dtype='a10', ndim=1),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_double),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  ctypes.POINTER(ctypes.c_int),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  ]
    libantenna.fitness_calc.restype = None

    n_b = 4
    n_s = 3
    n_p = [80, 70, 60]
    shift = [-5.0, 0.5, 5.0]
    pigment = ['chl_a', 'chl_b', 'chl_f']
    g = ga.Genome(n_b, n_s, n_p, shift, pigment)
    print("Python:")
    print("n_b = ", g.n_b)
    print("n_s = ", g.n_s)
    print("n_p = ", g.n_p)
    print("shift = ", g.shift)
    print("pigment = ", g.pigment)
    fr = np.zeros(3).astype(ctypes.c_double)
    n_p = np.array([constants.np_rc, *g.n_p], dtype=ctypes.c_int)
    shift = np.array([0., *g.shift], dtype=ctypes.c_double)
    # we have to format them like this, otherwise numpy truncates
    pigment = np.array([f"{p:<10}" for p in [constants.rc_type, *g.pigment]],
                       dtype='a10', order='F')
    kp = np.array(constants.k_params, dtype=ctypes.c_double)
    n_b = ctypes.byref(ctypes.c_int(g.n_b))
    n_s = ctypes.byref(ctypes.c_int(g.n_s))
    temp = ctypes.byref(ctypes.c_double(constants.T))
    gf = ctypes.byref(ctypes.c_double(constants.gamma_fac))
    ll = ctypes.byref(ctypes.c_int(len(l)))
    libantenna.fitness_calc(n_b, n_s,
                            n_p, shift, pigment, kp,
                            temp,
                            gf, l, ip_y,
                            ll, fr)
    pr = solvers.antenna_only(l, ip_y, g)
    print("fortran = ", fr)
    print("python = ", pr)
    diff = np.abs(pr - fr)
    print(str.format(("Δν_e = {:10.4e}, "
    + "Δϕ_e(γ) = {:10.6e}, Δϕ_e = {:10.6e}"), *diff))

def test_python_fortran(n_trials=1000):
    '''
    generate antennae and then solve, using python and my fortran code.
    check the results are the same and time them.
    '''
    python_res = np.zeros((n_trials, 3), dtype=np.float64)
    fortran_res = np.zeros((n_trials, 3), dtype=np.float64)
    spectrum_dict = [{'type': "am15", 'kwargs': {'dataset': "tilt"}}]
    spectrum, out_name = zip(*light.build(spectrum_dict))
    l = spectrum[0][:, 0].astype(ctypes.c_double)
    ip_y = spectrum[0][:, 1].astype(ctypes.c_double)
    rng = np.random.default_rng()

    # set up fortran stuff
    libantenna = ctypes.CDLL("./lib/libantenna.so")
    libantenna.fitness_calc.argtypes = [ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  np.ctypeslib.ndpointer(dtype='a10', ndim=1),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_double),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  ctypes.POINTER(ctypes.c_int),
                  np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
                  ]
    libantenna.fitness_calc.restype = None

    for i in range(n_trials):
        g = ga.new(rng)
        fr = np.zeros(3).astype(ctypes.c_double)
        n_p = np.array([constants.np_rc, *g.n_p], dtype=ctypes.c_int)
        shift = np.array([0., *g.shift], dtype=ctypes.c_double)
        # we have to format them like this, otherwise numpy truncates
        pigment = np.array([f"{p:<10}" for p in ["rc_ox", *g.pigment]],
                           dtype='a10', order='F')
        kp = np.array(constants.k_params, dtype=ctypes.c_double)
        n_b = ctypes.byref(ctypes.c_int(g.n_b))
        n_s = ctypes.byref(ctypes.c_int(g.n_s))
        temp = ctypes.byref(ctypes.c_double(constants.T))
        gf = ctypes.byref(ctypes.c_double(constants.gamma_fac))
        ll = ctypes.byref(ctypes.c_int(len(l)))
        libantenna.fitness_calc(n_b, n_s,
                                n_p, shift, pigment, kp,
                                temp,
                                gf, l, ip_y,
                                ll, fr)
        pr = solvers.antenna_only(l, ip_y, g)
        python_res[i] = pr
        fortran_res[i] = fr
        diff = np.abs(pr - fr)
        print(str.format(("Iteration {:d}: n_b = {:d}, n_s = {:d}, "
        + "Δν_e = {:10.4e}, "
        + "Δϕ_e(γ) = {:10.6e}, Δϕ_e = {:10.6e}"), i, g.n_b, g.n_s, *diff))

    return [python_res, fortran_res]

def test_pigment_fits(pigment_list=None):
    with open(os.path.join("pigments", "pigment_data.json")) as f:
        all_params = json.load(f)

    outdir = os.path.join("out", "pigment_fits")
    os.makedirs(outdir, exist_ok=True)
    if pigment_list is not None:
        names = pigment_list
    else:
        names = list(all_params.keys())
    files = [os.path.join("pigments", f"{n}_{f}.txt") for n in names for f in ("abs", "ems")]
    params = [all_params[n][f] for n in names for f in ("abs", "ems")]

    cmap = mpl.colormaps["turbo"]
    colours = [cmap(i / float(len(names))) for i in range(len(names))]
    cdict = {n: c for n, c in zip(names, colours)}

    xlim = [400.0, 1000.0]
    for n, f, p in zip(np.repeat(names, 2), files, params):
        print(n, f, p)
        data = pd.read_csv(f, delimiter='\t', comment='#').to_numpy()
        mu = p['mu']
        sigma = p['sigma']
        amp = p['amp']
        xtest = np.arange(*xlim)
        ytest = utils.gauss(xtest, mu, sigma, amp)
        np.savetxt(os.path.join(outdir, f"{os.path.splitext(os.path.basename(f))[0]}.txt"), np.column_stack((xtest, ytest)))

        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(xtest, ytest/np.max(ytest), label='test fit', color=cdict[n], lw=3.0)
        plt.plot(data[:, 0], data[:, 1]/np.max(data[:, 1]), label=f"{p['text']}", lw=3.0, color='k', ls='--')
        plt.axvline(p["0-0"], lw=2.0, ls='--', color='C0')
        ax.set_xlabel("wavelength (nm)")
        ax.set_xlim(xlim)
        ax.set_ylabel("intensity (arbitrary)")
        plt.grid(visible=True)
        plt.legend(fontsize=20)
        plt.savefig(os.path.join(outdir, f"{os.path.splitext(os.path.basename(f))[0]}.pdf"))
        plt.show()
        plt.close()

def test_overlaps(spectrum, pigment_list=None):
    with open(os.path.join("pigments", "pigment_data.json")) as f:
        all_params = json.load(f)

    outdir = os.path.join("out", "tests", "overlaps")
    os.makedirs(outdir, exist_ok=True)
    if pigment_list is not None:
        names = pigment_list
    else:
        names = list(all_params.keys())

    print(names)
    overlaps, gammas, abso, emis = utils.lookups(spectrum, names, True)

    xlim = [400.0, 1000.0]
    cmap = mpl.colormaps["turbo"]
    colours = [cmap(i / float(len(names))) for i in range(len(names))]
    cdict = {n: c for n, c in zip(names, colours)}
    for i in range(len(names)):
        fig, ax = plt.subplots(ncols=2,
                nrows=len(names), figsize=(30, 6 * len(names)),
                sharex=True, sharey=True)
        plt.subplots_adjust(wspace=None, hspace=None)
        for j in range(len(names)):
            ni = names[i]
            nj = names[j]
            oae = overlaps[nj][ni]
            ax[j][0].plot(spectrum[:, 0], abso[ni],
                    color=cdict[ni], label=f"absorber: {ni}")
            ax[j][0].plot(spectrum[:, 0], emis[nj],
                    color=cdict[nj], label=f"emitter: {nj}")
            ax[j][0].plot([], [], ' ', label=f"Overlap = {oae:6.3f}")
            ax[j][0].legend()
            oea = overlaps[ni][nj]
            ax[j][1].plot(spectrum[:, 0], emis[ni],
                    color=cdict[ni], label=f"emitter: {ni}")
            ax[j][1].plot(spectrum[:, 0], abso[nj],
                    color=cdict[nj], label=f"absorber: {nj}")
            ax[j][1].plot([], [], ' ', label=f"Overlap = {oea:6.3f}")
            ax[j][1].set_xlim(xlim)
            ax[j][1].legend()

        fig.supxlabel("wavelength (nm)")
        fig.supylabel("intensity (arb.)")
        plt.grid(visible=True)
        # fig.tight_layout()
        plt.savefig(os.path.join(outdir, f"{names[i]}_overlaps.pdf"))
        plt.close()

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
import antenna as la
import constants
import light

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

    c = ga.new(rng, 'random')
    c.connected = True
    u = ga.copy(c)
    u.connected = False
    cd = la.antenna(l, ip_y, c, debug=True)
    ud = la.antenna(l, ip_y, u, debug=True)

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
        c = ga.new(rng, 'random')
        c.connected = True
        u = ga.copy(c)
        u.connected = False
        cr = la.antenna(l, ip_y, c)
        ur = la.antenna(l, ip_y, u)
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
    g = constants.Genome(n_b, n_s, n_p, shift, pigment)
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
    pr = la.antenna(l, ip_y, g)
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
        g = ga.new(rng, 'random')
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
        pr = la.antenna(l, ip_y, g)
        python_res[i] = pr
        fortran_res[i] = fr
        diff = np.abs(pr - fr)
        print(str.format(("Iteration {:d}: n_b = {:d}, n_s = {:d}, "
        + "Δν_e = {:10.4e}, "
        + "Δϕ_e(γ) = {:10.6e}, Δϕ_e = {:10.6e}"), i, g.n_b, g.n_s, *diff))

    return [python_res, fortran_res]

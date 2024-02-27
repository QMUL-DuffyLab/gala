import numpy as np
import ctypes
import scipy.optimize as opt


if __name__ == "__main__":

    nmmax = 30
    rng = np.random.default_rng()
    for i in range(1000):
        n = rng.integers(nmmax)
        m = rng.integers(nmmax)
        a = np.zeros((m, n), order='F')
        rng.random(size=(m, n), out=a)
        x_ref = rng.random(size=n)
        b = a @ x_ref
        doubleptr = ctypes.POINTER(ctypes.c_double)
        intptr = ctypes.POINTER(ctypes.c_int)
        libnnls = ctypes.CDLL("./libnnls.so")
        libnnls.solve.argtypes = [doubleptr, doubleptr, doubleptr,
                                 intptr, intptr,
                                 intptr, doubleptr,
                                 intptr, doubleptr]
        libnnls.solve.restype = None
        mode = ctypes.c_int(0)
        res_f = ctypes.c_double(0.0)
        maxiter = ctypes.c_int(3 * n)
        tol = ctypes.c_double(1e-6)
        x_f = np.zeros(n, dtype=ctypes.c_double)
        libnnls.solve(a.ctypes.data_as(doubleptr),
                     b.ctypes.data_as(doubleptr),
                     x_f.ctypes.data_as(doubleptr),
                     ctypes.c_int(m),
                     ctypes.c_int(n),
                     ctypes.byref(mode),
                     ctypes.byref(res_f),
                     ctypes.byref(maxiter),
                     ctypes.byref(tol))
        try:
            x_py, res_py = opt.nnls(a, b)
        except RuntimeError:
            x_py = 0.0
            res_py = 0.0
        rf_diff = np.linalg.norm(x_f - x_ref)
        rp_diff = np.linalg.norm(x_py - x_ref)
        fp_diff = np.linalg.norm(x_f - x_py)
        if (fp_diff > 1e-6):
            print("i = {:4d}: res_f = {:10.6e}, res_py = {:10.6e}, |f-ref| = {:10.6e}, |py-ref| = {:10.6e}, |f-py| = {:10.6e}".format(i, res_f.value, res_py, rf_diff, rp_diff, fp_diff))
        else:
            print(i, fp_diff)

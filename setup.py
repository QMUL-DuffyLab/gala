from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
        Extension("build_matrix", ["build_matrix.pyx"],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ]
setup(
    name="build_matrix",
    ext_modules=cythonize(extensions)
)

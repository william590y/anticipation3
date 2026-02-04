"""
Setup script to compile Cython DTW alignment module.

Usage:
    python setup_dtw.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="dtw_alignment",
    ext_modules=cythonize(
        "dtw_alignment.pyx",
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)

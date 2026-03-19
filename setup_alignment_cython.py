from setuptools import Extension, setup

import numpy
from Cython.Build import cythonize


extensions = [
    Extension(
        "alignment_cython",
        ["alignment_cython.pyx"],
        include_dirs=[numpy.get_include()],
    )
]


setup(
    name="alignment_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3},
    ),
)

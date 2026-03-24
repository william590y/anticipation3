from setuptools import Extension, setup
from Cython.Build import cythonize


extensions = [
    Extension(
        "alignment_cython",
        sources=["alignment_cython.pyx"],
    ),
]


setup(
    name="alignment_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
        },
    ),
)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = 'C Cribbage Scorer',
    ext_modules = cythonize([Extension("c_cribbage_score", ["c_cribbage_score.pyx", "cribbage_score.c"])])
)

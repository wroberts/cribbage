from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = 'C Cribbage Scorer',
    ext_modules = cythonize([Extension("cribbage._cribbage_score",
                                       ["cribbage/cribbage_score_wrapper.pyx",
                                        "cribbage/cribbage_score.c"])])
)

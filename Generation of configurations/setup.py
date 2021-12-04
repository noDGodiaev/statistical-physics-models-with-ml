import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import mc_lib



ext = Extension("gen_samples_for_autocorr", ["gen_samples_for_autocorr.pyx"],
                include_dirs = [numpy.get_include(),
                                mc_lib.get_include()],
                language='c++',
                )

setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
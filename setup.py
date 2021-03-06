from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules = [Extension("cython_process_array", sources=["cython_process_array.pyx", "process_array.c", "process_GPS.c"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], include_dirs=[numpy.get_include()]), 
			   Extension("cython_simulator", sources=["cython_simulator.pyx", "simulator.c"], include_dirs=[numpy.get_include()])
			  ]
setup(
  name = 'Suite',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules)
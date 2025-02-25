from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

ext_modules = [
    Extension(
        "LebwohlLasher_cy_parallel",
        sources=["LebwohlLasher_cy_parallel.pyx"],
      extra_compile_args=['-fopenmp','-O3', "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
      extra_link_args=['-fopenmp', '-O3'], 
      include_dirs=[np.get_include()])]


setup(name="LebwohlLasher_cy_parallel",
      ext_modules=cythonize(ext_modules))


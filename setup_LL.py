from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

ext_modules = [
    Extension(
        "LebwohlLasher_cy",
        sources=["LebwohlLasher_cy.pyx"],
      extra_compile_args=['-O3', "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
      extra_link_args=['-O3'], 
      include_dirs=[np.get_include()])]


setup(name="LebwohlLasher_cy",
      ext_modules=cythonize(ext_modules))


from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

ext_modules = [
    Extension(
        "LebwohlLasher_cy",
        sources=["LebwohlLasher_cy.pyx"],
      extra_compile_args=['-O3'],
      extra_link_args=['-O3'] )]


setup(name="LebwohlLasher_cy",
      ext_modules=cythonize(ext_modules))


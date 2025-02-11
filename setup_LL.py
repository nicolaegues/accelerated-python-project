from distutils.core import setup
from Cython.Build import cythonize


setup(name="LebwohlLasher_cy",
      ext_modules=cythonize("LebwohlLasher_cy.pyx"))


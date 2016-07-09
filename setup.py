from setuptools import setup
from nphusl import __version__
import numpy
from Cython.Build import cythonize


setup(name='nphusl',
      version=__version__,
      py_modules=['nphusl', "_nphusl_expr"],
      ext_modules=cythonize("_nphusl_cython.pyx"),
      include_dirs=[numpy.get_include()],
)

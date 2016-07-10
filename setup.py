from setuptools import setup, Extension
from nphusl import __version__
import numpy
from Cython.Build import cythonize


extensions = [
    Extension("_nphusl_cython", sources=["_nphusl_cython.pyx"],
              extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"])
]


setup(name='nphusl',
      version=__version__,
      py_modules=['nphusl', "_nphusl_expr"],
      #ext_modules=cythonize("_nphusl_cython.pyx"),
      ext_modules=cythonize(extensions),
      include_dirs=[numpy.get_include()],
)



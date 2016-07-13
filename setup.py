from setuptools import setup, Extension
from nphusl import __version__
import numpy
from Cython.Build import cythonize


extensions = [
    Extension("nphusl._nphusl_cython", sources=["nphusl/_nphusl_cython.pyx"],
              extra_compile_args=["-fopenmp", "-O3", "-ffast-math"],
              extra_link_args=["-fopenmp"])
]


setup(name='nphusl',
      version=__version__,
      packages=["nphusl"],
      install_requires=["Cython", "numpy"],
      ext_modules=cythonize(extensions),
      include_dirs=[numpy.get_include()],
)



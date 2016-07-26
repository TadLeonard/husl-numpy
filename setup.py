import sys
from setuptools import setup, Extension
from nphusl import __version__
import numpy
if "--use-cython" in sys.argv:
    sys.argv.remove("--use-cython")
    USE_CYTHON = True
else:
    USE_CYTHON = False


url = "https://github.com/TadLeonard/husl-numpy"
download = "{}/archive/{}.tar.gz".format(url, __version__)

long_description = """
HUSL color space conversion
===========================

A color space conversion library that works with numpy. See
http://husl-colors.org to learn about the HUSL color space.


Features
--------

1. Fast conversion to RGB from HUSL and vice versa. Convert a 1080p image to
HUSL in less than a second.
2. Seamless performance improvements with `NumExpr`, `Cython`, and `OpenMP`
(whichever's available).
3. Flexible `numpy` arrays as inputs and outputs. Plays nicely with `OpenCV`,
`MoviePy`, etc.

Installation
------------

1. `virtualenv env -p python3`
2. `source env/bin/activate`
3. `pip install numpy`
4. (optional) `pip install Cython`  (or NumExpr, but Cython is preferred)
5. `pip install git+https://github.com/TadLeonard/husl-numpy.git`

Basic usage
-----------

* `to_rgb(hsl)` Convert HUSL array to RGB integer array
* `to_husl(rgb)` Convert RGB integer array or grayscale float array to HUSL
* array
* `to_hue(rgb)` Convert RGB integer array or grayscale float array to array of
* hue values

More
====

See {url} for complete documentation.
""".format(url=url)

description = ("A HUSL color space conversion library that works with numpy")

classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Developers',
  'Operating System :: OS Independent',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3 :: Only',
  'Intended Audience :: Science/Research',
  'Topic :: Multimedia',
  'Topic :: Multimedia :: Graphics',
]

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension("nphusl._cython_opt",
              sources=["nphusl/_cython_opt"+ext],
              extra_compile_args=["-fopenmp", "-O3", "-ffast-math"],
              extra_link_args=["-fopenmp"]),
    Extension("nphusl._simd_opt",
              sources=["nphusl/_simd_wrap"+ext,
                       "nphusl/_simd_opt.c"],
              include_dirs=["nphusl/"],
              extra_compile_args=["-fopenmp", "-O3", "-ffast-math"],
              extra_link_args=["-fopenmp"])
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


setup(name='nphusl',
      version=__version__,
      packages=["nphusl"],
      install_requires=["numpy"],
      ext_modules=extensions,
      include_dirs=[numpy.get_include()],
      licence="MIT",
      description=description,
      long_description=long_description,
      classifiers=classifiers,
      author="Tad Leonard",
      author_email="tadfleonard@gmail.com",
      keywords="husl hsl color conversion rgb image processing",
      url=url,
      download_url=download,
      zip_safe=False,
)



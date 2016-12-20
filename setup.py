import sys
assert sys.version_info >= (3, 4), "Python 3.4+ only!"

from collections import namedtuple
from enum import Enum
from pprint import pformat
from setuptools import setup, Extension
from nphusl import __version__
import numpy


CompileArg = namedtuple("CompileArg", "setup_arg cc_cmd")


class Arg(CompileArg, Enum):
    CYTHONIZE = CompileArg("--cythonize", None)
    NO_CYTHON_EXT = CompileArg("--no-cython-ext", None)
    NO_SIMD_EXT = CompileArg("--no-simd-ext", None)
    NO_LIGHT_LUT = CompileArg(
        "--no-light-lut", "-DUSE_LIGHT_LUT")
    NO_CHROMA_LUT = CompileArg(
        "--no-chroma-lut", "-DUSE_CHROMA_LUT")
    NO_HUE_ATAN2_APPROX = CompileArg(
        "--no-hue-atan2-approx", "-DUSE_HUE_ATAN2_APPROX")
    INTERPOLATE_CHROMA = CompileArg(
        "--interpolate-chroma", "-DINTERPOLATE_CHROMA")


args = {}
for arg in Arg:
    arg_enabled = arg.setup_arg in sys.argv
    args[arg] = arg_enabled
    if arg_enabled:
        sys.argv.remove(arg.setup_arg)

print("Application options:\n{}".format(
    pformat({k.name: v for k, v in args.items()})))

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

Installation from source
------------------------

1. `python3.5 -m venv env/`
2. `source env/bin/activate`
3. `git clone https://github.com/TadLeonard/husl-numpy.git`
4. `pip install -r dev-requirements.txt`
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


ext = '.pyx' if args[Arg.CYTHONIZE] else '.c'
extensions = []
cython_compile_args = ["-fopenmp", "-O3", "-ffast-math"]


simd_compile_args = [
     "-ftree-vectorize",
     "-ftree-vectorizer-verbose=2",
     "-std=c99",
     "-mtune=native",
] + cython_compile_args


simd_sources=["nphusl/_simd_opt"+ext,
              "nphusl/_simd.c",
              "nphusl/_linear_lookup.c",
              "nphusl/_scale_const.c",
]


if not args[Arg.NO_LIGHT_LUT]:
    simd_sources.append("nphusl/_light_lookup.c")
    simd_compile_args.append(Arg.NO_LIGHT_LUT.cc_cmd)
if not args[Arg.NO_CHROMA_LUT]:
    simd_sources.append("nphusl/_chroma_lookup.c")
    simd_compile_args.append(Arg.NO_CHROMA_LUT.cc_cmd)
if args[Arg.INTERPOLATE_CHROMA]:
    simd_compile_args.append(Arg.INTERPOLATE_CHROMA.cc_cmd)
if not args[Arg.NO_HUE_ATAN2_APPROX]:
    simd_compile_args.append(Arg.NO_HUE_ATAN2_APPROX.cc_cmd)


cython_ext = Extension("nphusl._cython_opt",
                       sources=["nphusl/_cython_opt"+ext],
                       extra_compile_args=cython_compile_args,
                       extra_link_args=["-fopenmp"])


simd_ext = Extension("nphusl._simd_opt",
                     sources=simd_sources,
                     extra_compile_args=simd_compile_args,
                     include_dirs=["nphusl/"],
                     extra_link_args=["-fopenmp"])


if not args[Arg.NO_CYTHON_EXT]:
    extensions.append(cython_ext)
if not args[Arg.NO_SIMD_EXT]:
    extensions.append(simd_ext)
if args[Arg.CYTHONIZE]:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


setup(name='nphusl',
      version=__version__,
      packages=["nphusl"],
      install_requires=["numpy"],
      ext_modules=extensions,
      include_dirs=[numpy.get_include()],
      license="MIT",
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



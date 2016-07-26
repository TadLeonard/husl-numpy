"""HUSL color space conversion.

The basics:

* to_rgb(hsl): Convert HUSL array to RGB integer array
* to_husl(rgb): Convert RGB integer array or grayscale float array to HUSL array
* to_hue(rgb): Convert RGB integer array or grayscale float array to array of hue values
"""

__version__ = "1.4.1"

from . import nphusl as _nphusl
from .nphusl import *
from . import constants

try:
    from . import _numexpr_opt
except ImportError:
    pass
try:
    from . import _cython_opt
except ImportError:
    pass
try:
    from . import _simd_opt
except ImportError:
    pass


def _enable_fns(fn_dictionary):
    for name, fn in fn_dictionary.items():
        globals()[name] = fn
        setattr(nphusl, name, fn)


def enable_standard_fns():
    _enable_fns(STANDARD)


def enable_cython_fns():
    _enable_fns(CYTHON)


def enable_numexpr_fns():
    _enable_fns(NUMEXPR)


def enable_simd_fns():
    _enable_fns(SIMD)


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
    from . import _nphusl_expr
except ImportError:
    pass
try:
    from . import _nphusl_cython
except ImportError:
    pass


def enable_standard_fns():
    for name, fn in STANDARD.items():
        globals()[name] = fn
        setattr(nphusl, name, fn)


def enable_cython_fns():
    for name, fn in CYTHON.items():
        globals()[name] = fn
        setattr(nphusl, name, fn)


def enable_numexpr_fns():
    for name, fn in NUMEXPR.items():
        globals()[name] = fn
        setattr(nphusl, name, fn)



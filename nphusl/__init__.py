"""HUSL color space conversion.

The basics:

* to_rgb(hsl): Convert HUSL array to RGB integer array
* to_husl(rgb): Convert RGB integer array or grayscale float array to HUSL array
* to_hue(rgb): Convert RGB integer array or grayscale float array to array of hue values
"""

__version__ = "1.5.0"

from contextlib import contextmanager
from functools import partial

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


def enable_best_optimized():
    order = SIMD, CYTHON, NUMEXPR, NUMPY
    for name, fn in NUMPY.items():
        fns = (fn_map.get(name) for fn_map in order)
        chosen_fn = next(f for f in fns if f)
        _select_fn(chosen_fn, name)


@contextmanager
def _with_fns(enable_other_fns, back_to_std=False):
    revert = enable_numpy if back_to_std else enable_best_optimized
    enable_other_fns()
    try:
        yield
    finally:
        revert()


def _enable_fns(fn_dictionary=None):
    _set_module_globals(NUMPY)
    if fn_dictionary:
        _set_module_globals(fn_dictionary)


def _set_module_globals(fn_dictionary):
    assert fn_dictionary, "implementation not available"
    for name, fn in fn_dictionary.items():
        _select_fn(fn, name)


def _select_fn(fn, name):
    globals()[name] = fn
    setattr(nphusl, name, fn)


enable_numpy = partial(_enable_fns, NUMPY)
enable_cython = partial(_enable_fns, CYTHON)
enable_numexpr = partial(_enable_fns, NUMEXPR)
enable_simd = partial(_enable_fns, SIMD)
best_enabled = partial(_with_fns, enable_best_optimized)
numpy_enabled = partial(_with_fns, enable_numpy)
cython_enabled = partial(_with_fns, enable_cython)
numexpr_enabled = partial(_with_fns, enable_numexpr)
simd_enabled = partial(_with_fns, enable_simd)


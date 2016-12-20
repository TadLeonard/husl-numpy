"""
HUSL (human-friendly Hue, Saturation, and Lightness)
color conversion. Found in this package:

HUSL <-> RGB color conversion API
   * `to_husl`: converts an RGB array to a HUSL array
   * `to_rgb`: converts a HUSL array to and RGB array
   * `to_hue`: converts an RGB array to an array of HUSL hue values

Context managers for enabling specific optimizations:
   * `with_simd`: enablel OpenMP SIMD-friendly C implementation
   * `with_cython`: enable Cython+OpenMp implementation
   * `with_numexpr`: enbable NumExpr implementation
   * `with_numpy`: enable the standard NumPy implementation

The C SIMD-friendly implementation is used if it's available.
"""

__version__ = "1.5.0"
__all__ = ["to_husl", "to_hue", "to_rgb"]


from contextlib import contextmanager
from functools import partial

from .nphusl import to_husl, to_hue, to_rgb
from .nphusl import SIMD, CYTHON, NUMEXPR, NUMPY
from . import nphusl
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


del partial
del contextmanager

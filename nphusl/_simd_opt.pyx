"""Wrapper for _simd.c, the HUSL <-> RGB conversion  C implementation."""

import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport free

from . import transform


hsl_type = np.float64
ctypedef np.float64_t hsl_t
cdef size_t data_size = sizeof(hsl_t)


cdef extern from "_simd.h":
    hsl_t* rgb_to_husl_nd(np.uint8_t *rgb, size_t size)


@transform.rgb_int_input
def _rgb_to_husl(rgb):
    cdef size_t size = rgb.size
    cdef int pixels
    cdef hsl_t[::1] hsl_flat
    cdef np.ndarray out
    try:
        pixels = size / 3
        rgb_flat = rgb.reshape((pixels, 3))
        hsl_flat = _rgb_to_husl_2d(rgb_flat, size)
        out = np.asarray(hsl_flat, dtype=hsl_type).reshape(rgb.shape)
        return out.copy()
    finally:
        free(<void*>&hsl_flat[0])


cdef hsl_t[::1] _rgb_to_husl_2d(np.uint8_t[:, ::1] rgb, size_t size):
    cdef hsl_t *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0], size)
    cdef hsl_t[::1] husl = <hsl_t[:size]> hsl_ptr
    return husl


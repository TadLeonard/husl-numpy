"""Wrapper for _simd.c, the HUSL <-> RGB conversion  C implementation."""

import numpy as np
cimport numpy as np
import cython

from . import transform


hsl_type = np.float64
ctypedef np.float64_t hsl_t
cdef size_t data_size = sizeof(hsl_t)


cdef extern from "_simd.h":
    hsl_t* rgb_to_husl_nd(np.uint8_t *rgb, int size)


@transform.rgb_int_input
def _rgb_to_husl(rgb):
    cdef int size = rgb.size
    cdef int pixels = size / 3
    rgb_flat = rgb.reshape((pixels, 3))
    cdef hsl_t[::1] hsl_flat
    hsl_flat = _rgb_to_husl_2d(rgb_flat, size)
    return np.asarray(hsl_flat, dtype=hsl_type).reshape(rgb.shape)


cdef hsl_t[::1] _rgb_to_husl_2d(np.uint8_t[:, ::1] rgb, int size):
    cdef hsl_t *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0], size)
    cdef hsl_t[::1] husl = <hsl_t[:size]> hsl_ptr
    return husl


"""Wrapper for _simd.c, the HUSL <-> RGB conversion  C implementation."""

import numpy as np
cimport numpy as np
import cython

from . import transform


data_type = np.float
ctypedef np.float_t data_type_t


cdef extern from "_simd.h":
    data_type_t* rgb_to_husl_nd(np.uint8_t *rgb, int size)


@transform.rgb_int_input
def _rgb_to_husl(rgb):
    cdef int size = rgb.size
    cdef int flat = len(rgb.shape) < 3
    cdef data_type_t[::1] hsl_flat
    if flat:
        hsl_flat = _rgb_to_husl_2d(rgb, size)
    else:
        hsl_flat = _rgb_to_husl_3d(rgb, size)
    return np.asarray(hsl_flat).reshape(rgb.shape)


cdef data_type_t[::1] _rgb_to_husl_3d(np.uint8_t[:, :, ::1] rgb, int size):
    cdef data_type_t *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0, 0], size)
    cdef data_type_t[::1] husl = <data_type_t[:size]> hsl_ptr
    return husl


cdef data_type_t[::1] _rgb_to_husl_2d(np.uint8_t[:, ::1] rgb, int size):
    cdef data_type_t *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0], size)
    cdef data_type_t[::1] husl = <data_type_t[:size]> hsl_ptr
    return husl


import numpy as np
cimport numpy as np
import cython


cdef extern from "_simd.h":
    ctypedef double double
    double *rgb_to_husl_nd(double*, int, int, int)


def rgb_to_husl(rgb):
    cdef int rows = rgb.shape[0]
    cdef int cols = rgb.shape[1]
    cdef int flat = len(rgb.shape) < 3
    cdef double[::1] hsl_flat
    if flat:
        hsl_flat = _rgb_to_husl_2d(rgb, rows, cols)
    else:
        hsl_flat = _rgb_to_husl_3d(rgb, rows, cols)
    return np.asarray(hsl_flat).reshape(rgb.shape)


cdef double[::1] _rgb_to_husl_3d(double[:, :, ::1] rgb, int rows, int cols):
    cdef double *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0, 0], rows, cols, 0)
    cdef double[::1] husl = <double[:rgb.size]> hsl_ptr
    return husl


cdef double[::1] _rgb_to_husl_2d(double[:, ::1] rgb, int rows, int cols):
    cdef double *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0], rows, cols, 1)
    cdef double[::1] husl = <double[:rgb.size]> hsl_ptr
    return husl


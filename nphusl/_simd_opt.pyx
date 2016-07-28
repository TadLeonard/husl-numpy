import numpy as np
cimport numpy as np
import cython


cdef extern from "_simd.h":
    double *rgb_to_husl_nd(double*, int, int)


def rgb_to_husl(rgb):
    cdef double[::1] hsl_flat
    if len(rgb.shape) == 3:
        hsl_flat = rgb_to_husl_3d(rgb)
    else:
        hsl_flat = rgb_to_husl_2d(rgb)
    return np.asarray(hsl_flat).reshape(rgb.shape)


cdef double[::1] rgb_to_husl_3d(double[:, :, ::1] rgb):
    cdef int rows = rgb.shape[0]
    cdef int cols = rgb.shape[1]

    cdef double *rgb_ptr = &rgb[0, 0, 0]
    cdef double *hsl_ptr = rgb_to_husl_nd(rgb_ptr, rows, cols)
    cdef double[::1] husl = <double[:rgb.size]> hsl_ptr
    return husl


cdef double[::1] rgb_to_husl_2d(double[:, ::1] rgb):
    cdef int rows = rgb.shape[0]
    cdef int cols = rgb.shape[1]

    cdef double *rgb_ptr = &rgb[0, 0]
    cdef double *hsl_ptr = rgb_to_husl_nd(rgb_ptr, rows, cols)
    cdef double[::1] husl = <double[:rgb.size]> hsl_ptr
    return husl

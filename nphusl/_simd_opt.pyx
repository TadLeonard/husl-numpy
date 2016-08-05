import numpy as np
cimport numpy as np
import cython


cdef extern from "_simd.h":
    double *rgb_to_husl_nd(np.uint8_t* rgb, int size)


def _rgb_to_husl(rgb):
    if rgb.dtype != np.uint8:
        print("CONVERTED")
        rgb = np.round(rgb * 255).astype(np.uint8)
    cdef int size = rgb.size
    cdef int flat = len(rgb.shape) < 3
    cdef double[::1] hsl_flat
    if flat:
        hsl_flat = _rgb_to_husl_2d(rgb, size)
    else:
        hsl_flat = _rgb_to_husl_3d(rgb, size)
    return np.asarray(hsl_flat).reshape(rgb.shape)


cdef double[::1] _rgb_to_husl_3d(np.uint8_t[:, :, ::1] rgb, int size):
    cdef double *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0, 0], size)
    cdef double[::1] husl = <double[:size]> hsl_ptr
    return husl


cdef double[::1] _rgb_to_husl_2d(np.uint8_t[:, ::1] rgb, int size):
    cdef double *hsl_ptr = rgb_to_husl_nd(&rgb[0, 0], size)
    cdef double[::1] husl = <double[:size]> hsl_ptr
    return husl


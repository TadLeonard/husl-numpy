import numpy as np
cimport numpy as np
import cython


cdef extern from "_simd.h":
    void rgb_to_husl_nd(double*, double*, int)


def rgb_to_husl(rgb):
    if len(rgb.shape) == 3:
        return np.asarray(rgb_to_husl_3d(rgb))
    else:
        return np.asarray(rgb_to_husl_2d(rgb))


cdef double[:, :, ::1] rgb_to_husl_3d(double[:, :, ::1] rgb):
    cdef int pixels = rgb.size / 3
    cdef double[:, :, ::1] husl
    husl = np.empty_like(rgb)
    rgb_to_husl_nd(&rgb[0, 0, 0], &husl[0, 0, 0], pixels)
    return husl


cdef double[:, ::1] rgb_to_husl_2d(double[:, ::1] rgb):
    cdef int pixels = rgb.size / 3
    cdef double[:, ::1] husl
    husl = np.empty_like(rgb)
    rgb_to_husl_nd(&rgb[0, 0], &husl[0, 0], pixels)
    return husl


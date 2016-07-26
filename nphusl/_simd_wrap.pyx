import numpy as np
cimport numpy as np
import cython


cdef extern from "_simd_opt.h":
    double to_light_c(double) nogil
    double to_linear_c(double) nogil
    void rgb_to_husl_3d_c(double*, double*, int, int)


def rgb_to_husl(rgb):
    if len(rgb.shape) == 3:
        return np.asarray(rgb_to_husl_3d(rgb))
    else:
        raise Exception("Not yet implemented in C")
        #return rgb_to_husl_2d(rgb)


cdef double[:, :, ::1] rgb_to_husl_3d(double[:, :, ::1] rgb):
    cdef Py_ssize_t size = rgb.size
    cdef Py_ssize_t rows = rgb.shape[0]
    cdef Py_ssize_t cols = rgb.shape[1]
    cdef double[:, :, ::1] husl
    husl = np.empty_like(rgb)
    rgb_to_husl_3d_c(&rgb[0, 0, 0], &husl[0, 0, 0], rows, cols)
    return husl



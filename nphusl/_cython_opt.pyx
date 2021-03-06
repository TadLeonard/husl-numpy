import numpy as np
cimport numpy as np
import cython

from . import constants
from . import transform
from cython.parallel import prange, parallel
from libc.math cimport sin, cos, M_PI, atan2, sqrt


cdef double[3][3] M = constants.M
cdef double[3][3] M_INV = constants.M_INV
cdef double REF_X = constants.REF_X
cdef double REF_Y = constants.REF_Y
cdef double REF_Z = constants.REF_Z
cdef double REF_U = constants.REF_U
cdef double REF_V = constants.REF_V
cdef double KAPPA = constants.KAPPA
cdef double EPSILON = constants.EPSILON


@transform.rgb_float_input
def _rgb_to_husl(rgb):
    is_3d = rgb.ndim == 3
    if is_3d:
        size = rgb.shape[0] * rgb.shape[1]
        rgb_2d = rgb.reshape((size, 3))
    else:
        rgb_2d = rgb
    husl = _rgb_to_husl_2d(rgb_2d)
    if is_3d:
        husl = husl.reshape(rgb.shape)
    return husl


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef np.ndarray[ndim=2, dtype=double] _rgb_to_husl_2d(
        np.ndarray[ndim=2, dtype=double] rgb):
    cdef int i
    cdef int rows = rgb.shape[0]
    cdef np.ndarray[ndim=2, dtype=double] husl = (
        np.zeros(dtype=float, shape=(rows, 3)))

    cdef double r, g, b
    cdef double x, y, z
    cdef double l, u, v
    cdef double var_u, var_v
    cdef double c, h, hrad, s

    for i in prange(rows, schedule="guided", nogil=True):
        # from linear RGB
        r = to_linear(rgb[i, 0])
        g = to_linear(rgb[i, 1])
        b = to_linear(rgb[i, 2])

        # to XYZ
        x = M_INV[0][0] * r + M_INV[0][1] * g + M_INV[0][2] * b
        y = M_INV[1][0] * r + M_INV[1][1] * g + M_INV[1][2] * b
        z = M_INV[2][0] * r + M_INV[2][1] * g + M_INV[2][2] * b

        # to LUV
        if x == y == z == 0:
            l = u = v = 0
        else:
            var_u = 4 * x / (x + 15 * y + 3 * z)
            var_v = 9 * y / (x + 15 * y + 3 * z)
            l = _to_light(y)
            u = 13 * l * (var_u - REF_U)
            v = 13 * l * (var_v - REF_V)

        # to LCH
        c = sqrt(u ** 2 + v ** 2)
        hrad = atan2(v, u)
        h = hrad * (180.0 / M_PI)
        if h < 0:
            h = h + 360

        # to HSL (finally!)
        if l > 99.99:
            s = 0
            l = 100
        elif l < 0.01:
            s = l = 0
        else:
            s = (c / max_chroma(l, h)) * 100.0
        husl[i, 0] = h
        husl[i, 1] = s
        husl[i, 2] = l

    return husl


@transform.rgb_float_input
def _rgb_to_hue(rgb):
    is_3d = rgb.ndim == 3
    if is_3d:
        size = rgb.shape[0] * rgb.shape[1]
        rgb_2d = rgb.reshape((size, 3))
    else:
        rgb_2d = rgb
    hue = _rgb_to_hue_2d(rgb_2d)
    if is_3d:
        hue = hue.reshape((rgb.shape[0], rgb.shape[1]))
    return hue


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef np.ndarray[ndim=1, dtype=double] _rgb_to_hue_2d(
        np.ndarray[ndim=2, dtype=double] rgb):
    cdef int i
    cdef int rows = rgb.shape[0]
    cdef np.ndarray[ndim=1, dtype=double] hue = (
        np.zeros(dtype=float, shape=(rows,)))

    cdef double r, g, b
    cdef double x, y, z
    cdef double l, u, v
    cdef double var_u, var_v
    cdef double c, h, hrad

    for i in prange(rows, schedule="guided", nogil=True):
        # from linear RGB
        r = to_linear(rgb[i, 0])
        g = to_linear(rgb[i, 1])
        b = to_linear(rgb[i, 2])

        # to XYZ
        x = M_INV[0][0] * r + M_INV[0][1] * g + M_INV[0][2] * b
        y = M_INV[1][0] * r + M_INV[1][1] * g + M_INV[1][2] * b
        z = M_INV[2][0] * r + M_INV[2][1] * g + M_INV[2][2] * b

        # to LUV
        if x == y == z == 0:
            l = u = v = 0
        else:
            var_u = 4 * x / (x + 15 * y + 3 * z)
            var_v = 9 * y / (x + 15 * y + 3 * z)
            l = _to_light(y)
            u = 13 * l * (var_u - REF_U)
            v = 13 * l * (var_v - REF_V)

        # to LCH
        c = sqrt(u ** 2 + v ** 2)
        hrad = atan2(v, u)
        h = hrad * (180.0 / M_PI)
        if h < 0:
            h = h + 360
        hue[i] = h

    return hue


@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _to_light(double y_value) nogil:
    if y_value > EPSILON:
        return 116 * (y_value / REF_Y) ** (1.0 / 3.0) - 16
    else:
        return (y_value / REF_Y) * KAPPA


@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double to_linear(double value) nogil:
    if value > 0.04045:
        return ((value + 0.055) / (1.0 + 0.055)) ** 2.4
    else:
        return value / 12.92


@transform.float_input
def _husl_to_rgb(hsl):
    is_3d = hsl.ndim == 3
    if is_3d:
        size = hsl.shape[0] * hsl.shape[1]
        hsl_2d = hsl.reshape((size, 3))
    else:
        hsl_2d = hsl
    rgb = _husl_to_rgb_2d(hsl_2d)
    if is_3d:
        rgb = rgb.reshape(hsl.shape)
    return rgb


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef np.ndarray[ndim=2, dtype=double] _husl_to_rgb_2d(
        np.ndarray[ndim=2, dtype=double] hsl):
    cdef int i, k
    cdef int rows = hsl.shape[0]
    cdef np.ndarray[ndim=2, dtype=double] rgb = (
        np.zeros(dtype=float, shape=(rows, 3)))

    cdef double h, s, l
    cdef double c
    cdef double u, v
    cdef double x, y, z
    cdef double hrad
    cdef double var_y, var_u, var_v

    for i in prange(rows, schedule="guided", nogil=True):
        # from HSL
        h = hsl[i, 0]
        s = hsl[i, 1]
        l = hsl[i, 2]

        # to LCH and LUV
        if l > 99.99:
            l = 100
            c = u = v = 0
        elif l < 0.01:
            l = c = u = v = 0
        else:
            c = max_chroma(l, h) / 100.0 * s
            hrad = h / 180.0 * M_PI
            u = cos(hrad) * c
            v = sin(hrad) * c

        # to XYZ
        if l == 0.0:
            x = y = z = 0.0
        else:
            if l > 8:
                var_y = REF_Y * ((l + 16.0) / 116.0) ** 3
            else:
                var_y = REF_Y * l / KAPPA
            var_u = u / (13.0 * l) + REF_U
            var_v = v / (13.0 * l) + REF_V
            y = var_y * REF_Y
            x = -(9.0 * y * var_u) / ((var_u - 4.0) * var_v - var_u * var_v)
            z = (9.0 * y - (15.0 * var_v * y) - (var_v * x)) / (3.0 * var_v)

        # to RGB (finally!)
        for k in range(3):
            rgb[i, k] = _from_linear(
                M[k][0] * x + M[k][1] * y + M[k][2] * z)

    return rgb


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _from_linear(double value) nogil:
    if value <= 0.0031308:
        return 12.92 * value
    else:
        return 1.055 * value ** (1.0 / 2.4) - 0.055


cpdef _grind_max_chroma(int n, double lightness, double hue):
    for _ in range(n):
        max_chroma(lightness, hue)


cpdef _test_max_chroma(double lightness, double hue):
    return max_chroma(lightness, hue)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double max_chroma(double lightness, double hue) nogil:
    """Find max chroma given an L, H pair"""
    cdef double sub1 = ((lightness + 16.0) ** 3) / 1560896.0
    cdef double sub2 = sub1 if sub1 > EPSILON else lightness / KAPPA
    cdef double top1
    cdef double top2
    cdef double top2_b
    cdef double bottom
    cdef double bottom_b
    cdef int i
    cdef double min_length
    min_length = 100000.0
    cdef double length1, length2
    cdef double m1, m2, b1, b2
    cdef double theta = hue / 360.0 * M_PI * 2.0
    cdef double sintheta = sin(theta)
    cdef double costheta = cos(theta)

    for i in range(3):
        top1 = (284517.0 * M[i][0] - 94839.0 * M[i][2]) * sub2
        top2 = ((838422.0 * M[i][2] + 769860.0 * M[i][1] + 731718.0 * M[i][0])
                * lightness * sub2)
        top2_b = top2 - (769860.0 * lightness)
        bottom = (632260.0 * M[i][2] - 126452.0 * M[i][1]) * sub2
        bottom_b = bottom + 126452.0
        m1 = top1 / bottom
        b1 = top2 / bottom
        length1 = b1 / (sintheta - m1 * costheta)
        if length1 < min_length:
            if length1 > 0:
                min_length = length1
        m2 = top1 / bottom_b
        b2 = top2_b / bottom_b
        length2 = b2 / (sintheta - m2 * costheta)
        if length2 < min_length:
            if length2 > 0:
                min_length = length2

    return min_length



import numpy as np
cimport numpy as np
import cython

from libc.math cimport sin, cos, M_PI

__version__ = "4.0.3"


cdef float[3][3] M = [
    [3.240969941904521, -1.537383177570093, -0.498610760293],
    [-0.96924363628087, 1.87596750150772, 0.041555057407175],
    [0.055630079696993, -0.20397695888897, 1.056971514242878]
]

cdef float[3][3] M_INV = [
    [0.41239079926595, 0.35758433938387, 0.18048078840183],
    [0.21263900587151, 0.71516867876775, 0.072192315360733],
    [0.019330818715591, 0.11919477979462, 0.95053215224966],
]

cdef float REF_X = 0.95045592705167
cdef float REF_Y = 1.0
cdef float REF_Z = 1.089057750759878
cdef float REF_U = 0.19783000664283
cdef float REF_V = 0.46831999493879
cdef float KAPPA = 903.2962962
cdef float EPSILON = 0.0088564516


def _test_husl_to_rgb(husl):
    cdef np.ndarray rgb = husl_to_rgb(husl)
    return rgb

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef np.ndarray[ndim=3, dtype=double] husl_to_rgb(
        np.ndarray[ndim=3, dtype=double] hsl):
    cdef int i, j, k
    cdef int rows = hsl.shape[0]
    cdef int cols = hsl.shape[1]
    cdef np.ndarray[ndim=3, dtype=double] rgb = (
        np.zeros(dtype=np.float, shape=(rows, cols, 3)))

    cdef float mc, chroma
    cdef float h, s, l
    cdef float c
    cdef float u, v
    cdef float hrad
    cdef float var_y, var_u, var_v

    for i in range(rows):
        for j in range(cols):
            # from HSL
            h = hsl[i, j, 0]
            s = hsl[i, j, 1]
            l = hsl[i, j, 2]

            # to LCH
            if l > 99.999:
                l = 100
                c = 0
            elif l < 0.0001:
                l = 0
                c = 0
            else:
                mc = max_chroma(l, h)
                c = mc / 100.0 * s

            # to LUV
            hrad = h / 180.0 * M_PI
            u = cos(hrad) * c
            v = sin(hrad) * c

            # to XYZ
            if l == 0:
                x = y = z = 0
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
                rgb[i, j, k] = _from_linear(
                    M[k][0] * x + M[k][1] * y + M[k][2] * z)

    return rgb


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _from_linear(double value):
    if value <= 0.0031308:
        return 12.92 * value
    else:
        return 1.055 * value ** (1.0/2.4) - 0.055


cpdef _grind_max_chroma(int n, float lightness, float hue):
    for _ in range(n):
        max_chroma(lightness, hue)


cpdef _test_max_chroma(float lightness, float hue):
    return max_chroma(lightness, hue)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float max_chroma(float lightness, float hue):
    """Find max chroma given an L, H pair"""
    cdef float sub1 = ((lightness + 16.0) ** 3) / 1560896.0
    cdef float sub2 = sub1 if sub1 > EPSILON else lightness / KAPPA
    cdef float top1
    cdef float top2
    cdef float top2_b
    cdef float bottom
    cdef float bottom_b
    cdef int i
    cdef float min_length
    min_length = 100000.0
    cdef float length1, length2
    cdef float m1, m2, b1, b2
    cdef float theta = hue / 360.0 * M_PI * 2.0
    cdef float sintheta = sin(theta)
    cdef float costheta = cos(theta)

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



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
cdef float REV_U = 0.19783000664283
cdef float REF_V = 0.46831999493879
cdef float KAPPA = 903.2962962
cdef float EPSILON = 0.0088564516



cdef inline husl_to_lch(float hue, float saturation, float lightness):
    


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



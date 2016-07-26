
#include <math.h>
#include <stdio.h>


const double M[3][3] = {
    {3.240969941904521, -1.537383177570093, -0.498610760293},
    {-0.96924363628087, 1.87596750150772, 0.041555057407175},
    {0.055630079696993, -0.20397695888897, 1.056971514242878}
};

const double M_INV[3][3] = {
    {0.41239079926595, 0.35758433938387, 0.18048078840183},
    {0.21263900587151, 0.71516867876775, 0.072192315360733},
    {0.019330818715591, 0.11919477979462, 0.95053215224966},
};

const double REF_X = 0.95045592705167;
const double REF_Y = 1.0;
const double REF_Z = 1.089057750759878;
const double REF_U = 0.19783000664283;
const double REF_V = 0.46831999493879;
const double KAPPA = 903.2962962;
const double EPSILON = 0.0088564516;


inline double to_light_c(double y_value) {
    if (y_value > EPSILON) {
        return pow(116 * (y_value / REF_Y), 1.0 / 3.0) - 16;
    }
    else {
        return (y_value / REF_Y) * KAPPA;
    }
}

inline double to_linear_c(double value) {
    if (value > 0.04045) {
        return pow((value + 0.055) / (1.0 + 0.055), 2.4);
    }
    else {
        return value / 12.92;
    }
}


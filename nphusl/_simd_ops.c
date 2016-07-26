
#include <math.h>
#include <stdio.h>


static double max_chroma(double, double);
double to_light_c(double);
double to_linear_c(double);

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
        return 116 * pow((y_value / REF_Y), 1.0 / 3.0) - 16;
    } else {
        return (y_value / REF_Y) * KAPPA;
    }
}

inline double to_linear_c(double value) {
    if (value > 0.04045) {
        return pow((value + 0.055) / (1.0 + 0.055), 2.4);
    } else {
        return value / 12.92;
    }
}

void rgb_to_husl_3d_c(double *rgb, double *hsl, int rows, int cols) {
    int i, j, idx;
    double r, g, b;
    double x, y, z;
    double l, u, v;
    double var_u, var_v;
    double c, h, hrad, s;
    idx = 0;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            // from linear RGB
            r = to_linear_c(rgb[idx]);
            g = to_linear_c(rgb[idx + 1]);
            b = to_linear_c(rgb[idx + 2]);

            // to XYZ
            x = M_INV[0][0] * r + M_INV[0][1] * g + M_INV[0][2] * b;
            y = M_INV[1][0] * r + M_INV[1][1] * g + M_INV[1][2] * b;
            z = M_INV[2][0] * r + M_INV[2][1] * g + M_INV[2][2] * b;

            // to LUV
            if (x == 0 && y == 0 && z == 0) {
                l = u = v = 0;
            } else {
                var_u = 4 * x / (x + 15 * y + 3 * z);
                var_v = 9 * y / (x + 15 * y + 3 * z);
                l = to_light_c(y);
                u = 13 * l * (var_u - REF_U);
                v = 13 * l * (var_v - REF_V);
            }

            // to LCH
            c = sqrt(pow(u, 2) + pow(v, 2));
            hrad = atan2(v, u);
            h = hrad * (180.0 / M_PI);
            if (h < 0) {
                h = h + 360;
            }

            // to HSL (finally!)
            if (l > 99.99) {
                s = 0;
                l = 100;
            } else if (l < 0.01) {
                //s = l = 0;
                s = 0;
                l = 0;
            } else {
                s = (c / max_chroma(l, h)) * 100.0;
            }
            hsl[idx] = h;
            hsl[idx + 1] = s;
            hsl[idx + 2] = l;
            idx += 3;
        }
    }
}

     
/*
Find max chroma given an L, H pair
*/
static double max_chroma(double lightness, double hue) {
    double sub1 = pow((lightness + 16.0), 3) / 1560896.0;
    double sub2;
    if (sub1 > EPSILON) {
        sub2 = sub1;
    } else {
        sub2 = lightness / KAPPA;
    }
    double top1;
    double top2;
    double top2_b;
    double bottom;
    double bottom_b;
    int i;
    double min_length = 100000.0;
    double length1, length2;
    double m1, m2, b1, b2;
    double theta = hue / 360.0 * M_PI * 2.0;
    double sintheta = sin(theta);
    double costheta = cos(theta);

    for (i = 0; i < 3; i++) {
        top1 = (284517.0 * M[i][0] - 94839.0 * M[i][2]) * sub2;
        top2 = ((838422.0 * M[i][2] + 769860.0 * M[i][1] + 731718.0 * M[i][0])
                * lightness * sub2);
        top2_b = top2 - (769860.0 * lightness);
        bottom = (632260.0 * M[i][2] - 126452.0 * M[i][1]) * sub2;
        bottom_b = bottom + 126452.0;

        m1 = top1 / bottom;
        b1 = top2 / bottom;
        length1 = b1 / (sintheta - m1 * costheta);
        if (length1 < min_length) {
            if (length1 > 0) {
                min_length = length1;
            }
        }

        m2 = top1 / bottom_b;
        b2 = top2_b / bottom_b;
        length2 = b2 / (sintheta - m2 * costheta);
        if (length2 < min_length) {
            if (length2 > 0) {
                min_length = length2;
            }
        }
    }

    return min_length;
}


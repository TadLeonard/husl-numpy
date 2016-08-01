
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <_linear_lookup.h>
#include <_light_lookup.h>
#include <_scale_const.h>


typedef unsigned char uint8;

static double max_chroma(double, double);
static double to_light(double);
double* rgb_to_husl_nd(double*, int, int);
static double min_chroma_length(
    int iteration, double lightness, double sub1, double sub2,
    double top2, double top2_b, double sintheta, double costheta);
static double to_hue_degrees(double, double);
static double to_saturation(double, double, double, double);

static const double WHITE_LIGHTNESS = 100.0;
static const double WHITE_HUE = 19.916405993809086;

/* RGB -> HUSL conversion
Converts an array of c-contiguous RGB doubles to an array of c-contiguous
HSL doubles. RGB doubles should be in the range [0,1].
*/
double* rgb_to_husl_nd(double *rgb, int rows, int cols) {
    int pixels = rows * cols;
    int size = pixels * 3;

    // HUSL array of H, S, L triplets to be returned
    double *hsl = (double*) calloc(size, sizeof(double));
    if (hsl == NULL) {
        fprintf(stderr, "Error: Couldn't allocate memory for HUSL array\n");
        exit(EXIT_FAILURE);
    }

    int i;
    double r, g, b;
    double x, y, z;
    double l, u, v;
    double var_u, var_v;
    double h, s;

    /* OpenMP parallel loop.
    default(none) is used so that all shared and private variables
    must be marked explicitly
    */
    #pragma omp parallel \
        default(none) \
        private(i, r, g, b, x, y, z, l, u, v, h, s, var_u, var_v) \
        shared(rgb, hsl, size)
    { // begin parallel

    #pragma omp for simd schedule(guided)
    for (i = 0; i < size; i+=3) {
        // to linear RGB
        r = rgb[i];
        g = rgb[i+1];
        b = rgb[i+2];

        // process pixel extremes
        if (!(r || g || b)) {
            // black pixels are {0, 0, 0} in HUSL
            continue;
        } else if (r == 1 && g == 1 && b == 1) {
            // white pixels are {19.916, 0, 100} in HUSL
            // the odd 19.916 hue is not meaningful in a white pixel,
            // but it's helpful to have this for unit testing
            hsl[i] = WHITE_HUE;
            hsl[i+2] = WHITE_LIGHTNESS;
            continue;
        }

        // to linear RGB
        r = linear_table[(uint8) (r*255)];
        g = linear_table[(uint8) (g*255)];
        b = linear_table[(uint8) (b*255)];

        // to XYZ
        x = M_INV[0][0]*r + M_INV[0][1]*g + M_INV[0][2]*b;
        y = M_INV[1][0]*r + M_INV[1][1]*g + M_INV[1][2]*b;
        z = M_INV[2][0]*r + M_INV[2][1]*g + M_INV[2][2]*b;

        // to LUV
        var_u = 4*x / (x + 15*y + 3*z);
        var_v = 9*y / (x + 15*y + 3*z);
        l = to_light(y);
        u = 13*l * (var_u - REF_U);
        v = 13*l * (var_v - REF_V);

        // to LCH to HSL
        h = to_hue_degrees(v, u);
        if (h < 0) {
            h += 360;  // negative angles wrap around into higher hues
        }
        s = to_saturation(u, v, l, h);
        hsl[i] = h;
        hsl[i+1] = s;
        hsl[i+2] = l;

    } // end OMP for
    } // end OMP parallel

    return hsl;
}


// Returns a saturation value.
// Saturation magnitude is found via sqrt(U**2 + V**2),
// then it's normalized by the max chroma (dictated by H and L)
static inline double to_saturation(double u, double v, double l, double h) {
    return 100*sqrt(pow(u, 2) + pow(v, 2)) / max_chroma(l, h);
}


static const double DEG_PER_RAD = 180.0 / M_PI;

// Returns the angle, in degrees, between the V and U values
// of the LUV color space. Results in a HUSL hue that may be negative.
// Negative hues should 'wrap around' to 360.
static inline double to_hue_degrees(double v_value, double u_value) {
    return atan2(v_value, u_value) * DEG_PER_RAD;
}


// Returns max chroma given an L, H pair.
static double max_chroma(double lightness, double hue) {
    double sub1 = pow((lightness + 16.0), 3) / 1560896.0;
    double sub2 = sub1 > EPSILON ? sub1 : lightness / KAPPA;
    double top2 = SCALE_SUB2 * lightness * sub2;
    double top2_b = top2 - (769860.0 * lightness);
    double theta = hue / 360.0 * M_PI * 2.0;
    double sintheta = sin(theta);
    double costheta = cos(theta);
    double len0, len1, len2;
    len0 = min_chroma_length(
        0, lightness, sub1, sub2,
        top2, top2_b, sintheta, costheta);
    len1 = min_chroma_length(
        1, lightness, sub1, sub2,
        top2, top2_b, sintheta, costheta);
    len2 = min_chroma_length(
        2, lightness, sub1, sub2,
        top2, top2_b, sintheta, costheta);
    return fmin(fmin(len0, len1), len2);
}


static inline double min_chroma_length(
        int iteration, double lightness, double sub1, double sub2,
        double top2, double top2_b, double sintheta, double costheta) {
    double top1 = SCALE_SUB1[iteration] * sub2;
    double bottom = SCALE_BOTTOM[iteration] * sub2;
    double bottom_b = bottom + 126452.0;
    double min_length = 10000.0;
    double len;

    len = (top2 / bottom) / (sintheta - (top1 / bottom) * costheta);
    min_length = len > 0 ? len : min_length;
    len = (top2_b / bottom_b) / (sintheta - (top1 / bottom_b) * costheta);
    min_length = len > 0 ? fmin(len, min_length) : min_length;
    return min_length;
}


// Define a function that takes the Y of the XYZ space
// and returns a lightness value. If -DUSE_LIGHT_LUT
// or -DUSE_MULTI_LIGHT_LUT are passed to GCC,
// a faster-but-less-accurate lookup table approach will be used.

#if defined(USE_LIGHT_LUT)
static double to_light(double y_value) {
    unsigned short l_idx;
    if (y_value < y_thresh_0) {
        l_idx = y_value/y_idx_step_0 + 0.5;
    } else if (y_value < y_thresh_1) {
        l_idx = ((y_value - y_thresh_0)/y_idx_step_1 + 0.5) + L_TABLE_SIZE;
    } else {
        l_idx = ((y_value - y_thresh_1)/y_idx_step_2 + 0.5) + L_TABLE_SIZE*2;
    }
    return big_light_table[l_idx];
}

#elif defined(USE_MULTI_LIGHT_LUT)
static inline double to_light(double y_value) {
    unsigned short l_idx;
    if (y_value < y_thresh_0) {
        l_idx = (y_value*L_TABLE_SIZE + 1.5);
        return light_table_0[l_idx];
    } else if (y_value < y_thresh_1) {
        l_idx = ((y_value-y_thresh_0)*L_TABLE_SIZE + 1.5);
        return light_table_1[l_idx];
    } else {
        l_idx = ((y_value-y_thresh_1)*L_TABLE_SIZE + 1.5);
        return light_table_2[l_idx];
    }
}

#else
static inline double to_light(double y_value) {
    if (y_value > EPSILON) {
        return 116 * pow((y_value / REF_Y), 1.0 / 3.0) - 16;
    } else {
        return (y_value / REF_Y) * KAPPA;
    }
}

#endif


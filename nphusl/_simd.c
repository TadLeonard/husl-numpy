
#include <math.h>
#include <omp.h>
#include <stdlib.h>
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

static const double WHITE_LIGHTNESS = 100.0;
static const double WHITE_HUE = 19.916405993809086;

/* RGB -> HUSL conversion
Converts an array of c-contiguous RGB doubles to an array of c-contiguous
HSL doubles. RGB doubles should be in the range [0,1].
*/
double* rgb_to_husl_nd(double *rgb, int rows, int cols) {
    int pixels = rows * cols;
    int size = pixels * 3;
    double *hsl = (double*) calloc(size, sizeof(double));
    //double *linear_rgb = (double*) malloc(size * sizeof(double));
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

    #pragma omp for schedule(guided)
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
    
        if (y < light_step_0) {
            l = light_table_0[(unsigned short) (y * L_TABLE_SIZE + 0.5)];
        } else if (y < light_step_1) {
            l = light_table_1[(unsigned short) ((y-light_step_1) * L_TABLE_SIZE + 0.5)];
        } else {
            l = light_table_2[(unsigned short) ((y-light_step_1) * L_TABLE_SIZE + 0.5)];
        }
          
        //printf("%d: %f\n", \
               (unsigned short) (y * 4096), \
               light_table[(unsigned int) (y * 
        u = 13*l * (var_u - REF_U);
        v = 13*l * (var_v - REF_V);

        // to LCH
        h = atan2(v, u) * (180.0 / M_PI);
        if (h < 0) {
            h += 360.0;
        }
        s = 100 * sqrt(pow(u, 2) + pow(v, 2)) / max_chroma(l, h);
        hsl[i] = h;
        hsl[i+1] = s > 100.1 ? 0.0: s;
        hsl[i+2] = l;

    } // end OMP for
    } // end OMP parallel

    return hsl;
}


/*
Find max chroma given an L, H pair.
*/
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


static inline double to_light(double y_value) {
    if (y_value > EPSILON) {
        return 116 * pow((y_value / REF_Y), 1.0 / 3.0) - 16;
    } else {
        return (y_value / REF_Y) * KAPPA;
    }
}


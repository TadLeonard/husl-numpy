// HUSL color space conversion with OpenMP+SIMD
//
// Important functions:
// 1) rgb_to_husl_nd: RGB -> HUSL
// 2) husl_to_rgb_nd: HUSL -> RGB
// 3) rgb_to_hue_nd: RGB -> HUSL hue
// 4) rgb_to_lightness_nd: RGB -> HUSL lightness

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>
#include <_linear_lookup.h>
#include <_scale_const.h>


typedef unsigned char uint8;


static double max_chroma(double, double);
static double to_light(double);
static double to_hue_degrees(double, double);
static double to_saturation(double, double, double, double);


#if defined(USE_LIGHT_LUT)
#include <_light_lookup.h>
static double interpolate_light(double, double, double);
#endif


#if defined(USE_CHROMA_LUT)
#include <_chroma_lookup.h>
static double interpolate_chroma(double, double, double);
#else
static double min_chroma_length(
    int iteration, double lightness, double sub1, double sub2,
    double top2, double top2_b, double sintheta, double costheta);
#endif


// Constants for (1, 1, 1) pixels
static const double WHITE_LIGHTNESS = 100.0;
static const double WHITE_HUE = 19.916405993809086;


// RGB -> HUSL conversion
// Converts an array of c-contiguous RGB doubles to an array of c-contiguous
// HSL doubles. RGB doubles should be in the range [0,1].
double* rgb_to_husl_nd(double *rgb, int size) {
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
    double var_u, var_v, var_scale;
    double h, s;

    // OpenMP parallel loop.
    // default(none) is used so that all shared and private variables
    // must be marked explicitly
    #pragma omp parallel \
        default(none) \
        shared(rgb, hsl, size) \
        private(i, r, g, b, x, y, z, l, u, v, \
                h, s, var_u, var_v, var_scale)

    { // begin parallel
    #pragma omp for simd linear(i:3) schedule(guided)
    for (i = 0; i < size; i+=3) {
        // to linear RGB
        r = rgb[i];
        g = rgb[i+1];
        b = rgb[i+2];

        // process color extremes
        /*
        if (!(r || g || b)) {
            // black pixels are {0, 0, 0} in HUSL
            continue;
        } else if (r == 1 && g == 1 && b == 1) {
            // white pixels are {19.916, 0, 100} in HUSL
            // the weird 19.916 hue value is not meaningful in a white pixel,
            // but it's helpful to have this for unit testing
            hsl[i] = WHITE_HUE;
            hsl[i+2] = WHITE_LIGHTNESS;
            continue;
        }
        */

        // to linear RGB
        r = linear_table[(uint8) (r*255)];
        g = linear_table[(uint8) (g*255)];
        b = linear_table[(uint8) (b*255)];

        // to CIE XYZ
        x = M_INV[0][0]*r + M_INV[0][1]*g + M_INV[0][2]*b;
        y = M_INV[1][0]*r + M_INV[1][1]*g + M_INV[1][2]*b;
        z = M_INV[2][0]*r + M_INV[2][1]*g + M_INV[2][2]*b;

        // to CIE LUV
        var_scale = x + 15*y + 3*z;
        var_u = 4*x / var_scale;
        var_v = 9*y / var_scale;
        l = to_light(y);
        u = 13*l * (var_u - REF_U);
        v = 13*l * (var_v - REF_V);

        // to CIE LCH, then finally to HUSL!
        h = to_hue_degrees(v, u);
        s = to_saturation(u, v, l, h);
        hsl[i] = h;
        hsl[i+1] = s;
        hsl[i+2] = l;
    } // end OMP for
    } // end OMP parallel

    return hsl;
}


// Returns the HUSL hue.
// This is the angle, in degrees, between VU (of CIELUV color space).
#pragma omp declare simd
static inline double to_hue_degrees(double v_value, double u_value) {
    const double DEG_PER_RAD = 180.0 / M_PI;
    double hue = atan2f(v_value, u_value) * DEG_PER_RAD;
    if (hue < 0.0f) {
        hue += 360;  // negative angles wrap around into higher hues
    }
    return hue;
}


// Returns a saturation value from UV (of CIELUV), lightness, and hue.
// Saturation magnitude (hypotenuse b/t U & V) is found via sqrt(U**2 + V**2),
// then it's normalized by the max chroma, which is dictated by H and L.
#pragma omp declare simd
static inline double to_saturation(double u, double v, double l, double h) {
    return 100*sqrt(u*u + v*v) / max_chroma(l, h);
}


#if defined(USE_CHROMA_LUT)
// Returns a maximum chroma value  given an L, H pair.
// Uses the chroma lookup table to perform bilinear interpolation.
// This LUT approach is important, because finding the max chroma is
// the most expensive operation in RGB -> HUSL conversion.
//
// Reference (see Unit Square section):
// https://en.wikipedia.org/wiki/Bilinear_interpolation
#pragma omp declare simd
static double max_chroma(double lightness, double hue) {
    // Compute H-value indices (axis 0) and L-value indices (axis 1)
    double h_idx = hue / h_idx_step;
    double l_idx = lightness / l_idx_step;
    unsigned short h_idx_floor = floor(h_idx); 
    unsigned short l_idx_floor = floor(l_idx);
    h_idx_floor = fmax(0, fmin(C_TABLE_SIZE-2, h_idx_floor));
    l_idx_floor = fmax(0, fmin(C_TABLE_SIZE-2, l_idx_floor));

    // Find four known f() values in the unit square bilinear interp. approach
    double chroma_00 = chroma_table[h_idx_floor][l_idx_floor];
    double chroma_10 = chroma_table[h_idx_floor+1][l_idx_floor];
    double chroma_01 = chroma_table[h_idx_floor][l_idx_floor+1];
    double chroma_11 = chroma_table[h_idx_floor+1][l_idx_floor+1];

    // Find *normalized* x, y, (1-x), and (1-y) values
    // It's a coordinate system where the four known chromas are at
    // (0,0), (1,0), (0,1), and (1,1), so we normalize x and y.
    double h_norm = h_idx - h_idx_floor;  // our "x" value
    double l_norm = l_idx - l_idx_floor;  // our "y" value
    double h_inv = 1 - h_norm;  // (1-x)
    double l_inv = 1 - l_norm;  // (1-y)

    // Compute f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
    double interp = chroma_00*h_inv*l_inv +
                    chroma_10*h_norm*l_inv +
                    chroma_01*h_inv*l_norm +
                    chroma_11*h_norm*l_norm;
    return interp;
} 

static inline double interpolate_chroma(
        double val_1, double val_2, double delta_idx) {
    double val_lo = fmin(val_1, val_2);
    return val_lo + delta_idx*(fabs(val_2 - val_2));
}
    
#else
// Returns max chroma given an L, H pair.
// Very expensive operation.
#pragma omp declare simd
static double max_chroma(double lightness, double hue) {
    double sub1 = pow(lightness + 16.0, 3) / 1560896.0;
    double sub2 = sub1 > EPSILON ? sub1 : lightness / KAPPA;
    double top2 = SCALE_SUB2 * lightness * sub2;
    double top2_b = top2 - 769860.0*lightness;
    double theta = hue / 360.0 * M_PI * 2.0;  // hue in radians
    double sintheta = sinf(theta);
    double costheta = cosf(theta);
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


// Returns a min chroma "length" in the HSL space from H and L.
// The HUSL color space is basically CIELUV, but
// it overcomes its floating chroma value by "stretching"
// the color space so that a new channel, "saturation"
// is a percentage in [0, 100] for all possible values of hue and lightness.
// The images at husl-colors.org explain this more clearly!
#pragma omp declare simd
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
#endif


// Define a function that takes the Y of the XYZ space
// and returns a lightness value. If -DUSE_LIGHT_LUT is passed to GCC,
// a faster-but-less-accurate lookup table approach will be used.

#if defined(USE_LIGHT_LUT)
// Return a light value from a CIEXYZ Y-value.
// A light value lookup that accounts for a lack of precision
// at low Y values. The lookup table is combined from three smaller
// tables, each with a different Y-value to L-value scale.
static double to_light(double y_value) {
    double idx;
    double light_hi, light_lo;
    unsigned short idx_floor;
    if (y_value < y_thresh_0) {
        idx = y_value/y_idx_step_0;
    } else if (y_value < y_thresh_1) {
        idx = ((y_value - y_thresh_0)/y_idx_step_1) + L_SEGMENT_SIZE;
    } else {
        idx = ((y_value - y_thresh_1)/y_idx_step_2) + L_SEGMENT_SIZE*2;
    }
    idx_floor = floor(idx);
    idx_floor = fmax(0, fmin(L_FULL_TABLE_SIZE-2, idx_floor));
    light_lo = light_table_big[idx_floor];
    light_hi = light_table_big[idx_floor+1];
    return interpolate_light(light_hi, light_lo, idx-idx_floor);
}

// Linear interpolation. Allows for smaller, more cache-friendly light tables.
static inline double interpolate_light(
        double light_lo, double light_hi, double delta_idx) {
    return light_lo + delta_idx*(light_hi - light_lo);
}

#else
// Return a light value from a CIEXYZ Y-value.
// Uses an expensive branch/pow().
static inline double to_light(double y_value) {
    if (y_value > EPSILON) {
        return 116 * pow((y_value / REF_Y), 1.0 / 3.0) - 16;
    } else {
        return (y_value / REF_Y) * KAPPA;
    }
}

#endif


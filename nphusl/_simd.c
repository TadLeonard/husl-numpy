// HUSL color space conversion with OpenMP+SIMD
//
// Important functions:
// 1) rgb_to_husl_nd: RGB -> HUSL
// 2) husl_to_rgb_nd: HUSL -> RGB
// 3) rgb_to_hue_nd: RGB -> HUSL hue
// 4) rgb_to_lightness_nd: RGB -> HUSL lightness

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <_simd.h>
#include <_linear_lookup.h>
#include <_light_lookup.h>
#include <_scale_const.h>


typedef unsigned char uint8;


static simd_t max_chroma(simd_t, simd_t);
static simd_t to_light(simd_t);
static simd_t min_chroma_length(
    int iteration, simd_t lightness, simd_t sub1, simd_t sub2,
    simd_t top2, simd_t top2_b, simd_t sintheta, simd_t costheta);
static simd_t to_hue_degrees(simd_t, simd_t);
static simd_t to_saturation(simd_t, simd_t, simd_t, simd_t);
static simd_t interpolate_light(simd_t, simd_t, simd_t);


static const simd_t WHITE_LIGHTNESS = 100.0;
static const simd_t WHITE_HUE = 19.916405993809086;


// RGB -> HUSL conversion
// Converts an array of c-contiguous RGB simd_ts to an array of c-contiguous
// HSL simd_ts. RGB simd_ts should be in the range [0,1].
double* rgb_to_husl_nd(double *rgb, int rows, int cols, int is_flat) {
    int pixels = rows * cols;
    int size = pixels;
    if (!is_flat) {
        size *= 3;
    }

    // HUSL array of H, S, L triplets to be returned
    double *hsl = (double*) calloc(size, sizeof(double));
    if (hsl == NULL) {
        fprintf(stderr, "Error: Couldn't allocate memory for HUSL array\n");
        exit(EXIT_FAILURE);
    }

    int i;
    simd_t r, g, b;
    simd_t x, y, z;
    simd_t l, u, v;
    simd_t var_u, var_v, var_scale;
    simd_t h, s;

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


// Returns a saturation value from UV (of CIELUV), lightness, and hue.
// Saturation magnitude (hypotenuse) is found via sqrt(U**2 + V**2),
// then it's normalized by the max chroma, which is dictated by H and L
#pragma omp declare simd
static inline simd_t to_saturation(simd_t u, simd_t v, simd_t l, simd_t h) {
    return 100*sqrt(u*u + v*v) / max_chroma(l, h);
}


// Returns the HUSL hue.
// This is the angle, in degrees, between VU (of CIELUV color space).
#pragma omp declare simd
static inline simd_t to_hue_degrees(simd_t v_value, simd_t u_value) {
    const simd_t DEG_PER_RAD = 180.0 / M_PI;
    simd_t hue = atan2f(v_value, u_value) * DEG_PER_RAD;
    if (hue < 0.0f) {
        hue += 360;  // negative angles wrap around into higher hues
    }
    return hue;
}


// Returns max chroma given an L, H pair.
#pragma omp declare simd
static simd_t max_chroma(simd_t lightness, simd_t hue) {
    simd_t sub1 = pow(lightness + 16.0, 3) / 1560896.0;
    simd_t sub2 = sub1 > EPSILON ? sub1 : lightness / KAPPA;
    simd_t top2 = SCALE_SUB2 * lightness * sub2;
    simd_t top2_b = top2 - 769860.0*lightness;
    simd_t theta = hue / 360.0 * M_PI * 2.0;  // hue in radians
    simd_t sintheta = sinf(theta);  // sinf: precision isn't critical
    simd_t costheta = cosf(theta);  // cosf: precision isn't critical
    simd_t len0, len1, len2;
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
static inline simd_t min_chroma_length(
        int iteration, simd_t lightness, simd_t sub1, simd_t sub2,
        simd_t top2, simd_t top2_b, simd_t sintheta, simd_t costheta) {
    simd_t top1 = SCALE_SUB1[iteration] * sub2;
    simd_t bottom = SCALE_BOTTOM[iteration] * sub2;
    simd_t bottom_b = bottom + 126452.0;
    simd_t min_length = 10000.0;
    simd_t len;

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

// Linear interpolation. Allows for smaller, more cache-friendly light tables.
static inline simd_t interpolate_light(
        simd_t light_lo, simd_t light_hi, simd_t delta_idx) {
    return light_lo + delta_idx*(light_hi - light_lo);
}

#if defined(USE_SEGMENTED_LIGHT_LUT)
// A light value lookup that accounts for a lack of precision
// at low Y values. The lookup table is combined from three smaller
// tables, each with a different Y-value to L-value scale.
static simd_t to_light(simd_t y_value) {
    simd_t idx;
    simd_t light_hi, light_lo;
    unsigned short idx_floor;
    if (y_value < y_thresh_0) {
        idx = y_value/y_idx_step_0;
    } else if (y_value < y_thresh_1) {
        idx = ((y_value - y_thresh_0)/y_idx_step_1) + L_TABLE_SIZE;
    } else {
        idx = ((y_value - y_thresh_1)/y_idx_step_2) + L_TABLE_SIZE*2;
    }
    idx_floor = floor(idx);
    light_lo = light_table_big[idx_floor];
    light_hi = light_table_big[idx_floor+1];
    return interpolate_light(light_hi, light_lo, idx-idx_floor);
}

#elif defined(USE_LINEAR_LIGHT_LUT)
// Use a simpler linear lookup table that may be less precise
// for small Y-values.
static simd_t to_light(simd_t y_value) {
    simd_t idx;
    simd_t light_hi, light_lo;
    unsigned short idx_floor;
    idx = y_value*y_idx_step_linear;
    idx_floor = floor(idx);
    light_lo = light_table_linear[idx_floor];
    light_hi = light_table_linear[idx_floor + 1];
    return interpolate_light(light_hi, light_lo, idx-idx_floor);
}

#else
static inline simd_t to_light(simd_t y_value) {
    if (y_value > EPSILON) {
        return 116 * pow((y_value / REF_Y), 1.0 / 3.0) - 16;
    } else {
        return (y_value / REF_Y) * KAPPA;
    }
}

#endif


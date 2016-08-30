// HUSL color space conversion with OpenMP
//
// Important functions:
// 1) rgb_to_husl_nd: RGB -> HUSL
// 2) husl_to_rgb_nd: HUSL -> RGB
// 3) rgb_to_hue_nd: RGB -> HUSL hue
// 4) rgb_to_lightness_nd: RGB -> HUSL lightness


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
 

#ifdef _OPENMP
// Min array size for OpenMP parallelized loops
#define MIN_IMG_SIZE_THREADED 30*30*3
#include <omp.h>
#endif


#include <_simd.h>
#include <_linear_lookup.h>
#include <_scale_const.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


////////////////////////////////////////////
// Conversion in the RGB -> HUSL direction
////////////////////////////////////////////


static double *allocate_hsl(size_t size);
static void rgb_to_luv_nd(uint8_t *rgb, double *luv, size_t size);
static void rgbluv_to_husl_nd(uint8_t *rgb, double *luv_hsl, size_t size);
static void to_linear_rgb(uint8_t r, uint8_t g, uint8_t b,
                          double *rl, double *gl, double *bl);
static void to_xyz(double r, double g, double b,
                   double *x, double *y, double *z);
static void to_luv(double x, double y, double z,
                   double *l, double *u, double *v);
static double to_light(double);
static double to_hue(double u, double v);
static double to_saturation(double, double, double, double);
static double max_chroma(double, double);


// Enable luminance lookup interpolation based on compile flag
#if defined(USE_LIGHT_LUT)
#include <_light_lookup.h>
#endif


// Choose CIE-LUV -> LCH croma function based on compile flag
#if defined(USE_CHROMA_LUT)
#include <_chroma_lookup.h>
static double linear_interp_chroma(double, double, double);
#else
#define CHROMA_SCALE 1
static double min_chroma_length(
    int iteration, double lightness, double sub1, double sub2,
    double top2, double top2_b, double sintheta, double costheta);
#endif


// Choose CIE-LUV -> Hue function based on compile flag
#ifdef USE_HUE_ATAN2_APPROX
static double atan2_approx(double u, double v);
static double atan_approx(double z);
#endif


// Constant HUSL H, S, and L for white pixels
static const double WHITE_HUE = 19.916405993809086;
static const double WHITE_SATURATION = 0.0;
static const double WHITE_LIGHTNESS = 100.0;


// RGB -> HUSL conversion
// Converts an array of c-contiguous RGB ints to an array of c-contiguous
// HSL doubles. RGB ints should be in the interval [0, 255]
double* rgb_to_husl_nd(uint8_t *restrict rgb, size_t size) {
    double *hsl = allocate_hsl(size);  // HUSL H, S, L tripets

    // Choose private variables for OpenMP threads
    // We want chroma and luminance LUTs to be firstprivate if present
    #if defined(USE_CHROMA_LUT) && defined(USE_LIGHT_LUT)
    #pragma omp parallel \
        default(none) shared(hsl, rgb) \
        firstprivate(size, chroma_table, light_table_big, \
                     CL_TABLE_SIZE, CH_TABLE_SIZE) \
        if (size >= MIN_IMG_SIZE_THREADED)
    #else  // else we don't have tables to make firstprivate
    #pragma omp parallel \
        default(none) shared(hsl, rgb) \
        firstprivate(size) \
        if (size >= MIN_IMG_SIZE_THREADED)
    #endif  // end OMP pragma

    { // begin OMP parallel
    rgb_to_luv_nd(rgb, hsl, size);
    #pragma omp barrier  // ensure LUV elements are done being written
    rgbluv_to_husl_nd(rgb, hsl, size);
    } // end OMP parallel

    return hsl;
}


// Aligned malloc for HUSL double arrays
static double* __attribute__((alloc_size(1))) allocate_hsl(size_t size) {
    double *hsl __attribute__((aligned(64))) = \
        (double*) malloc(size * sizeof(double));
    if (hsl == NULL) {
        fprintf(stderr, "Error: Couldn't allocate memory for HUSL array\n");
        exit(EXIT_FAILURE);
    }
    return hsl;
}


// Convert nonlinear RGB to CIE-LUV
static void rgb_to_luv_nd(uint8_t *restrict rgb, double *restrict luv, size_t size) {
    unsigned int i;
    #pragma omp for schedule(static)
    for (i = 0; i < size; i+=3) {
        double *luv_p = luv + i;
        uint8_t *rgb_p = rgb + i;
        const uint8_t r = *(rgb_p);
        const uint8_t g = *(++rgb_p);
        const uint8_t b = *(++rgb_p);

        // from RGB in [0, 255] to RGB-linear in [0,1]
        double rl, gl, bl;
        to_linear_rgb(r, g, b, &rl, &gl, &bl);

        // to CIE-XYZ to CIE-LUV
        double x, y, z;
        double *l, *u, *v;
        l = luv_p;
        u = (++luv_p);
        v = (++luv_p);
        to_xyz(rl, gl, bl, &x, &y, &z);
        to_luv(x, y, z, l, u, v);
    }
}


// Convert RGB to linear RGB.
static inline void to_linear_rgb(
        uint8_t r, uint8_t g, uint8_t b,
        double *restrict rl, double *restrict gl, double *restrict bl) {
    *rl = linear_table[r];
    *gl = linear_table[g];
    *bl = linear_table[b];
}


// Convert linear RGB to CIE-XYZ space.
// Note that this is somewhat different than the husl.py reference
// implementation by Boronine, the creator of HUSL. See Celebi's paper
// "Fast Color Space Transformations Using Minimax Approximations".
static inline void to_xyz(
        double r, double g, double b,
        double *restrict x, double *restrict y, double *restrict z) {
    *x = 0.412391*r + 0.357584*g + 0.180481*b;
    *y = 0.212639*r + 0.715169*g + 0.072192*b;
    *z = 0.019331*r + 0.119195*g + 0.950532*b;
}


// Convert CIE-XYZ to CIE-LUV
static inline void to_luv(
        double x, double y, double z,
        double *restrict l, double *restrict u, double *restrict v) {
    const double var_scale = x + 15*y + 3*z;
    const double var_u = 4*x / var_scale;
    const double var_v = 9*y / var_scale;
    *l = to_light(y);
    const double l13 = (*l)*13;
    *u = l13*(var_u - REF_U);
    *v = l13*(var_v - REF_V);
}


// Convert CIE-LUV to HUSL. The original RGB array is still passed in
// for the handling of boundary conditions (white and black pixels).
static void rgbluv_to_husl_nd(
        uint8_t *restrict rgb, double *restrict luv_hsl, size_t size) {
    unsigned int i;
    #pragma omp for schedule(guided)
    for (i = 0; i < size; i+=3) {
        double *hsl_p = luv_hsl + i;
        uint8_t *rgb_p = rgb + i;

        const uint8_t r = *rgb_p;
        const uint8_t g = *(++rgb_p);
        const uint8_t b = *(++rgb_p);

        if (r == 255 && g == 255 && b == 255) {
            *(hsl_p) = WHITE_HUE;
            *(++hsl_p) = WHITE_SATURATION;
            *(++hsl_p) = WHITE_LIGHTNESS;
        } else if (!r && !g && !b) {
            *(hsl_p) = 0;
            *(++hsl_p) = 0;
            *(++hsl_p) = 0;
        } else {
            // This is the most expensive part of the RGB->HUSL chain
            const double l = *hsl_p;
            const double u = *(hsl_p+1);
            const double v = *(hsl_p+2);
            const double h = to_hue(u, v);
            const double s = to_saturation(l, u, v, h);
            *(hsl_p) = h;
            *(++hsl_p) = s;
            *(++hsl_p) = l;
        }
    } // end OMP for
}


static const double DEG_PER_RAD = 180.0 / M_PI;


///////////////////////////////////////////////////
// Define a to_hue function based on compile flags
///////////////////////////////////////////////////


#if defined(USE_HUE_ATAN2_APPROX)  // if compiled with -DUSE_HUE_ATAN2_APPROX


#define PI 3.141592653589793
#define PIBY2 1.5707963267948966


// Returns HUSL hue given U & V of CIE-LUV
// The hue is the phase angle, in degrees, between U and V
// Hue values are in the interval [0, 360]
static double to_hue(double u, double v) {
    const double z = v/u;
    double hue;
    if (fabs(z) < 1.0) {
        // If we're in |V/U] < 1, use the faster, more accurate approx
        hue = atan_approx(z)*DEG_PER_RAD;
        if (u < 0) {
            hue += 180.0;
        } else if (v < 0) {
            hue += 360.0;
        }
    } else {
        // Else, for |V/U| >= 1, use the approx that works for all |V/U|
        hue = atan2_approx(v, u) * DEG_PER_RAD;
        if (hue < 0.0) {
            hue += 360.0;
        }
    }
    return hue;
}


static const double M_PI_4 = M_PI/4;


// a fast approximation of atan for |V/U| < 1
static inline double atan_approx(double z) {
    return M_PI_4*z - z*(fabs(z) - 1)*(0.2447 + 0.0663*fabs(z));
}


// a fast atan2 approxmiation
// it's somewhat slower and somewhat less accurate than the
// per-quadrant atan approx, but it works for |y/x| > 1 so we
// use this function where our input is out of bounds for atan_approx
static double atan2_approx(double y, double x) {
    if (x == 0.0) {
        if (y > 0.0) {
            return PIBY2;
        } else if (y == 0.0) {
            return 0.0;
        } else {
            return -PIBY2;
        }
    }

    const double z = y/x;
    double atan;

    if (fabs(z) < 1.0) {
        atan = z/(1.0 + 0.28*z*z);
        if (x < 0.0) {
            if (y < 0.0) {
                return atan - PI;
            } else {
                return atan + PI;
            }
        }
    } else {
        atan = PIBY2 - z/(z*z + 0.28);
        if (y < 0.0) {
            return atan - PI;
        }
    }
    return atan;
}


#else  // else use math.h's atan2


// CIE-UV to HUSL hue with a more costly atan2 call
static inline double to_hue(double u, double v) {
    double hue = atan2f(v, u) * DEG_PER_RAD;
    if (hue < 0) {
        hue += 360;
    }
    return hue;
}


#endif  // End to_hue definition


// Returns a saturation value from UV (of CIELUV), lightness, and hue.
// Saturation magnitude (hypotenuse b/t U & V) is found via sqrt(U**2 + V**2),
// then it's normalized by the max chroma, which is dictated by H and L.
static inline double to_saturation(double l, double u, double v, double h) {
    const double saturation = (100*CHROMA_SCALE) * sqrt(u*u + v*v) / max_chroma(l, h);
    return fmin(saturation, 100.0f);
}


///////////////////////////////////////////////////////
// Define a max_chroma function based on compile flags
///////////////////////////////////////////////////////


#if defined(USE_CHROMA_LUT)  // if compiled to use chroma lookup table


#if defined(INTERPOLATE_CHROMA)  // if compiled to use bilinear interpolation


#define CH_MAX_IDX CH_TABLE_SIZE-2
#define CL_MAX_IDX CL_TABLE_SIZE-2

// Returns a maximum chroma given an L, H pair.
// This max chroma is used to scale HUSL's saturation value.
// Uses _chroma_lookup.c with bilinear interpolation.
// This LUT approach is important, because finding the max chroma is
// the most expensive operation in RGB -> HUSL conversion.
// Reference (see Unit Square section):
// https://en.wikipedia.org/wiki/Bilinear_interpolation
static inline double max_chroma(double lightness, double hue) {
    // Compute H-value indices (axis 0) and L-value indices (axis 1)
    const double h_idx = hue / H_IDX_STEP;
    const double l_idx = lightness / L_IDX_STEP;
    const unsigned short h_idx_floor = fmax(0.0, fmin(CH_MAX_IDX, floorf(h_idx)));
    const unsigned short l_idx_floor = fmax(0.0, fmin(CL_MAX_IDX, floorf(l_idx)));

    // Find four known f() values in the unit square bilinear interp. approach
    const c_table_t chroma_00 = chroma_table[h_idx_floor][l_idx_floor];
    const c_table_t chroma_10 = chroma_table[h_idx_floor+1][l_idx_floor];
    const c_table_t chroma_01 = chroma_table[h_idx_floor][l_idx_floor+1];
    const c_table_t chroma_11 = chroma_table[h_idx_floor+1][l_idx_floor+1];

    // Find *normalized* x, y, (1-x), and (1-y) values
    // It's a coordinate system where the four known chromas are at
    // (0,0), (1,0), (0,1), and (1,1), so we normalize hue and luminance.
    const double h_norm = h_idx - h_idx_floor;  // our "x" value
    const double l_norm = l_idx - l_idx_floor;  // our "y" value
    const double h_inv = 1 - h_norm;  // (1-x)
    const double l_inv = 1 - l_norm;  // (1-y)
 
    // Compute f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
    const double chroma = chroma_00*h_inv*l_inv +
                          chroma_10*h_norm*l_inv +
                          chroma_01*h_inv*l_norm +
                          chroma_11*h_norm*l_norm;
    return fmax(1e-10f, chroma);
}


#else  // else just round, don't interpolate


#define CH_MAX_IDX CH_TABLE_SIZE-1
#define CL_MAX_IDX CL_TABLE_SIZE-1


// Returns chroma directly from the chroma LUT, given a HUSL [L, H] pair
// This makes RGB -> HUSL 5-10% faster
static inline double max_chroma(double lightness, double hue) {
    // Compute H-value indices (axis 0) and L-value indices (axis 1)
    const double h_scaled = hue / H_IDX_STEP;
    const double l_scaled = lightness / L_IDX_STEP;
    const unsigned short h_idx = fmax(0.0, fmin(CH_MAX_IDX, roundf(h_scaled)));
    const unsigned short l_idx = fmax(0.0, fmin(CL_MAX_IDX, roundf(l_scaled)));
    return (double) fmax(1e-10, chroma_table[h_idx][l_idx]);
}


#endif  // end chroma LUT definition


static inline double linear_interp_chroma(
        double chroma_0, double chroma_1, double offset) {
    return chroma_0 + offset*(chroma_1 - chroma_0);
}


#else  // else define accurate-but-slow max_chroma


// Returns max chroma given an L, H pair
// This max chroma is used to scale the HUSL saturation value
// so that it fits in [0, 100].
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
// The HUSL color space is basically CIE-LUV, but
// it overcomes its floating chroma value by "stretching"
// the color space so that a new channel, "saturation"
// is a percentage in [0, 100] for all possible values of hue and lightness.
// The images at husl-colors.org explain this more clearly!
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


#endif // end conditional max_chroma definition


////////////////////////////////////////////////////////////////////////
// Define a function that returns luminance from Y of the CIE-XYZ space
////////////////////////////////////////////////////////////////////////


#if defined(USE_LIGHT_LUT)  // if compiled with -DUSE_LIGHT_LUT


// Return a light value from a CIE-XYZ Y value.
// A light value lookup that accounts for a nonlinear
// relationship between Y, the input, and L, the output.
// The lookup table is combined from three smaller
// tables, each with a different Y-value-to-L-value scale.
static double to_light(double y_value) {
    double idx;
    if (y_value < Y_THRESH_0) {
        idx = y_value/Y_IDX_STEP_0;
    } else if (y_value < Y_THRESH_1) {
        idx = ((y_value - Y_THRESH_0)/Y_IDX_STEP_1) + L_SEGMENT_SIZE;
    } else {
        idx = ((y_value - Y_THRESH_1)/Y_IDX_STEP_2) + L_SEGMENT_SIZE*2;
    }
    const unsigned short idx_round = fmax(0, fmin(L_FULL_TABLE_SIZE-1, roundf(idx)));
    return (double) light_table_big[idx_round] / LIGHT_SCALE;
}


#else  // else define a to_light function with an expensive cube root


// Return a light value from a CIE-XYZ Y value.
static inline double to_light(double y_value) {
    if (y_value > EPSILON) {
        return 116 * cbrt(y_value / REF_Y) - 16;
    } else {
        return (y_value / REF_Y) * KAPPA;
    }
}


#endif // end to_light conditional definition


///////////////////////////////////////////
// Conversion in the HUSL -> RGB direction
///////////////////////////////////////////





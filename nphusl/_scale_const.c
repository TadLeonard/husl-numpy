#include <stdio.h>

/*
Constants as defined in the reference implementation, husl.py.
We could gather these from constants.py as this isn't very DRY, but
that seems like a lot of work.
*/


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


/*
As found via print functions below, the original husl.py reference
implementation used these constants in finding the max chroma value
given a lightness and hue pair.

The constant that dictates the "top2" variable in the min chroma function
is the same no matter which part of the M array we're using, so
as an optimization we just define a const double for SCALE_SUB2.
*/
const double SCALE_SUB1[3] = {969398.790856, -279707.331753, -84414.418054};
const double SCALE_BOTTOM[3] = {-120846.461733, -210946.241904, 694074.104001};
const double SCALE_SUB2 = 769860.000000;


void print_SCALE_SUB1(void) {
    int i;
    printf("double SCALE_SUB1[3] = {");
    for (i = 0; i < 3; i++) {
        printf("%f, ", 284517.0 * M[i][0] - 94839.0 * M[i][2]);
    }
    printf("}\n");
}


void print_SCALE_SUB2(void) {
    int i;
    printf("double SCALE_SUB2[3] = {");
    for (i = 0; i < 3; i++) {
        printf("%f, ",
            838422.0 * M[i][2] + 769860.0 * M[i][1] + 731718.0 * M[i][0]);
    }
    printf("}\n");
}


void print_SCALE_BOTTOM(void) {
    int i;
    printf("double SCALE_BOTTOM[3] = {");
    for (i = 0; i < 3; i++) {
        printf("%f, ",  632260.0 * M[i][2] - 126452.0 * M[i][1]);
    }
    printf("}\n");
}


int main(int argc, char* argv[]) {
    print_SCALE_SUB1();
    print_SCALE_SUB2();
    print_SCALE_BOTTOM();
} 


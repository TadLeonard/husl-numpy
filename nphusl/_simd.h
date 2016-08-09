
#include <stdint.h>

typedef double hsl_type;
extern hsl_type *rgb_to_husl_nd(uint8_t* rgb, size_t size);
extern hsl_type *rgb_to_husl_triplet(uint8_t r, uint8_t g, uint8_t b);


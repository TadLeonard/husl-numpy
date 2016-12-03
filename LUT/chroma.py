
"""
Generate a 2D lookup table for hue & lightness vs. chroma
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, ".")  # run from project root
import nphusl
import alignment
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-u",  "--hue-axis-size", default=256, type=int)
parser.add_argument("-l", "--light-axis-size", default=256, type=int)
parser.add_argument("-o", "--output-file-prefix", default=None)
parser.add_argument("-t", "--table-type", default="float",
                    choices=["double", "float", "unsigned short",
                             "uint_fast16_t", "int", "uint16_t",
                             "uint64_t"])
parser.add_argument("-i", "--int-scale", default=100, type=int)

args = parser.parse_args()
assert args.hue_axis_size < (1 << 16), "size must fit in 16 bits"
assert args.light_axis_size < (1 << 16), "size must fit in 16 bits"
NL = args.light_axis_size
NH = args.hue_axis_size
table_type = args.table_type
is_int_type = "int" in table_type or "short" in table_type
int_scale = args.int_scale if is_int_type else 1
out = args.output_file_prefix

# set up file objects for .c and .h files
out_h = open("{}.h".format(out), "w") if out else sys.stdout
out_c = open("{}.c".format(out), "w") if out else sys.stdout
out_header_name = os.path.split(out_h.name)[-1] if out else "<stdout>"

# write "generated with" message at tops of files
print("""// {}: generated with `python {}`

#include <stdint.h>

#define CHROMA_SCALE {}
""".format(out_h.name, " ".join(sys.argv), int_scale), file=out_h)
print("""// {}: generated with `python {}`

#include <{}>
""".format(out_c.name, " ".join(sys.argv), out_header_name), file=out_c)

# declare table types, sizes
print("const unsigned short CH_TABLE_SIZE;", file=out_h)
print("const unsigned short CL_TABLE_SIZE;", file=out_h)
print("const unsigned short CH_TABLE_SIZE = {};".format(NH), file=out_c)
print("const unsigned short CL_TABLE_SIZE = {};".format(NL), file=out_c)
print("typedef {} c_table_t;".format(table_type), file=out_h)

# choose value formatter
if is_int_type:
    def print_table_value(chroma):
        chroma = int(round(chroma * int_scale))
        print("{:5d}".format(chroma), end=", ", file=out_c)
    value_description = "(Chroma*{})".format(int_scale)
else:
    def print_table_value(chroma):
        print("{:7.3f}".format(chroma), end=", ", file=out_c)
    value_description = "Chroma"

# build chroma table
h_idx_step = 360.0 / (NH-1)
l_idx_step = 100.0 / (NL-1)
hues = np.arange(0, 360+h_idx_step, step=h_idx_step)
lights = np.arange(0, 100+l_idx_step, step=l_idx_step)
chroma_table = np.zeros(shape=(NH, NL), dtype=np.float)
print("Building chroma lookup table. . .")
for i, hue in enumerate(tqdm.tqdm(hues)):
    lch = np.zeros((lights.size, 3))
    lch[..., 2] = hue
    lch[..., 0] = lights
    c = nphusl.nphusl._max_lh_chroma(lch[1:])  # skip L=0, avoid NaN
    chroma_table[i, 1:] = c

# declare table, consts
print("const c_table_t chroma_table[{}][{}];".format(NH, NL), file=out_h)
print("const double H_IDX_STEP;", file=out_h)
print("const double L_IDX_STEP;", file=out_h)
print("const double H_IDX_STEP = {};".format(h_idx_step), file=out_c)
print("const double L_IDX_STEP = {};".format(l_idx_step), file=out_c)

# some statistics on the lookup tables to gauge their accuracy
h_diff = np.abs(chroma_table[1:, :] - chroma_table[:-1, :])
h_max = np.max(h_diff)
h_avg = np.sum(h_diff) / ((NH-1)*NL)
l_diff = np.abs(chroma_table[:, 1:] - chroma_table[:, :-1])
l_max = np.max(l_diff)
l_avg = np.sum(l_diff) / (NH*(NL-1))
print("", file=out_c)
print("// Ave delta across hue (1st axis): {}".format(h_avg),  file=out_c)
print("// Max delta across hue: {}".format(h_max), file=out_c)
print("// Ave delta across luminance (2nd axis): {}".format(l_avg), file=out_c)
print("// Max delta across luminance: {}".format(l_max), file=out_c)

# write out table initializer
if out:
    print("Writing chroma lookup table to disk. . .")
    iterhues = tqdm.tqdm(hues)
else:
    iterhues = hues
alignment = "__attribute__((aligned({})))".format(alignment.N)
print("const c_table_t {} chroma_table[{}][{}] = {{".format(
      alignment, NH, NL), file=out_c)

for i, hue in enumerate(iterhues):
    max_delta = np.max(np.abs(chroma_table[i, 1:] - chroma_table[i, :-1]))
    print("  // Index {}: {} for hue={} and luminance in [0, 100]"
          .format(i, value_description, hue), file=out_c)
    print("  // Max chroma delta: {}".format(max_delta), file=out_c)
    print("  {", file=out_c)
    print("    ", end="", file=out_c)
    l_start = 0
    for j, light in enumerate(lights):
        print_table_value(chroma_table[i][j])
        if not (j+1) % 16 and j:
            print(" // L={:0.3f} to L={:0.3f}".format(l_start, light), end="",
                  file=out_c)
            print("\n    ", end="", file=out_c)
            l_start = light + l_idx_step
    print("\n  },", file=out_c)
print("};", file=out_c)


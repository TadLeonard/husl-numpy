
"""
Generate a 2D lookup table for hue & lightness vs. chroma
"""
import os
import sys
import argparse

import numpy as np
import nphusl
import husl

import alignment


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--table-size", default=256, type=int)
parser.add_argument("-o", "--output-file-prefix", default=None)
parser.add_argument("-t", "--table-type", default="float",
                    choices=["double", "float"])

args = parser.parse_args()
N = args.table_size
table_type = args.table_type
out = args.output_file_prefix

# set up file objects for .c and .h files
out_h = open("{}.h".format(out), "w") if out else sys.stdout
out_c = open("{}.c".format(out), "w") if out else sys.stdout
out_header_name = os.path.split(out_h.name)[-1] if out else "<stdout>"

# write "generated with" message at tops of files
print("""// {}: generated with `python {}`

""".format(out_h.name, " ".join(sys.argv), N), file=out_h)
print("""// {}: generated with `python {}`

#include <{}>

""".format(out_c.name, " ".join(sys.argv), out_header_name), file=out_c)

# declare table types, sizes
print("extern const unsigned short C_TABLE_SIZE;", file=out_h)
print("const unsigned short C_TABLE_SIZE = {};".format(N), file=out_c)
print("typedef {} c_table_t;".format(table_type), file=out_h)

# build chroma table
h_idx_step = 360.0 / N
l_idx_step = 100.0 / N
hues = np.arange(0, 360, step=h_idx_step)
lights = np.arange(0, 100, step=l_idx_step)
chroma_table = np.zeros(shape=(N, N), dtype=table_type)
for i, hue in enumerate(hues):
    for j, light in enumerate(lights):
        chroma_table[i][j] = husl.max_chroma_for_LH(light, hue)
chroma_table = np.nan_to_num(chroma_table)  # NaN @ hue=0

# declare table, consts
print("const c_table_t chroma_table[{}][{}];".format(N, N), file=out_h)
print("extern const c_table_t h_idx_step;", file=out_h)
print("extern const c_table_t l_idx_step;", file=out_h)
print("const c_table_t h_idx_step = {};".format(h_idx_step), file=out_c)
print("const c_table_t l_idx_step = {};".format(l_idx_step), file=out_c)

# some statistics on the lookup tables to gauge their accuracy
h_diff = np.sum(np.abs(chroma_table[1:, :] - chroma_table[:-1, :]), axis=1) / (N-1)
h_max = np.max(h_diff)
h_diff = np.sum(h_diff) / len(h_diff)
l_diff = np.sum(np.abs(chroma_table[:, 1:] - chroma_table[:, :-1]), axis=0) / (N-1)
l_max = np.max(l_diff)
l_diff = np.sum(l_diff) / len(l_diff)
print("", file=out_c)
print("// Ave delta across hue (1st axis): {}".format(h_diff),  file=out_c)
print("// Max delta across hue: {}".format(h_max), file=out_c)
print("// Ave delta across luminance (2nd axis): {}".format(l_diff), file=out_c)
print("// Max delta across luminance: {}".format(l_max), file=out_c)

# write out table initializer
alignment = "__attribute__((aligned({})))".format(alignment.N)
print("const c_table_t {} chroma_table[{}][{}] = {{".format(
      alignment, N, N), file=out_c)
for i, hue in enumerate(hues):
    print("  // Index {}: Chromas for hue={} and luminance in [0, 100)"
          .format(i, hue), file=out_c)
    print("  {", file=out_c)
    print("    ", end="", file=out_c)
    l_start = 0
    for j, light in enumerate(lights):
        print("{:7.3f}".format(chroma_table[i][j]), end=", ", file=out_c)
        if not (j+1) % 8 and j:
            print(" // L={:0.3f} to L={:0.3f}".format(l_start, light), end="",
                  file=out_c)
            print("\n    ", end="", file=out_c)
            l_start = light + l_idx_step
    print("\n  },", file=out_c)
print("};", file=out_c)


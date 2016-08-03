
"""
Generate a 2D lookup table for hue & lightness vs. chroma
"""
import os
import sys
import argparse

import numpy as np
import nphusl
import husl


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
print("extern const c_table_t chroma_table[{}][{}];".format(N, N), file=out_h)
print("extern const c_table_t h_idx_step_{};".format(i), file=out_h)
print("extern const c_table_t l_idx_step_{};".format(i), file=out_h)
print("const c_table_t h_idx_step = {};".format(h_idx_step), file=out_c)
print("const c_table_t l_idx_step = {};".format(l_idx_step), file=out_c)

# write out table initializer
print("const c_table_t chroma_table[{}][{}] = {{".format(N, N), file=out_c)
for i in range(N):
    print("  {", file=out_c)
    print("    ", end="", file=out_c)
    for j in range(N):
        print("{:7.3f}".format(chroma_table[i][j]), end=", ", file=out_c)
        if not (j+1) % 8 and j:
            print("\n    ", end="", file=out_c)
    print("\n  },", file=out_c)
print("};", file=out_c)

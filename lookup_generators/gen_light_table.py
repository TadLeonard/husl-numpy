"""
Generate a lookup table for CIE XYZ Y-value mapped to
a HUSL luminance value.
"""
import os
import sys
import argparse

import numpy as np
import nphusl

import alignment

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--table-size", default=256, type=int)
parser.add_argument("-o", "--output-file-prefix", default=None)
parser.add_argument("-t", "--table-type", default="float",
                    choices=["double", "float"])
parser.add_argument("-y", "--y-steps", default=[0.05, 0.3],
                    nargs="+", type=float)

args = parser.parse_args()
N = args.table_size
table_type = args.table_type
out = args.output_file_prefix
y_steps = args.y_steps + [1.0]
N_BIG = N * len(y_steps)

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
print("const unsigned short L_SEGMENT_SIZE;", file=out_h)
print("const unsigned short L_FULL_TABLE_SIZE;", file=out_h)
print("const unsigned short L_SEGMENT_SIZE = {};".format(N), file=out_c)
print("const unsigned short L_FULL_TABLE_SIZE = {};".format(N*3), file=out_c)
print("typedef {} l_table_t;".format(table_type), file=out_h)

# declare tables, Y value steps
for i, step in enumerate(y_steps[:-1]):
    print("const l_table_t y_thresh_{};".format(i), file=out_h)
start = 0.0
for i, step in enumerate(y_steps):
    y_idx_step = (step - start) / N
    print("const l_table_t light_table_{}[{}];".format(i, N), file=out_h)
    print("const l_table_t y_idx_step_{};".format(i), file=out_h)
    print("const l_table_t y_idx_step_{} = {};".format(i, y_idx_step), file=out_c)
    print("const l_table_t y_thresh_{} = {:0.04f};".format(i, step), file=out_c)
    start = step

# declare big segmented light LUT
print("const l_table_t light_table_big[{}];".format(
      N_BIG), file=out_h)
print("", file=out_c)

# initializing segmented light LUTs
# initialize little tables, collect values into big table
big_light_lookup = np.zeros((N_BIG,), dtype=float)
start = 0.0
alignment = "__attribute__((aligned({})))".format(alignment.N)
for i, stop in enumerate(y_steps):
    # Generate our LUT from a range of Y-values in [0, 1.0)
    # NOTE: this uniform range works because L increases monitonically with y
    step = (stop - start) / N
    uniform_y = np.arange(start, stop, step=step)
    light_lookup = nphusl.nphusl._to_light(uniform_y)
    big_light_lookup[i*N: i*N+N] = light_lookup

    # collect statistics on LUT to gauge its usefulness
    light_diff = light_lookup[1:] - light_lookup[:-1]
    avg_light_diff = np.sum(light_diff) / (N-1)
    max_light_diff = np.max(light_diff)
    print("// Avg light value step size: {:0.4f}".format(
        avg_light_diff), file=out_c)
    print("// Max light value step size: {:0.4f}".format(
        max_light_diff), file=out_c)
    print("// Max light value error: {:0.4f}".format(
        max_light_diff/2), file=out_c)

    # write out LUT initializer
    print("// Light values for {} <= Y < {}".format(start, stop), file=out_c)
    print("const l_table_t {} light_table_{}[{}] = {{".format(
          alignment, i, N), file=out_c)
    for i, l in enumerate(light_lookup):
        print("{:7.4f}".format(l), end=", ", file=out_c)
        if not (i+1) % 8 and i:
            print("", file=out_c)
    print("};\n", file=out_c)
    start = stop

# initialize big segmented table (all values concatenated into one)
print("const l_table_t {} light_table_big[{}] = {{".format(
      alignment, N_BIG), file=out_c)
for i, l in enumerate(big_light_lookup):
    print("{:6.3f}".format(l), end=", ", file=out_c)
    if not (i+1) % 8 and i:
        print("", file=out_c)
print("};\n", file=out_c)


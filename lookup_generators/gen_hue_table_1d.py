"""Generate a 1D lookup table for LUV to HUSL hue.
The input is the quotient of V and U from CIE-LUV.
This replaces an expensive arctan2 call."""

import argparse
import math
import os
import sys

from itertools import product

import numpy as np

#u: [-83.080338, 174.980330], v: [-134.266145, 107.444822]
#u: [-83.080338, 174.980330], v: [-134.266145, 107.444822]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--table-size", type=int, default=1024)
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
print("extern const unsigned short H_TABLE_SIZE;", file=out_h)
print("const unsigned short H_TABLE_SIZE = {};".format(N), file=out_c)
print("typedef {} h_table_t;".format(table_type), file=out_h)

# U, V bounds for the LUV color space
# Arctangent(V, U) is discontinuous across V=0 and U=0
# We must break up the function into four tables to accomodate for this
# NOTE: Boundary values were found by converting rgb16million.tiff to LUV
U_MIN = -83.080338
U_MAX = 174.980330
V_MIN = -134.266145
V_MAX = 107.444822
# declare table, consts
table_signs = [(0, 0), (0, 1), (1, 0), (1, 1)]
table_indices = [(vsign << 1) | usign for vsign, usign in table_signs]
table_postfixes = ["{:d}{:d}".format(usign, vsign)
                   for usign, vsign in table_signs]

"""
Find V/U index steps.
In other words, find out how much V/U increases for each
increased index in the hue table for each of the four quadrants.

Quadrant 1: 00-90 degrees (V+, U+)
  * maxes out when V is high and U is low
  * V/U quotient from 0 (0 degrees) to INF (90 degrees)
  * realistically, V/U is in [0, N]
Quadrant 2: 90-180 degrees (V+, U-)
  * maxes out when V is low and |U| is high
  * V/U quotient from -INF (90 degrees) to -0 (180 degrees)
  * realistically, V/U is in [-N, -0]
Quadrant 3: 270-360 degrees (V-, U+)
  * maxes out when |V| is low and U is high
  * V/U quotient from -INF (270 degrees) to -0 (360 degrees)
  * realistically, V/U is in [-0, -N]
Quadrant 4: 180-270 degrees (V-, U-)
  * maxes out when |V| is high and |U| is low
  * V/U quotient from 0 (180 degrees) to INF (270 degrees)
  * realistically, V/U is in [0, N]
"""
max_vu_quotient = 100
vu_00_idx_step = max_vu_quotient/N
vu_01_idx_step = max_vu_quotient/N
vu_10_idx_step = max_vu_quotient/N
vu_11_idx_step = max_vu_quotient/N
table_index_steps = [vu_00_idx_step,
                     vu_01_idx_step,
                     vu_10_idx_step,
                     vu_11_idx_step,]

for index in table_indices:
    usign, vsign = table_signs[index]
    vustep = table_index_steps[index]
    postfix = table_postfixes[index]
    print("extern const h_table_t hue_table_{}[{}];".format(postfix, N), file=out_h)
    print("extern const h_table_t vu_idx_step_{};".format(postfix), file=out_h)
    print("const h_table_t vu_idx_step_{} = {};".format(
          postfix, vustep), file=out_c)

# build four hue tables for U+V+/U+V-/U-V+/U-V- or 00/01/10/11 cases
# another way of looking at it is we're making a table for each of the
# four quadrants for arctan(V/H)
hue_tables = [np.zeros((N, 2), dtype=np.float)
              for _ in table_indices]

# V/U -> Hue functions for the four arctan quadrants
hue_converters = [
    lambda vu: math.degrees(math.atan(vu)),
    lambda vu: math.degrees(math.atan(vu)) + 180,
    lambda vu: math.degrees(math.atan(vu)) + 360,
    lambda vu: math.degrees(math.atan(vu)) + 180,]

# fill out our four tables
for index in table_indices:
    postfix = table_postfixes[index]
    hue_table = hue_tables[index]
    to_hue = hue_converters[index]
    vu_step = table_index_steps[index]
    for i in range(N):
        vu_value = i*vu_step
        hue_value = to_hue(vu_value)
        hue_table[i] = vu_value, hue_value

# some statistics on the lookup tables to gauge their accuracy
for index in table_indices:
    vneg, uneg = table_signs[index]
    hue_table = hue_tables[index]
    table_postfix = table_postfixes[index]
    vu_step = table_index_steps[index]
    hue_diffs = np.abs(hue_table[1:, 1] - hue_table[:-1, -1])
    hue_diff_ave = np.sum(hue_diffs) / (N - 1)
    hue_diff_max = np.max(hue_diffs)
    v_fmt = "V<0" if vneg else "V>0"
    u_fmt = "U<0" if uneg else "U>0"
    print("", file=out_c)
    print("// V/U -> H where {} & {}".format(v_fmt, u_fmt), file=out_c)
    print("// Ave delta: {}".format(hue_diff_ave), file=out_c)
    print("// Max delta: {}".format(hue_diff_max), file=out_c)

    # write out table initializer for UV-H table with V<0
    print("const h_table_t hue_table_{}[{}] = {{".format(
          table_postfix, N, N), file=out_c)
    print("    ", end="", file=out_c)
    vu_start = hue_table[0, 0]
    for i, (vu, hue) in enumerate(hue_table):
        print("{:7.3f}".format(hue_table[i, 1]), end=", ", file=out_c)
        if not (i+1) % 8 and i and i != N-1:
            print(" // V/U in [{:0.3f}, {:0.3f}]".format(
                  vu_start, vu), end="", file=out_c)
            print("\n    ", end="", file=out_c)
            vu_start = vu + vu_step
    print(" // V/U in [{:0.3f}, {:0.3f}]".format(
          vu_start, vu), end="", file=out_c)
    print("\n};\n", file=out_c)


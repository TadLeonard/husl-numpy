"""Generate a 2D lookup table for LUV to HUSL hue.
The inputs are U and V from CIE-LUV.
This replaces an expensive arctan2 call."""

import argparse
import math
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)


#u: [-83.080338, 174.980330], v: [-134.266145, 107.444822]
#u: [-83.080338, 174.980330], v: [-134.266145, 107.444822]


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--table-size", type=int, default=256)
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
# Arctangent(V, U) is discontinuous across V=0
# We must break up the function into two tables to accomodate for this
U_MIN = -83.080338
U_MAX = 174.980330
V_MIN = -134.266145
V_MAX = 107.444822
u_idx_step = (U_MAX - U_MIN) / N
v_neg_idx_step = (0 - V_MIN) / N
v_pos_idx_step = (V_MAX - 0) / N

# declare table, consts
print("extern const h_table_t hue_table_v_neg[{}][{}];".format(N, N), file=out_h)
print("extern const h_table_t hue_table_v_pos[{}][{}];".format(N, N), file=out_h)
print("extern const h_table_t u_idx_step;", file=out_h)
print("extern const h_table_t v_neg_idx_step;", file=out_h)
print("extern const h_table_t v_pos_idx_step;", file=out_h)
print("const h_table_t u_idx_step = {};".format(u_idx_step), file=out_c)
print("const h_table_t v_neg_idx_step = {};".format(v_neg_idx_step), file=out_c)
print("const h_table_t v_pos_idx_step = {};".format(v_pos_idx_step), file=out_c)

# build four hue tables for U+V+/U+V-/U-V+/U-V- or 00/01/10/11 cases
# another way of looking at it is we're making a table for each of the
# four quadrants for arctan(V/H)
uvneg_h = np.zeros((N, N, 3), dtype=np.float)
uvpos_h = np.zeros((N, N, 3), dtype=np.float)
U_neg, V_neg, H_neg = (uvneg_h[..., i] for i in range(3))
U_pos, V_pos, H_pos = (uvpos_h[..., i] for i in range(3))
u_values = np.arange(U_MIN, U_MAX, u_idx_step)
v_neg_values = np.arange(V_MIN, 0, v_neg_idx_step)
v_pos_values = np.arange(0, V_MAX, v_pos_idx_step)

# fill our negative-V LV-to-H table
# and also our positive-V LV-to-H table
for i in range(N):
    u = u_values[i]
    for j in range(N):
        v_neg = v_neg_values[j]
        v_pos = v_pos_values[j]
        h_neg = math.degrees(math.atan2(v_neg, u))
        h_neg += (360 if h_neg < 0 else 0)
        h_pos = math.degrees(math.atan2(v_pos, u))
        h_pos += (360 if h_pos < 0 else 0)
        uvneg_h[i, j] = u, v_neg, h_neg
        uvpos_h[i, j] = u, v_pos, h_pos

# some statistics on the lookup tables to gauge their accuracy
u_neg_diff = np.sum(np.abs(H_neg[1:, :] - H_neg[:-1, :]), axis=1) / (N-1)
u_neg_max = np.max(u_neg_diff)
u_neg_diff = np.sum(u_neg_diff) / len(u_neg_diff)
u_pos_diff = np.sum(np.abs(H_neg[1:, :] - H_neg[:-1, :]), axis=1) / (N-1)
u_pos_max = np.max(u_pos_diff)
u_pos_diff = np.sum(u_pos_diff) / len(u_pos_diff)
v_neg_diff = np.sum(np.abs(H_pos[:, 1:] - H_pos[:, :-1]), axis=0) / (N-1)
v_neg_max = np.max(v_neg_diff)
v_neg_diff = np.sum(v_neg_diff) / len(v_neg_diff)
v_pos_diff = np.sum(np.abs(H_pos[:, 1:] - H_pos[:, :-1]), axis=0) / (N-1)
v_pos_max = np.max(v_pos_diff)
v_pos_diff = np.sum(v_pos_diff) / len(v_pos_diff)
print("", file=out_c)
print("// Negative-V UV-to-H table", file=out_c)
print("// Ave delta across U (1st axis): {}".format(u_neg_diff),  file=out_c)
print("// Max delta across U: {}".format(u_neg_max), file=out_c)
print("// Ave delta across V (2nd axis): {}".format(v_neg_diff), file=out_c)
print("// Max delta across V: {}".format(v_neg_max), file=out_c)
print("", file=out_c)
print("// Positive-V UV-to-H table", file=out_c)
print("// Ave delta across U (1st axis): {}".format(u_pos_diff),  file=out_c)
print("// Max delta across U: {}".format(u_pos_max), file=out_c)
print("// Ave delta across V (2nd axis): {}".format(v_pos_diff), file=out_c)
print("// Max delta across V: {}".format(v_neg_max), file=out_c)

# write out table initializer for UV-H table with V<0
print("const h_table_t hue_table_neg[{}][{}] = {{".format(N, N), file=out_c)
for i, u in enumerate(u_values):
    print("  // Index {}: Hues for U={} and V in [{}, {})"
          .format(i, u, V_MIN, 0), file=out_c)
    print("  {", file=out_c)
    print("    ", end="", file=out_c)
    v_start = V_MIN
    for j, v in enumerate(v_neg_values):
        print("{:7.3f}".format(H_neg[i][j]), end=", ", file=out_c)
        if not (j+1) % 8 and j:
            print(" // V={:0.3f} to V={:0.3f} (V/U = {:0.3f})".format(v_start,
            v, v/u), end="",
                  file=out_c)
            print("\n    ", end="", file=out_c)
            v_start = v + v_pos_idx_step
    print("\n  },", file=out_c)
print("};\n", file=out_c)

# write out table initializer for UV-H table with V>0
print("const h_table_t hue_table_pos[{}][{}] = {{".format(N, N), file=out_c)
for i, u in enumerate(u_values):
    print("  // Index {}: Hues for U={} and V in [{}, {})"
          .format(i, u, 0, V_MAX), file=out_c)
    print("  {", file=out_c)
    print("    ", end="", file=out_c)
    v_start = 0
    for j, v in enumerate(v_pos_values):
        print("{:7.3f}".format(H_pos[i][j]), end=", ", file=out_c)
        if not (j+1) % 8 and j:
            print(" // V={:0.3f} to V={:0.3f} (V/U = {:0.3f})".format(v_start,
            v, v/u), end="",
                  file=out_c)
            print("\n    ", end="", file=out_c)
            v_start = v + v_pos_idx_step
    print("\n  },", file=out_c)
print("};", file=out_c)


"""
Generate a lookup table for Y-coordinate to HUSL light value
"""
import os
import sys
import argparse

import numpy as np
import nphusl


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

"""
rgb = np.zeros(shape=(N, N, 3), dtype=np.uint8)
rgb[:] = np.round(np.random.rand(N, N, 3) * 255.0)
rgb_f = rgb.astype(np.float) / 255.0
rgb[0, 0] = 0  # at least one black pixel
rgb[1, 1] = 1  # at least one white pixel
unique_rgb = np.sum(rgb_f, axis=2)
print("Unique RGB triplets: {}".format(unique_rgb.size))

xyz = nphusl.rgb_to_xyz(rgb_f)
y = xyz[..., 1]
unique_y = np.unique(y)
print("Unique Y values: {}".format(unique_y.size))
print("Min Y: {}, Max Y: {}".format(np.min(y), np.max(y)))

light = nphusl._nphusl._f(y)
unique_light = np.unique(light)
print("Unique light values: {}".format(unique_light.size))
avg_spacing = np.sum(unique_light[1:] - unique_light[:-1]) / \
              (unique_light.size - 1)
print("Average light value spacing: {}".format(avg_spacing))
"""

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
print("extern const unsigned short L_TABLE_SIZE;", file=out_h)
print("extern const unsigned short L_BIG_TABLE_SIZE;".format(N_BIG), file=out_h)
print("const unsigned short L_TABLE_SIZE = {};".format(N), file=out_c)
print("const unsigned short L_BIG_TABLE_SIZE = {};".format(N_BIG), file=out_c)
print("typedef {} l_table_t;".format(table_type), file=out_h)

# declare tables, Y value steps
for i, step in enumerate(y_steps[:-1]):
    print("extern const l_table_t y_thresh_{};".format(i), file=out_h)
start = 0.0
for i, step in enumerate(y_steps):
    y_idx_step = (step - start) / N
    print("extern const l_table_t light_table_{}[{}];".format(i, N), file=out_h)
    print("extern const l_table_t y_idx_step_{};".format(i), file=out_h)
    print("const l_table_t y_idx_step_{} = {};".format(i, y_idx_step), file=out_c)
    print("const l_table_t y_thresh_{} = {:0.04f};".format(i, step), file=out_c)
    start = step
print("extern const l_table_t big_light_table[{}];".format(N_BIG), file=out_h)
print("", file=out_c)
big_light_lookup = np.zeros((N_BIG,), dtype=float)

# write out little tables
start = 0.0
for i, stop in enumerate(y_steps):
    # Generate our LUT from a range of Y-values in [0, 1.0)
    # NOTE: this uniform range works because L increases monitonically with y
    step = (stop - start) / N
    uniform_y = np.arange(start, stop, step=step)
    light_lookup = nphusl._nphusl._f(uniform_y)
    big_light_lookup[i*N: i*N+N] = light_lookup
    avg_y_thresh = \
        np.sum(light_lookup[1:] - light_lookup[:-1]) / (N-1)
    print("// Avg light value step size: {:0.4f}".format(
        avg_y_thresh), file=out_c)
    print("// Light values for {} <= Y < {}".format(start, stop), file=out_c)
    print("const l_table_t light_table_{}[{}] = {{".format(
          i, N), file=out_c)
    for i, l in enumerate(light_lookup):
        print("{:6.3f}".format(l), end=", ", file=out_c)
        if not i % 8 and i:
            print("", file=out_c)
    print("};\n", file=out_c)
    start = stop

# write out big table (all values concatenated into one)
print("const l_table_t big_light_table[{}] = {{".format(
      N_BIG), file=out_c)
for i, l in enumerate(big_light_lookup):
    print("{:6.3f}".format(l), end=", ", file=out_c)
    if not i % 8 and i:
        print("", file=out_c)
print("};\n", file=out_c)

"""
Generate a lookup table for Y-coordinate to HUSL light value
"""
import sys
import argparse

import numpy as np
import nphusl


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--table-size", default=256, type=int)
parser.add_argument("-o", "--output-file-prefix", default=None)
args = parser.parse_args()
N = args.table_size
out = args.output_file_prefix


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


out_h = open("{}.h".format(out), "rw") if out else sys.stdout
out_c = open("{}.c".format(out), "rw") if out else sys.stdout

print("""// {} generated with `python {}`

extern const light_table[{}];

""".format(out_h.name, " ".join(sys.argv), N), file=out_h)

print("""// {} generated with `python {}`

#include <{}>

""".format(out_c.name, " ".join(sys.argv), N), file=out_c)

steps = (0.05, 0.3, 1.0)
for i, step in enumerate(steps[:-1]):
    print("extern const float step_{};".format(i, step), file=out_h)
    print("const float step_{} = {:0.04f};".format(i, step), file=out_c)
print("", file=out_c)
start = 0.0
for i, stop in enumerate(steps):
    # Generate our LUT from a range of Y-values in [0, 1.0)
    # NOTE: this uniform range works because L increases monitonically with y
    step = (stop - start) / N
    uniform_y = np.arange(start, stop, step=step)
    light_lookup_table = nphusl._nphusl._f(uniform_y)
    avg_light_step = \
        np.sum(light_lookup_table[1:] - light_lookup_table[:-1]) / (N-1)
    print("// Avg light value step size: {:0.4f}".format(
        avg_light_step), file=out_c)
    print("// Light values for {} <= Y < {}".format(start, stop), file=out_c)
    print("const float light_table_{}[{}] = {{".format(
        i, N), file=out_c)
    for i, l in enumerate(light_lookup_table):
        print("{:6.3f}".format(l), end=", ", file=out_c)
        if not i % 8 and i:
            print("", file=out_c)
    print("};\n", file=out_c)
    start = stop

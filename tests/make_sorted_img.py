import imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("img", default="rgb16million.tiff")
parser.add_argument("-e", "--ext", default=None)
args = parser.parse_args()
input_name, input_ext = args.img.split(".")
ext = args.ext or input_ext

rgb3d = imageio.imread(args.img)
rgb2d = rgb3d.reshape((rgb3d.shape[0]*rgb3d.shape[1], 3))
r, g, b = (rgb2d[..., n] for n in range(3))
sort2d = np.lexsort((b, g, r))

rgb_sorted = rgb2d[sort2d]
rgb3d = rgb_sorted.reshape(rgb3d.shape)
imageio.imwrite("{}_sorted.{}".format(input_name, ext), rgb3d, quality=100)

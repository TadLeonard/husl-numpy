import imageio
import numpy as np
import itertools

pixels = 256**3
size = pixels * 3
sidelen = int(pixels ** 0.5)
rgb_iterator = itertools.product(range(256), repeat=3)
color_iterator = (color for rgb in rgb_iterator for color in rgb)

rgb_1d = np.fromiter(color_iterator, dtype=np.uint8)
rgb_2d = np.zeros(shape=(pixels, 3), dtype=np.uint8)

rgb_2d[..., 0] = rgb_1d[::3]
rgb_2d[..., 1] = rgb_1d[1::3]
rgb_2d[..., 2] = rgb_1d[2::3]
rgb_3d = rgb_2d.reshape((sidelen, sidelen, 3))

"""
rgb_3d = imread.imread("rgb16million.jpg")
rgb_iterator = itertools.product(range(256), repeat=3)
for rgb in rgb_iterator:
    print(rgb, np.sum(np.all(rgb_3d == rgb, axis=2)))
"""

filename = "rgb16million.tiff"
imageio.imwrite(filename, rgb_3d)
print("Wrote {}".format(filename))

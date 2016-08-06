import imageio
import numpy as np

rgb3d = imageio.imread("rgb16million.tiff")
rgb2d = rgb3d.reshape((rgb3d.shape[0]**2, 3))
sum2d = np.sum(rgb2d, axis=1, dtype=np.int32)

sum_sorted = np.argsort(sum2d)
rgb_sorted = np.zeros_like(rgb2d)
rgb_sorted = rgb2d[sum_sorted]
rgb3d = rgb_sorted.reshape(rgb3d.shape)
imageio.imwrite("rgb16million_sorted.jpg", rgb3d, quality=100)
import sys; sys.exit()

for iarr, jarr in zip(rgb):
    print(sum_sorted[i, :10])
    rgb_sorted[i] = rgb_sum[sum_sorted[i]]
print(rgb_sorted)

print(sum_sorted)

import sys; sys.exit()
rgb_sorted = rgb[sum_sorted]
pixels = rgb_sorted.shape[0]*rgb_sorted.shape[1]
rgb_flat = rgb_sorted.reshape((pixels, 3))
eq = np.all(rgb_flat[:-1] == rgb_flat[1:], axis=1)
print("Total equal pixels:", np.sum(eq))
imageio.imwrite("rgb16million_sorted.jpg", rgb_sorted)

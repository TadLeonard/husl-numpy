import imageio
import numpy as np

img = (255 * np.random.rand(2000, 2000, 3)).astype(np.uint8)
imageio.imwrite("tests/rand_4million.jpg", img)


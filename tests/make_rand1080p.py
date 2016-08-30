import imageio
import numpy as np

img = (255 * np.random.rand(1080, 1920, 3)).astype(np.uint8)
imageio.imwrite("tests/rand_1080p.jpg", img)


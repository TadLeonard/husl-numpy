import sys
import imread
import numpy as np
import nphusl


def reveal_red(img):
    hue = nphusl.to_hue(img)
    below_max = hue < 100
    above_min = hue > 50
    select = np.logical_and(below_max, above_min)
    img[select] = (0, 200, 125)
    return img


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    reveal_red(img)
    imread.imwrite("red.jpg", img)

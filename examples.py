import sys
import imread
import numpy as np
import nphusl


def reveal_red(img):
    # convert an integer RGB image to an 2D array of HUSL hue values
    hue = nphusl.to_hue(img)
    # create a filter for pixels with hues between 50 and 100
    below_max = hue < 100
    above_min = hue > 50
    select = np.logical_and(below_max, above_min)
    img[select] = (0, 200, 125)  # reveal the selected region
    return img


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    reveal_red(img)
    imread.imwrite("red.jpg", img)

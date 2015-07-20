import sys
import imread
import numpy as np
import nphusl


def reveal_blue(img):
    """Easy mode! Selecting bluish pixels with the HUSL color space."""
    # convert an integer RGB image to an 2D array of HUSL hue values
    hue = nphusl.to_hue(img)
    # create a filter for pixels with hues between 50 and 100
    select = np.logical_and(hue > 250, hue < 290)
    out = img.copy()
    out[select] = (0, 0, 255)  # reveal the selected region
    return out


def reveal_blue_rgb(img):
    """Hard mode! Selecting blueish pixels without the convenience of the
    HUSL color space."""
    R, G, B = (img[..., n] for n in range(3))  # break out RGB color channels
    # we'll try to create a "bluish" selection by choosing pixels for which
    # the blue channel has a greater value than the others
    select = np.logical_and(B > R, B > G)  # no overpowering red or green
    select = np.logical_and(select, B > 125)  # strong enough blue channel
    out = img.copy()
    out[select] = (0, 0, 255)  # reveal the selected region
    return out


def hue_graybow(img):
    for hue in range(0, 400, 10)


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    good = reveal_blue(img)
    bad = reveal_blue_rgb(img) 
    imread.imwrite("blue.jpg", good)
    imread.imwrite("blue_bad.jpg", bad)


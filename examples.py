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
    out[select] = 0xFF  # reveal the selected region
    return out, "blue"


def reveal_blue_rgb(img):
    """Hard mode! Selecting blueish pixels without the convenience of the
    HUSL color space."""
    R, G, B = (img[..., n] for n in range(3))  # break out RGB color channels
    # we'll try to create a "bluish" selection by choosing pixels for which
    # the blue channel has a greater value than the others
    select = np.logical_and(B > R, B > G)  # no overpowering red or green
    select = np.logical_and(select, B > 125)  # strong enough blue channel
    out = img.copy()
    out[select] = 0xFF  # reveal the selected region
    return out, "blue_rgb"


def reveal_pink(img):
    hue = nphusl.to_hue(img)  # a 2D array of HUSL hue values
    pinkish = np.logical_or(hue < 10, hue > 320)  # orange pixel selection
    out = img.copy()
    out[pinkish] = 0xFF  # change selection to white
    return out, "pink"


def hue_graybow(img):
    out = img.copy()
    hue = nphusl.to_hue(img)
    for low, high in nphusl.chunk(360, 5):
        is_odd = low % 10
        select = np.logical_and(hue > low, hue < high)
        c = int(2 * high / 3)
        if is_odd:
            color = c, 0x00, c
        else:
            color = 0x00, c, c // 2
        out[select] = color
    return out, "graybow"


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    transforms = reveal_blue, reveal_blue_rgb, reveal_pink, hue_graybow
    for t in transforms:
        out, name = t(img)
        imread.imwrite(name + ".jpg", out)


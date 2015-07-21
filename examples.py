import sys
import imread
import numpy as np
import nphusl


def reveal_blue(img):
    """Easy mode! Selecting bluish pixels with the HUSL color space."""
    # convert an integer RGB image to an 2D array of HUSL hue values
    hue = nphusl.to_hue(img)
    # create a filter for pixels with hues between 50 and 100
    bluish = np.logical_and(hue > 250, hue < 290)
    out = img.copy()
    out[~bluish] *= 0.5  # reveal the selected region
    return out, "blue"


def reveal_blue_rgb(img):
    """Hard mode! Selecting blueish pixels without the convenience of the
    HUSL color space."""
    R, G, B = (img[..., n] for n in range(3))  # break out RGB color channels
    # we'll try to create a "bluish" selection by choosing pixels for which
    # the blue channel has a greater value than the others
    bluish = np.logical_and(B > R, B > G)  # no overpowering red or green
    bluish = np.logical_and(bluish, B > 125)  # strong enough blue channel
    out = img.copy()
    out[~bluish] *= 0.5  # reveal the selected region
    return out, "blue_rgb"


def reveal_light(img):
    hsl = nphusl.to_husl(img)  # a 2D array of HUSL hue values
    lightness = hsl[..., 2]  # just the lightness channel
    dark = lightness < 62
    out = img.copy()
    out[dark] = 0x00  # change selection to black
    return out, "light"


def reveal_light_rgb(img):
    R, G, B = (img[..., n] for n in range(3))
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    dark = luminance < 140
    out = img.copy()
    out[dark] = 0x00  # change selection to black
    return out, "light_rgb"


def hue_watermelon(img):
    out = img.copy()
    hsl = nphusl.to_husl(img)
    hue, _, lightness = (hsl[..., n] for n in range(3))
    pink =  1.3, 0.3, 0.6 
    green = 0.3, 1.1, 0.3
    for low, high in nphusl.chunk(360, 5):  # chunks of the hue range
        select = np.logical_and(hue > low, hue < high)
        is_odd = low % 10
        color = pink if is_odd else green
        out[select] *= color
    for low, high in nphusl.chunk(100, 10):  # chunks of the lightness range
        select = np.logical_and(lightness > low, lightness < high)
        out[select] *= min(1, (high / 90.0))
    return out, "watermelon"


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    transforms = reveal_blue, reveal_blue_rgb, hue_watermelon,\
                 reveal_light, reveal_light_rgb
    for t in transforms:
        out, name = t(img)
        imread.imwrite(name + ".jpg", out)


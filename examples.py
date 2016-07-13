import random
import sys
import imread
import numpy as np
import nphusl
from moviepy.editor import VideoClip


def reveal_blue(img):
    """Easy mode! Selecting bluish pixels with the HUSL color space."""
    # convert an integer RGB image to HUSL array of floats
    hsl = nphusl.to_husl(img)
    hue = hsl[..., 0]  # separate out the hue channel
    # create a mask for pixels with hues between 250 and 290 (blue)
    bluish = np.logical_and(hue > 250, hue < 290)
    hsl[..., 2][~bluish] *= 0.5  # halve lightness of non-bluish areas
    return nphusl.to_rgb(hsl), "blue"


def reveal_blue_rgb(img):
    """Hard mode! Selecting blueish pixels without the convenience of the
    HUSL color space."""
    R, G, B = (img[..., n] for n in range(3))  # break out RGB color channels
    # we'll try to create a "bluish" selection by choosing pixels for which
    # the blue channel has a greater value than the others
    bluish = np.logical_and(B > R, B > G)  # no overpowering red or green
    bluish = np.logical_and(bluish, B > 125)  # strong enough blue channel
    out = img.copy()
    out[~bluish] = out[~bluish] * 0.5
    return out, "blue_rgb"


def reveal_light(img):
    hsl = nphusl.to_husl(img)
    lightness = hsl[..., 2]  # just the lightness channel
    dark = lightness < 62
    hsl[..., 2][dark] = 0  # darkish areas to completely dark
    return nphusl.to_rgb(hsl), "light"


def reveal_light_rgb(img):
    R, G, B = (img[..., n] for n in range(3))
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    dark = luminance < 140
    out = img.copy()
    out[dark] = 0x00  # change selection to black
    return out, "light_rgb"


def highlight_saturation(img):
    hsl = nphusl.to_husl(img)
    hsl[..., 2][hsl[..., 1] < 80] = 0
    return nphusl.to_rgb(hsl), "saturation"


def hue_watermelon(img):
    hsl = nphusl.to_husl(img)
    hue, saturation, lightness = (hsl[..., n] for n in range(3))
    hue_out = hue.copy()
    pink = 360  # very end of the H spectrum
    green = 130
    chunksize = 5
    for low, high in nphusl.chunk(360, chunksize):  # chunks of the hue range
        select = np.logical_and(hue > low, hue < high)
        is_odd = low % (chunksize * 2)
        color = pink if is_odd else green
        hue_out[select] = color
    hue[:] = hue_out
    return nphusl.to_rgb(hsl), "watermelon"


def melonize(img, n_frames):
    hsl = nphusl.to_husl(img)
    hue, sat, lit = (hsl[..., n] for n in range(3))
    #sat[:] = 99
    pink = 360  # very end of the H spectrum
    green = 130

    def gen_chunksizes():
        yield from range(1, 100):
        yield from range(100, 1, -1):

    for chunksize in gen_chunksizes():
        hsl_out = hsl.copy()
        hue_out, sat_out, lit_out = (hsl_out[..., i] for i in range(3))
        for low, high in nphusl.chunk(100, chunksize):  # chunks of the hue range
            select = np.logical_and(lit > low, lit < high)
            is_odd = low % (chunksize * 2)
            color = pink if is_odd else green
            hue_out[select] = color
            select = np.logical_and(lit > (low - 1), lit < low)
            select = np.logical_and(select, lit > 60)
            ave = (low + high) / 2
            select = np.logical_and(lit > (ave - 2), lit < (ave + 2))
            sat_out[select] = 100
        yield nphusl.to_rgb(hsl_out)


def microwave(img):
    hsl = nphusl.to_husl(img)
    hue = hsl[..., 0]
    rows, cols = hue.shape
    yield nphusl.to_rgb(hsl)
    while True:
        for chunk, ((rs, re), (cs, ce)) in nphusl.chunk_img(hue, chunksize=8):
            hue_left = hue[rs, cs-1]
            hue_up = hue[rs-1, cs]
            this_hue = chunk[0, 0]
            new_hue = (-random.randrange(30, 50) * (hue_up / 360)
                       -10*random.randrange(1, 10) * (hue_left / 360))
            new_hue = (15*this_hue + 2*new_hue) / 17
            chunk[:] = new_hue
        np.mod(hue, 360, out=hue)
        yield nphusl.to_rgb(hsl)


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    transforms = reveal_blue, reveal_blue_rgb, hue_watermelon,\
                 reveal_light, reveal_light_rgb, highlight_saturation
    for t in transforms:
        out, name = t(img)
        imread.imwrite(name + ".jpg", out.astype(np.uint8))

    n_frames = 300
    fps = 50
    duration = n_frames / fps
    #frames = microwave(img)
    #animation = VideoClip(lambda _: next(frames), duration=duration)
    #animation.write_gif("video2.gif", fps=fps, opt="OptimizePlus")
    frames = melonize(img, n_frames)
    animation = VideoClip(lambda _: next(frames), duration=duration)
    animation.write_gif("melonized.gif", fps=fps, opt="OptimizePlus")


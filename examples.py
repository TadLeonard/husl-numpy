import imread
import huslnp


def reveal_red(img):
    hue = nphusl.to_hue(img)
    img[hue < 50] = (255, 0, 0)
    return img


if __name__ == "__main__":
    filename = sys.argv[1]
    img = imread.imread(filename)
    reveal_red(img)
    imread.imwrite("{}-{}".format("red_areas_", filename))

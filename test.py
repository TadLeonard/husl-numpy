import husl_numpy as husl
import imread


def print_husl():
    img = imread.imread("examples/gelface.jpg")
    img_float = img / 255.0
    print(husl.rgb_to_husl(img_float))


if __name__ == "__main__":
    print_husl()

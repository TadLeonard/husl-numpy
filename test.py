import husl_numpy as husl
import husl as old_husl
import imread


def print_husl():
    img = imread.imread("examples/gelface.jpg")
    img_float = img / 255.0
    new = husl.rgb_to_husl(img_float)
    old = old_husl.rgb_to_husl(*img_float[0, 0])
    print(old, new[0, 0])
    


if __name__ == "__main__":
    print_husl()

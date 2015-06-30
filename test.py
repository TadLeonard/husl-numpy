import numpy as np
import imread
import husl_numpy as husl
import husl as old_husl


def test_f():
    val_a = old_husl.epsilon + 0.4
    val_b = old_husl.epsilon - 0.003
    assert old_husl.f(val_a) == husl.f(np.array([val_a]))[0]
    assert old_husl.f(val_b) == husl.f(np.array([val_b]))[0]
    print(old_husl.f(val_a), husl.f(np.array([val_a]))[0])
    print(old_husl.f(val_b), husl.f(np.array([val_b]))[0])


def print_husl():
    img = imread.imread("examples/gelface.jpg")
    img_float = img / 255.0
    new = husl.rgb_to_husl(img_float)
    old = old_husl.rgb_to_husl(*img_float[0, 0])
    print(old, new[0, 0])
    


if __name__ == "__main__":
    print_husl()

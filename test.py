import numpy as np
import imread
import husl_numpy as husl
import husl as old_husl


def test_rgb_to_xyz():
    tests = [[0, 0, 0], [1, 1, 1], [0.52, 0.1, 0.25]]
    for rgb in tests:
        nd = np.ndarray((1, 3), float)
        nd[:] = rgb
        nd /= 255.0
        assert np.all(husl.rgb_to_xyz(nd)[0] == old_husl.rgb_to_xyz(rgb))


def test_f():
    val_a = old_husl.epsilon + 0.4
    val_b = old_husl.epsilon - 0.003
    assert old_husl.f(val_a) == husl.f(np.array([val_a]))[0]
    assert old_husl.f(val_b) == husl.f(np.array([val_b]))[0]


def test_channel():
    a = np.zeros((40, 40, 3))
    a[:] = (20, 30, 40)  # r = 20, b = 30, g = 40
    assert np.all(husl._channel(a, 0) == 20)
    assert np.all(husl._channel(a, 1) == 30)
    assert np.all(husl._channel(a, 2) == 40)


def test_channel_assignment():
    a = np.zeros((40, 40, 3))
    a[:] = (20, 30, 40)  # r = 20, b = 30, g = 40
    husl._channel(a, 0)[:] = 1
    husl._channel(a, 1)[:] = 2
    husl._channel(a, 2)[:] = 3
    assert np.all(husl._channel(a, 0) == 1)
    assert np.all(husl._channel(a, 1) == 2)
    assert np.all(husl._channel(a, 2) == 3)
    


def print_husl():
    img = imread.imread("examples/gelface.jpg")
    img_float = img / 255.0
    new = husl.rgb_to_husl(img_float)
    old = old_husl.rgb_to_husl(*img_float[0, 0])
    print(old, new[0, 0])
    


if __name__ == "__main__":
    print_husl()

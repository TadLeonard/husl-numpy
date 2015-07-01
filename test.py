import numpy as np
import imread
import husl_numpy as husl
import husl as old_husl


def test_rgb_to_xyz():
    rgb_arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.52, 0.1, 0.25],
               [0.7, 0.8, 0.8], [0.9, 0.9, 0.1], [0.0, 1.0, 0.1]]
    rgb_arr = np.asarray(rgb_arr)
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    for xyz, rgb in zip(xyz_arr, rgb_arr):
        diff =  xyz - old_husl.rgb_to_xyz(rgb)
        assert np.all(diff < 0.0001)


def test_to_linear():
    val_a = 0.055 + 0.330
    val_b = 0.055 - 0.020
    assert old_husl.to_linear(val_a) == husl._to_linear(np.array([val_a]))[0]
    assert old_husl.to_linear(val_b) == husl._to_linear(np.array([val_b]))[0]


def test_dot():
    a = np.array([0.1, 0.2, 0.3])
    b = np.ndarray((3, 3))
    b[:] = a
    c = np.ndarray((3, 3, 3))
    c[:] = a
    for arr in (b, c):
        _check_dot(arr)


def _check_dot(test_array):
    m_inv = old_husl.m_inv
    new_dot = husl._dot_product(m_inv, test_array)
    flat_input = test_array.reshape((test_array.size // 3, 3))
    flat_output = new_dot.reshape((new_dot.size // 3, 3))
    for new_dot, rgb in zip(flat_output, flat_input):
        old_dot = list(map(lambda row: old_husl.dot_product(row, rgb), m_inv))
        assert np.all(new_dot == old_dot)


def test_f():
    val_a = old_husl.epsilon + 0.4
    val_b = old_husl.epsilon - 0.003
    assert old_husl.f(val_a) == husl._f(np.array([val_a]))[0]
    assert old_husl.f(val_b) == husl._f(np.array([val_b]))[0]


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

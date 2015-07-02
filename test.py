import numpy as np
import imread
import husl_numpy as husl
import husl as old_husl


def test_rgb_to_husl():
    img = imread.imread("examples/gelface.jpg")
    img /= 255.0
    husl_new = husl.rgb_to_husl(img)
    for row in range(husl_new.shape[0]):
        for col in range(husl_new.shape[1]):
            husl_old = old_husl.rgb_to_husl(*img[row, col]) 
            assert np.all(husl_new[row, col] == husl_old)


def test_lch_to_husl():
    rgb_arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.52, 0.1, 0.25],
               [0.7, 0.8, 0.8], [0.9, 0.9, 0.1], [0.0, 1.0, 0.1]]
    rgb_arr = np.asarray(rgb_arr)
    lch_arr = husl.rgb_to_lch(rgb_arr)
    hsl_arr = husl.lch_to_husl(lch_arr)
    for hsl, lch in zip(hsl_arr, lch_arr):
        diff =  hsl - old_husl.lch_to_husl(lch)
        assert np.all(diff < 0.0001)


#def test_bounds():
    


def test_ray_length():
    thetas = np.asarray([0.1, 4.0, 44.4, 500.2])
    lines = (1.0, 4.0), (0.01, 2.0), (3.5, 0.0), (0.0, 0.0)
    new_lens = [husl._ray_length(thetas, l) for l in lines]
    for i, (new_len, theta, line) in enumerate(zip(new_lens, thetas, lines)):
        old_len = old_husl.length_of_ray_until_intersect(theta, line)
        if new_len[i] > -0.0001 and np.isfinite(new_len[i]):
            assert new_len[i] == old_len
        elif old_len is not None:
            assert False, "Expected a valid length from husl._ray_length"


def test_luv_to_lch():
    rgb_arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.52, 0.1, 0.25],
               [0.7, 0.8, 0.8], [0.9, 0.9, 0.1], [0.0, 1.0, 0.1]]
    rgb_arr = np.asarray(rgb_arr)
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    lch_arr = husl.luv_to_lch(luv_arr)
    for lch, luv in zip(lch_arr, luv_arr):
        diff =  lch - old_husl.luv_to_lch(luv)
        assert np.all(diff < 0.0001)


def test_rgb_to_lch():
    rgb_arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.52, 0.1, 0.25],
               [0.7, 0.8, 0.8], [0.9, 0.9, 0.1], [0.0, 1.0, 0.1]]
    rgb_arr = np.asarray(rgb_arr)
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    lch_arr = husl.luv_to_lch(luv_arr)
    lch_arr2 = husl.rgb_to_lch(rgb_arr)
    assert np.all(lch_arr == lch_arr2)
    for lch, luv in zip(lch_arr2, luv_arr):
        diff =  lch - old_husl.luv_to_lch(luv)
        assert np.all(diff < 0.0001)


def test_xyz_to_luv():
    rgb_arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.52, 0.1, 0.25],
               [0.7, 0.8, 0.8], [0.9, 0.9, 0.1], [0.0, 1.0, 0.1]]
    rgb_arr = np.asarray(rgb_arr)
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    for luv, xyz in zip(luv_arr, xyz_arr):
        diff =  luv - old_husl.xyz_to_luv(xyz)
        assert np.all(diff < 0.0001)


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

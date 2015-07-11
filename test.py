import sys
import numpy as np
import imread
import husl_numpy as husl
import husl as old_husl


def test_rgb_to_husl():
    rgb_arr = _img()[:, 0]
    husl_new = husl.rgb_to_husl(rgb_arr)
    for hsl, rgb in zip(husl_new, rgb_arr):
        husl_old = old_husl.rgb_to_husl(*rgb)
        assert _diff(hsl, husl_old)


def test_rgb_to_husl_3d():
    rgb_arr = _img()
    husl_new = husl.rgb_to_husl(rgb_arr)
    for row in range(husl_new.shape[0]):
        for col in range(husl_new.shape[1]):
            husl_old = old_husl.rgb_to_husl(*rgb_arr[row][col])
            assert _diff(husl_new[row, col], husl_old)


def test_lch_to_husl():
    rgb_arr = _img()[:, 0]
    lch_arr = husl.rgb_to_lch(rgb_arr)
    hsl_from_lch_arr = husl.lch_to_husl(lch_arr)
    hsl_from_rgb_arr = husl.rgb_to_husl(rgb_arr)
    arrays = zip(hsl_from_rgb_arr, hsl_from_lch_arr, lch_arr, rgb_arr)
    for hsl_r, hsl_l, lch, rgb in arrays:
        old_lch = old_husl.rgb_to_lch(*rgb)
        hsl_old = old_husl.lch_to_husl(old_lch)
        assert _diff(lch, old_lch)
        assert _diff(hsl_l, hsl_old)
        assert _diff(hsl_r, hsl_old)


def test_lch_to_husl_3d():
    img = _img()
    lch_new = husl.rgb_to_lch(img)
    hsl_new = husl.lch_to_husl(lch_new)
    for row in range(lch_new.shape[0]):
        for col in range(lch_new.shape[1]):
            lch_old = old_husl.rgb_to_lch(*img[row, col])
            assert _diff(lch_old, lch_new[row, col])
            hsl_old = old_husl.lch_to_husl(lch_old)
            assert _diff(hsl_new[row, col], hsl_old) 


def test_max_lh_for_chroma():
    rgb_arr = _img()[:, 0]
    lch_arr = husl.rgb_to_lch(rgb_arr)
    with np.errstate(invalid="ignore"):
        mx_arr = husl._max_lh_chroma(lch_arr)
    arrays = zip(mx_arr, lch_arr, rgb_arr)
    for mx, lch, rgb in arrays:
        try:
            mx_old = old_husl.max_chroma_for_LH(float(lch[0]), float(lch[2]))
        except ZeroDivisionError:
            # NOTE: Divide by zero is avoided in husl.py
            # we're taking a backdoor here by using max_chroma_for_LH directly 
            assert np.isinf(mx)
        else:
            assert _diff(mx, mx_old)


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
    rgb_arr = _img()[:, 0]
    rgb_arr = _img()
    rgb_arr = rgb_arr.reshape((rgb_arr.size // 3, 3))
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    lch_arr = husl.luv_to_lch(luv_arr)
    for i in range(rgb_arr.shape[0]):
        xyz = old_husl.rgb_to_xyz(rgb_arr[i])
        assert _diff(xyz, xyz_arr[i])
        luv = old_husl.xyz_to_luv(xyz)
        assert _diff(luv, luv_arr[i])
        lch = old_husl.luv_to_lch(luv)
        assert _diff(lch, lch_arr[i])


def test_luv_to_lch_3d():
    img = _img()
    xyz_arr = husl.rgb_to_xyz(img)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    lch_new = husl.luv_to_lch(luv_arr)
    for row in range(lch_new.shape[0]):
        for col in range(lch_new.shape[1]):
            lch_old = old_husl.rgb_to_lch(*img[row, col]) 
            assert _diff(lch_new[row, col], lch_old)


def test_rgb_to_lch():
    rgb_arr = _img()[:, 0]
    lch_arr = husl.rgb_to_lch(rgb_arr)
    for lch, rgb in zip(lch_arr, rgb_arr):
        diff = lch - old_husl.rgb_to_lch(*rgb)
        assert _diff(lch, old_husl.rgb_to_lch(*rgb))


def test_rgb_to_lch_3d():
    rgb_arr = _img()
    lch_arr = husl.rgb_to_lch(rgb_arr)
    for row in range(lch_arr.shape[0]):
        for col in range(lch_arr.shape[1]):
            old_lch = old_husl.rgb_to_lch(*rgb_arr[row, col])
            assert _diff(lch_arr[row, col], old_lch)


def test_rgb_to_lch_chain():
    rgb_arr = _img()[:, 0]
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    lch_arr = husl.luv_to_lch(luv_arr)
    lch_arr2 = husl.rgb_to_lch(rgb_arr)
    assert np.all(lch_arr == lch_arr2)


def test_xyz_to_luv():
    rgb_arr = _img()[:, 0]
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    for luv, xyz in zip(luv_arr, xyz_arr):
        diff = luv - old_husl.xyz_to_luv(xyz)
        assert _diff(luv, old_husl.xyz_to_luv(xyz))


def test_xyz_to_luv_3d():
    rgb_arr = _img()
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    luv_arr = husl.xyz_to_luv(xyz_arr)
    for row in range(luv_arr.shape[0]):
        for col in range(luv_arr.shape[1]):
            old_luv = old_husl.xyz_to_luv(xyz_arr[row, col]) 
            assert _diff(luv_arr[row, col], old_luv)


def test_rgb_to_xyz():
    rgb_arr = _img()[:, 0]
    xyz_arr = husl.rgb_to_xyz(rgb_arr)
    for xyz, rgb in zip(xyz_arr, rgb_arr):
        diff = xyz - old_husl.rgb_to_xyz(rgb)
        assert _diff(xyz, old_husl.rgb_to_xyz(rgb))


def test_rgb_to_xyz_3d():
    img = _img()
    xyz_arr = husl.rgb_to_xyz(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            assert _diff(xyz_arr[row, col],
                         old_husl.rgb_to_xyz(img[row, col]))


def test_to_linear():
    a = 0.055 + 0.330
    b = 0.055 - 0.020
    c = 0.0
    d = 1.0
    for val in (a, b, c, d):
        assert old_husl.to_linear(val) == husl._to_linear(np.array([val]))[0]


def test_dot():
    a = np.array([0.1, 0.2, 0.3])
    b = np.ndarray((3, 3))
    b[:] = a
    c = np.ndarray((6, 6, 3))
    c[:] = a
    c[0, 1] = (0.0, 1.0, 0.0)
    c[0, 3] = (1.0, 1.0, 1.0)
    c[1, 1] = (0.0, 0.0, 0.0)
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
    

def _diff(arr_a, arr_b, diff=0.0000000001):
    return np.all(np.abs(arr_a - arr_b) < diff)


IMG_CACHED = [None]


def _img():
    if IMG_CACHED[0] is None:
        i = imread.imread("examples/gelface.jpg") / 255.0
        i = i[::4, ::4]
        i[0] = 0.0  # ensure we get all black
        i[1] = 1.0  # ensure we get all white
        IMG_CACHED[0] = i
    return IMG_CACHED[0]


@profile
def main():
    img_int = imread.imread(sys.argv[1])
    print(img_int[0, 0])
    img_float = np.zeros(img_int.shape, dtype=np.float)
    np.divide(img_int, 255.0, out=img_float)  
    avg_rows = np.average(img_float[:, ::2], axis=1)
    hsl = husl.rgb_to_husl(avg_rows)
    print(hsl[..., 2])
    print(np.average(img_int[0], axis=1))
    img_int[hsl[..., 2] > 95] = 0
    imread.imwrite("hork.jpg", img_int)


if __name__ == "__main__":
    main()

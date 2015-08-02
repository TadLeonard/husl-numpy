import sys
import numpy as np
import imread
import nphusl
import husl  # the original husl-colors.org library


### Tests for conversion in the RGB -> HUSL direction


def test_to_husl():
    img = _img()
    rgb_arr = img  * 255
    husl_new = nphusl.to_husl(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            husl_old = husl.rgb_to_husl(*img[row, col])
            assert _diff(husl_new[row, col], husl_old)


def test_to_husl_gray():
    img = _img()
    img[..., 1] = img[..., 0]
    img[..., 2] = img[..., 0]
    rgb_arr = img[..., 0] * 255  # single channel
    print(rgb_arr.shape)
    husl_new = nphusl.to_husl(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            husl_old = husl.rgb_to_husl(*img[row, col])
            assert _diff(husl_new[row, col], husl_old)


def test_to_hue():
    img = _img()
    rgb_arr = img  * 255
    hue_new = nphusl.to_hue(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            husl_old = husl.rgb_to_husl(*img[row, col])[0]
            assert _diff(hue_new[row, col], husl_old)


def test_to_hue_gray():
    img = _img()
    img[..., 1] = img[..., 0]
    img[..., 2] = img[..., 0]
    rgb_arr = img[..., 0] * 255  # single channel
    print(rgb_arr.shape)
    hue_new = nphusl.to_hue(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            hue_old = husl.rgb_to_husl(*img[row, col])[0]
            assert _diff(hue_new[row, col], hue_old)


def test_rgb_to_husl():
    rgb_arr = _img()[:, 0]
    husl_new = nphusl.rgb_to_husl(rgb_arr)
    for hsl, rgb in zip(husl_new, rgb_arr):
        husl_old = husl.rgb_to_husl(*rgb)
        assert _diff(hsl, husl_old)


def test_rgb_to_husl_3d():
    rgb_arr = _img()
    husl_new = nphusl.rgb_to_husl(rgb_arr)
    for row in range(husl_new.shape[0]):
        for col in range(husl_new.shape[1]):
            husl_old = husl.rgb_to_husl(*rgb_arr[row][col])
            assert _diff(husl_new[row, col], husl_old)


def test_lch_to_husl():
    rgb_arr = _img()[:, 0]
    lch_arr = nphusl.rgb_to_lch(rgb_arr)
    hsl_from_lch_arr = nphusl.lch_to_husl(lch_arr)
    hsl_from_rgb_arr = nphusl.rgb_to_husl(rgb_arr)
    arrays = zip(hsl_from_rgb_arr, hsl_from_lch_arr, lch_arr, rgb_arr)
    for hsl_r, hsl_l, lch, rgb in arrays:
        old_lch = husl.rgb_to_lch(*rgb)
        hsl_old = husl.lch_to_husl(old_lch)
        assert _diff(lch, old_lch)
        assert _diff(hsl_l, hsl_old)
        assert _diff(hsl_r, hsl_old)


def test_lch_to_husl_3d():
    img = _img()
    lch_new = nphusl.rgb_to_lch(img)
    hsl_new = nphusl.lch_to_husl(lch_new)
    for row in range(lch_new.shape[0]):
        for col in range(lch_new.shape[1]):
            lch_old = husl.rgb_to_lch(*img[row, col])
            assert _diff(lch_old, lch_new[row, col])
            hsl_old = husl.lch_to_husl(lch_old)
            assert _diff(hsl_new[row, col], hsl_old) 


def test_max_lh_for_chroma():
    rgb_arr = _img()[:, 0]
    lch_arr = nphusl.rgb_to_lch(rgb_arr)
    with np.errstate(invalid="ignore"):
        mx_arr = nphusl._max_lh_chroma(lch_arr)
    arrays = zip(mx_arr, lch_arr, rgb_arr)
    for mx, lch, rgb in arrays:
        try:
            mx_old = husl.max_chroma_for_LH(float(lch[0]), float(lch[2]))
        except ZeroDivisionError:
            # NOTE: Divide by zero is avoided in nphusl.py
            # we're taking a backdoor here by using max_chroma_for_LH directly 
            assert np.isinf(mx)
        else:
            assert _diff(mx, mx_old)


def test_ray_length():
    thetas = np.asarray([0.1, 4.0, 44.4, 500.2])
    lines = (1.0, 4.0), (0.01, 2.0), (3.5, 0.0), (0.0, 0.0)
    new_lens = [nphusl._ray_length(thetas, l) for l in lines]
    for i, (new_len, theta, line) in enumerate(zip(new_lens, thetas, lines)):
        old_len = husl.length_of_ray_until_intersect(theta, line)
        if new_len[i] > -0.0001 and np.isfinite(new_len[i]):
            assert new_len[i] == old_len
        elif old_len is not None:
            assert False, "Expected a valid length from nphusl._ray_length"


def test_luv_to_lch():
    rgb_arr = _img()[:, 0]
    rgb_arr = _img()
    rgb_arr = rgb_arr.reshape((rgb_arr.size // 3, 3))
    xyz_arr = nphusl.rgb_to_xyz(rgb_arr)
    luv_arr = nphusl.xyz_to_luv(xyz_arr)
    lch_arr = nphusl.luv_to_lch(luv_arr)
    for i in range(rgb_arr.shape[0]):
        xyz = husl.rgb_to_xyz(rgb_arr[i])
        assert _diff(xyz, xyz_arr[i])
        luv = husl.xyz_to_luv(xyz)
        assert _diff(luv, luv_arr[i])
        lch = husl.luv_to_lch(luv)
        assert _diff(lch, lch_arr[i])


def test_luv_to_lch_3d():
    img = _img()
    xyz_arr = nphusl.rgb_to_xyz(img)
    luv_arr = nphusl.xyz_to_luv(xyz_arr)
    lch_new = nphusl.luv_to_lch(luv_arr)
    for row in range(lch_new.shape[0]):
        for col in range(lch_new.shape[1]):
            lch_old = husl.rgb_to_lch(*img[row, col]) 
            assert _diff(lch_new[row, col], lch_old)


def test_rgb_to_lch():
    rgb_arr = _img()[:, 0]
    lch_arr = nphusl.rgb_to_lch(rgb_arr)
    for lch, rgb in zip(lch_arr, rgb_arr):
        diff = lch - husl.rgb_to_lch(*rgb)
        assert _diff(lch, husl.rgb_to_lch(*rgb))


def test_rgb_to_lch_3d():
    rgb_arr = _img()
    lch_arr = nphusl.rgb_to_lch(rgb_arr)
    for row in range(lch_arr.shape[0]):
        for col in range(lch_arr.shape[1]):
            old_lch = husl.rgb_to_lch(*rgb_arr[row, col])
            assert _diff(lch_arr[row, col], old_lch)


def test_rgb_to_lch_chain():
    rgb_arr = _img()[:, 0]
    xyz_arr = nphusl.rgb_to_xyz(rgb_arr)
    luv_arr = nphusl.xyz_to_luv(xyz_arr)
    lch_arr = nphusl.luv_to_lch(luv_arr)
    lch_arr2 = nphusl.rgb_to_lch(rgb_arr)
    assert np.all(lch_arr == lch_arr2)


def test_xyz_to_luv():
    rgb_arr = _img()[:, 0]
    xyz_arr = nphusl.rgb_to_xyz(rgb_arr)
    luv_arr = nphusl.xyz_to_luv(xyz_arr)
    for luv, xyz in zip(luv_arr, xyz_arr):
        diff = luv - husl.xyz_to_luv(xyz)
        assert _diff(luv, husl.xyz_to_luv(xyz))


def test_xyz_to_luv_3d():
    rgb_arr = _img()
    xyz_arr = nphusl.rgb_to_xyz(rgb_arr)
    luv_arr = nphusl.xyz_to_luv(xyz_arr)
    for row in range(luv_arr.shape[0]):
        for col in range(luv_arr.shape[1]):
            old_luv = husl.xyz_to_luv(xyz_arr[row, col]) 
            assert _diff(luv_arr[row, col], old_luv)


def test_rgb_to_xyz():
    rgb_arr = _img()[:, 0]
    xyz_arr = nphusl.rgb_to_xyz(rgb_arr)
    for xyz, rgb in zip(xyz_arr, rgb_arr):
        diff = xyz - husl.rgb_to_xyz(rgb)
        assert _diff(xyz, husl.rgb_to_xyz(rgb))


def test_rgb_to_xyz_3d():
    img = _img()
    xyz_arr = nphusl.rgb_to_xyz(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            assert _diff(xyz_arr[row, col],
                         husl.rgb_to_xyz(img[row, col]))


def test_to_linear():
    a = 0.055 + 0.330
    b = 0.055 - 0.020
    c = 0.0
    d = 1.0
    for val in (a, b, c, d):
        assert husl.to_linear(val) == nphusl._to_linear(np.array([val]))[0]


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
    m_inv = husl.m_inv
    new_dot = nphusl._dot_product(m_inv, test_array)
    flat_input = test_array.reshape((test_array.size // 3, 3))
    flat_output = new_dot.reshape((new_dot.size // 3, 3))
    for new_dot, rgb in zip(flat_output, flat_input):
        old_dot = list(map(lambda row: husl.dot_product(row, rgb), m_inv))
        assert np.all(new_dot == old_dot)


def test_f():
    val_a = husl.epsilon + 0.4
    val_b = husl.epsilon - 0.003
    assert husl.f(val_a) == nphusl._f(np.array([val_a]))[0]
    assert husl.f(val_b) == nphusl._f(np.array([val_b]))[0]


### Tests for conversion in HUSL -> RGB direction


def test_to_rgb():
    img = _img()
    int_img = np.ndarray(shape=img.shape, dtype=np.uint8)
    int_img[:] = img * 255
    husl = nphusl.rgb_to_husl(img)
    rgb = nphusl.to_rgb(husl)
    assert np.all(rgb == int_img)


def test_husl_to_rgb():
    img = _img()
    husl = nphusl.rgb_to_husl(img)
    rgb = nphusl.husl_to_rgb(husl)
    assert _diff(img, rgb)


def test_lch_to_rgb():
    img = _img()
    lch = nphusl.rgb_to_lch(img)
    rgb = nphusl.lch_to_rgb(lch)
    assert _diff(rgb, img)


def test_xyz_to_rgb():
    img = _img()
    xyz = nphusl.rgb_to_xyz(img)
    rgb = nphusl.xyz_to_rgb(xyz)
    assert _diff(img, rgb)


def test_from_linear():
    a = 0.003 + 0.330
    b = 0.003 - 0.0020
    for val in (a, b):
        assert husl.from_linear(val) == nphusl._from_linear(np.array([val]))[0]


def test_husl_to_lch():
    img = _img()
    lch = nphusl.rgb_to_lch(img)
    husl = nphusl.rgb_to_husl(img)
    lch_2 = nphusl.husl_to_lch(husl)
    assert _diff (lch, lch_2)
   

def test_luv_to_xyz():
    img = _img()
    xyz = nphusl.rgb_to_xyz(img)
    luv = nphusl.xyz_to_luv(xyz)
    xyz_2 = nphusl.luv_to_xyz(luv)
    assert _diff(xyz_2, xyz)


def test_lch_to_luv():
    img = _img()
    lch = nphusl.rgb_to_lch(img)
    luv = nphusl.lch_to_luv(lch)  # we're testing this
    xyz = nphusl.rgb_to_xyz(img)
    luv_2 = nphusl.xyz_to_luv(xyz)
    lch_2 = nphusl.luv_to_lch(luv_2)
    assert _diff(lch_2, lch)  # just a sanity check on RGB -> LCH
    assert _diff(luv, luv_2)


def test_f_inv():
    val_a = 8 + 1.5
    val_b = 8 - 3.5
    assert husl.f_inv(val_a) == nphusl._f_inv(np.array([val_a]))[0]
    assert husl.f_inv(val_b) == nphusl._f_inv(np.array([val_b]))[0]
 

### Tests for convenience functions, utilities


def test_channel():
    a = np.zeros((40, 40, 3))
    a[:] = (20, 30, 40)  # r = 20, b = 30, g = 40
    assert np.all(nphusl._channel(a, 0) == 20)
    assert np.all(nphusl._channel(a, 1) == 30)
    assert np.all(nphusl._channel(a, 2) == 40)


def test_channel_assignment():
    a = np.zeros((40, 40, 3))
    a[:] = (20, 30, 40)  # r = 20, b = 30, g = 40
    nphusl._channel(a, 0)[:] = 1
    nphusl._channel(a, 1)[:] = 2
    nphusl._channel(a, 2)[:] = 3
    assert np.all(nphusl._channel(a, 0) == 1)
    assert np.all(nphusl._channel(a, 1) == 2)
    assert np.all(nphusl._channel(a, 2) == 3)
    

def test_chunk():
    """Ensures that the chunk iterator breaks an image into NxN squares"""
    img = _img()
    assert not np.all(img[:10, :10] == 0)
    for i, (chunk, _) in enumerate(nphusl.chunk_img(img, 10)):
        chunk[:] = i
    for i, (chunk, _) in enumerate(nphusl.chunk_img(img, 10)):
        assert np.all(chunk == i)
    assert np.all(img[:10, :10] == 0)


def test_chunk_transform():
    img = _img()
    assert np.sum(img == 0) != 0  # assert that there are some 0,0,0 pixels
    zero_at = np.where(img == 0)

    def transform(chunk):
        chunk[chunk == 0] = 100
        return chunk

    chunks = nphusl.chunk_img(img, 10)
    nphusl.chunk_transform(transform, chunks, img)
    assert np.sum(img == 0) == 0
    assert np.all(img[zero_at] == 100)


def test_transform_rgb():
    img = _img()
    as_husl = nphusl.rgb_to_husl(img / 255.0)
    chunk_husl = nphusl.transform_rgb(img, nphusl.rgb_to_husl, 10)
    assert np.all(as_husl == chunk_husl)     


def test_to_hue():
    img = _img()
    as_husl = nphusl.rgb_to_husl(img / 255.0)
    just_hue = nphusl.to_hue(img)
    assert np.all(as_husl[..., 0] == just_hue)


def _diff(arr_a, arr_b, diff=0.0000000001):
    return np.all(np.abs(arr_a - arr_b) < diff)


IMG_CACHED = [None]


def _img():
    if IMG_CACHED[0] is None:
        i = imread.imread("images/gelface.jpg") / 255.0
        i = i[::4, ::4]
        i[0] = 0.0  # ensure we get all black
        i[1] = 1.0  # ensure we get all white
        IMG_CACHED[0] = i
    return IMG_CACHED[0]


def main():
    img_int = imread.imread(sys.argv[1])[:1000, :1000]
    img_float = img_int / 255.0

    out = np.zeros(img_float.shape, dtype=np.float)
    chunks = nphusl.chunk_img(img_float)
    nphusl.chunk_transform(nphusl.rgb_to_husl, chunks, out)
    nphusl.transform_rgb(img_int, nphusl.rgb_to_husl)
    nphusl.to_hue(img_int)


if __name__ == "__main__":
    main()

import argparse
import sys
import functools

import imread
import numpy as np
import pytest

import nphusl
from nphusl.nphusl import _channel
from nphusl import transform
from nphusl.transform import ensure_int, ensure_float
from nphusl import nphusl as _nphusl
import husl  # the original husl-colors.org library
from enum import Enum


np.set_printoptions(threshold=np.inf)


### Tests for conversion in the RGB -> HUSL direction


class Opt(str, Enum):
    cython = "cython"
    simd = "simd"
    numexpr = "numexpr"


_optimized = set()


def try_optimizations(*opts):
    opts = opts or (Opt.cython, Opt.simd, Opt.numexpr)

    def wrapped(fn):
        _optimized.add(fn.__name__)

        def with_numpy(*args, **kwargs):
            with nphusl.numpy_enabled(back_to_std=True):
                fn(*args, **kwargs)

        def with_expr(*args, **kwargs):
            assert hasattr(nphusl, "_numexpr_opt")
            with nphusl.numexpr_enabled(back_to_std=True):
                fn(*args, **kwargs)

        def with_cyth(*args, **kwargs):
            assert hasattr(nphusl, "_cython_opt")
            with nphusl.cython_enabled(back_to_std=True):
                fn(*args, **kwargs)

        def with_simd(*args, **kwargs):
            assert hasattr(nphusl, "_simd_opt")
            with nphusl.simd_enabled(back_to_std=True):
                fn(*args, **kwargs)

        globals()[fn.__name__ + "__with_numpy"] = with_numpy
        if Opt.numexpr in opts:
            globals()[fn.__name__ + "__with_numexpr"] = with_expr
        if Opt.cython in opts:
            globals()[fn.__name__ + "__with_cython"] = with_cyth
        if Opt.simd in opts:
            globals()[fn.__name__ + "__with_simd"] = with_simd

        return with_numpy
    return wrapped


@try_optimizations()
def test_to_husl_2d():
    img = np.ascontiguousarray(_img()[:, 22])
    float_img = transform.ensure_rgb_float(img)
    husl_new = nphusl.to_husl(img)
    for row in range(img.shape[0]):
        husl_old = husl.rgb_to_husl(*float_img[row])
        assert _diff_husl(husl_new[row], husl_old)


@try_optimizations()
def test_to_husl_3d():
    img = _img()
    float_img = transform.ensure_rgb_float(img)
    husl_new = nphusl.to_husl(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            husl_old = _ref_to_husl(float_img[row, col])
            assert _diff_husl(husl_new[row, col], husl_old)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_husl_gray():
    img = _img()
    img[..., 1] = img[..., 0]
    img[..., 2] = img[..., 0]
    rgb_arr = img[..., 0] * 255  # single channel
    husl_new = nphusl.to_husl(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            husl_old = _ref_to_husl(img[row, col])
            assert _diff_husl(husl_new[row, col], husl_old)


@try_optimizations()
def test_to_husl_gray_3d():
    img = _img()
    img[..., 1] = img[..., 0]  # makes things gray
    img[..., 2] = img[..., 0]  # makes things gray
    img_float = transform.ensure_rgb_float(img)
    husl_new = nphusl.to_husl(img)
    was_wrong = False
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            husl_old = husl.rgb_to_husl(*img_float[row, col])
            a = husl.husl_to_rgb(*husl_old)
            b = husl.husl_to_rgb(*husl_new[row, col])
            a = np.asarray(a)
            b = np.asarray(b)
            i = row*img.shape[1]*3 + col*3
            assert _diff_husl(husl_new[row, col], husl_old)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_hue_vs_old():
    img = _img()
    hue_new = nphusl.to_hue(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            husl_old = _ref_to_husl(img[row, col])
            diff = 5.0 if husl_old[1] < 1 else 0.1
            assert _diff(hue_new[row, col], husl_old[0], diff=diff)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_hue_gray():
    img = _img()
    img[..., 1] = img[..., 0]
    img[..., 2] = img[..., 0]
    rgb_arr = img[..., 0] * 255  # single channel
    hue_new = nphusl.to_hue(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            hue_old = _ref_to_husl(img[row, col])
            diff = 5.0 if hue_old[1] < 1 else 0.0001
            assert _diff(hue_new[row, col], hue_old[0], diff=diff)


@try_optimizations()
def test_rgb_to_husl():
    rgb_arr = _img()
    husl_new = _nphusl._rgb_to_husl(rgb_arr)
    for row in range(rgb_arr.shape[0]):
        for col in range(rgb_arr.shape[1]):
            husl_old = _ref_to_husl(rgb_arr[row, col])
            assert _diff_husl(husl_new[row, col], husl_old)


@try_optimizations()
def test_rgb_to_husl_3d():
    rgb_arr = np.ascontiguousarray(_img()[:5, :5])
    float_arr = transform.ensure_rgb_float(rgb_arr)
    husl_new = _nphusl._rgb_to_husl(rgb_arr)
    for row in range(husl_new.shape[0]):
        for col in range(husl_new.shape[1]):
            husl_old = husl.rgb_to_husl(*float_arr[row, col])
            assert _diff_husl(husl_new[row, col], husl_old)


@try_optimizations(Opt.numexpr)
def test_lch_to_husl():
    rgb_arr = _img()
    lch_arr = _nphusl._rgb_to_lch(rgb_arr)
    hsl_from_lch_arr = _nphusl._lch_to_husl(lch_arr)
    hsl_from_rgb_arr = _nphusl._rgb_to_husl(rgb_arr)
    assert _diff_husl(hsl_from_lch_arr, hsl_from_rgb_arr)
    for i in range(rgb_arr.shape[0]):
        old_lch = _ref_to_lch(rgb_arr[i, 0])
        assert _diff(lch_arr[i, 0], old_lch)


@try_optimizations(Opt.numexpr)
def test_lch_to_husl_3d():
    img = _img()
    lch_new = _nphusl._rgb_to_lch(img)
    hsl_new = _nphusl._lch_to_husl(lch_new)
    for row in range(lch_new.shape[0]):
        for col in range(lch_new.shape[1]):
            lch_old = _ref_to_lch(img[row, col])
            assert _diff(lch_old, lch_new[row, col])
            hsl_old = husl.lch_to_husl(lch_old)
            assert _diff(hsl_new[row, col], hsl_old)


@try_optimizations(Opt.numexpr)
def test_max_lh_for_chroma():
    rgb_arr = _img()[:, 0]
    lch_arr = _nphusl._rgb_to_lch(rgb_arr)
    with np.errstate(invalid="ignore"):
        mx_arr = _nphusl._max_lh_chroma(lch_arr)
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


@try_optimizations(Opt.numexpr)
def test_ray_length():
    thetas = np.asarray([0.1, 4.0, 44.4, 500.2])
    lines = (1.0, 4.0), (0.01, 2.0), (3.5, 0.0), (0.0, 0.0)
    new_lens = [_nphusl._ray_length(thetas, l) for l in lines]
    for i, (new_len, theta, line) in enumerate(zip(new_lens, thetas, lines)):
        old_len = husl.length_of_ray_until_intersect(theta, line)
        if new_len[i] > -0.0001 and np.isfinite(new_len[i]):
            assert new_len[i] == old_len
        elif old_len is not None:
            assert False, "Expected a valid length from _nphusl._ray_length"


@try_optimizations(Opt.numexpr)
def test_luv_to_lch():
    rgb_arr = _img()[:, 14]
    float_arr = transform.ensure_rgb_float(rgb_arr)
    rgb_arr = rgb_arr.reshape((rgb_arr.size // 3, 3))
    xyz_arr = _nphusl._rgb_to_xyz(rgb_arr)
    luv_arr = _nphusl._xyz_to_luv(xyz_arr)
    lch_arr = _nphusl._luv_to_lch(luv_arr)
    for i in range(rgb_arr.shape[0]):
        xyz = husl.rgb_to_xyz(float_arr[i])
        assert _diff(xyz, xyz_arr[i])
        luv = husl.xyz_to_luv(xyz)
        assert _diff(luv, luv_arr[i])
        lch = husl.luv_to_lch(luv)
        assert _diff(lch, lch_arr[i])


@try_optimizations(Opt.numexpr)
def test_luv_to_lch_3d():
    img = _img()
    xyz_arr = _nphusl._rgb_to_xyz(img)
    luv_arr = _nphusl._xyz_to_luv(xyz_arr)
    lch_new = _nphusl._luv_to_lch(luv_arr)
    for row in range(lch_new.shape[0]):
        for col in range(lch_new.shape[1]):
            lch_old = _ref_to_lch(img[row, col])
            assert _diff(lch_new[row, col], lch_old)


@try_optimizations(Opt.numexpr)
def test_rgb_to_lch():
    rgb_arr = _img()[:, 0]
    lch_arr = _nphusl._rgb_to_lch(rgb_arr)
    for lch, rgb in zip(lch_arr, rgb_arr):
        diff = lch - _ref_to_lch(rgb)
        assert _diff(lch, _ref_to_lch(rgb))


@try_optimizations(Opt.numexpr)
def test_rgb_to_lch_3d():
    rgb_arr = _img()
    lch_arr = _nphusl._rgb_to_lch(rgb_arr)
    for row in range(lch_arr.shape[0]):
        for col in range(lch_arr.shape[1]):
            old_lch = _ref_to_lch(rgb_arr[row, col])
            assert _diff(lch_arr[row, col], old_lch)


@try_optimizations(Opt.numexpr)
def test_rgb_to_lch_chain():
    rgb_arr = _img()[:, 0]
    xyz_arr = _nphusl._rgb_to_xyz(rgb_arr)
    luv_arr = _nphusl._xyz_to_luv(xyz_arr)
    lch_arr = _nphusl._luv_to_lch(luv_arr)
    lch_arr2 = _nphusl._rgb_to_lch(rgb_arr)
    assert np.all(lch_arr == lch_arr2)


@try_optimizations(Opt.numexpr)
def test_xyz_to_luv():
    rgb_arr = _img()[:, 0]
    xyz_arr = _nphusl._rgb_to_xyz(rgb_arr)
    luv_arr = _nphusl._xyz_to_luv(xyz_arr)
    for luv, xyz in zip(luv_arr, xyz_arr):
        diff = luv - husl.xyz_to_luv(xyz)
        assert _diff(luv, husl.xyz_to_luv(xyz))


@try_optimizations(Opt.numexpr)
def test_xyz_to_luv_3d():
    rgb_arr = _img()
    xyz_arr = _nphusl._rgb_to_xyz(rgb_arr)
    luv_arr = _nphusl._xyz_to_luv(xyz_arr)
    for row in range(luv_arr.shape[0]):
        for col in range(luv_arr.shape[1]):
            old_luv = husl.xyz_to_luv(xyz_arr[row, col])
            assert _diff(luv_arr[row, col], old_luv)


@try_optimizations(Opt.numexpr)
def test_rgb_to_xyz():
    rgb_arr = _img()[:, 0]
    float_arr = transform.ensure_rgb_float(rgb_arr)
    xyz_arr = _nphusl._rgb_to_xyz(rgb_arr)
    for xyz, rgb, rgbf in zip(xyz_arr, rgb_arr, float_arr):
        diff = xyz - husl.rgb_to_xyz(rgbf)
        assert _diff(xyz, husl.rgb_to_xyz(rgbf))


@try_optimizations(Opt.numexpr)
def test_rgb_to_xyz_3d():
    img = _img()
    float_img = transform.ensure_rgb_float(img)
    xyz_arr = _nphusl._rgb_to_xyz(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            assert _diff(xyz_arr[row, col],
                         husl.rgb_to_xyz(float_img[row, col]))


@try_optimizations(Opt.numexpr)
def test_to_linear():
    a = 0.055 + 0.330
    b = 0.055 - 0.020
    c = 0.0
    d = 1.0
    for val in (v * 255 for v in (a, b, c, d)):
        assert husl.to_linear(val) == _nphusl._to_linear(np.array([val]))[0]


@try_optimizations(Opt.numexpr)
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
    new_dot = _nphusl._dot_product(m_inv, test_array)
    flat_input = test_array.reshape((test_array.size // 3, 3))
    flat_output = new_dot.reshape((new_dot.size // 3, 3))
    for new_dot, rgb in zip(flat_output, flat_input):
        old_dot = list(map(lambda row: husl.dot_product(row, rgb), m_inv))
        assert np.all(new_dot == old_dot)


@try_optimizations(Opt.numexpr)
def test_to_light():
    val_a = husl.epsilon + 0.4
    val_b = husl.epsilon - 0.003
    assert husl.f(val_a) == _nphusl._to_light(np.array([val_a]))[0]
    assert husl.f(val_b) == _nphusl._to_light(np.array([val_b]))[0]


### Tests for conversion in HUSL -> RGB direction


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_rgb_3d():
    img = _img()
    husl = _nphusl._rgb_to_husl(img)
    rgb = nphusl.to_rgb(husl)
    assert np.all(rgb == img)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_rgb_2d():
    img = np.ascontiguousarray(_img()[:, 17])
    husl = nphusl.to_husl(img)
    rgb = nphusl.to_rgb(husl)
    assert _diff(rgb, img, diff=1)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_husl_to_rgb():
    img = np.ascontiguousarray(_img()[25:, :5])
    husl = _nphusl._rgb_to_husl(img)
    rgb = transform.ensure_rgb_int(_nphusl._husl_to_rgb(husl))
    assert _diff(img, rgb)


def test_lch_to_rgb():
    img = _img()
    float_img = transform.ensure_rgb_float(img)
    lch = _nphusl._rgb_to_lch(float_img)
    rgb = _nphusl._lch_to_rgb(lch)
    int_rgb = transform.ensure_rgb_int(rgb)
    assert _diff(int_rgb, img, diff=1)


def test_xyz_to_rgb():
    img = ensure_float(_img())
    xyz = _nphusl._rgb_to_xyz(img)
    rgb = _nphusl._xyz_to_rgb(xyz)
    assert _diff(img, rgb)


def test_from_linear():
    a = 0.003 + 0.330
    b = 0.003 - 0.0020
    from nphusl.nphusl import _from_linear
    for val in (a, b):
        assert husl.from_linear(val) == _from_linear(np.array([val]))[0]


@try_optimizations(Opt.numexpr)
def test_husl_to_lch():
    img = _img()
    float_img = transform.ensure_rgb_float(img)
    lch = _nphusl._rgb_to_lch(float_img)
    husl = nphusl.to_husl(img)
    lch_2 = _nphusl._husl_to_lch(husl)
    img_2 = _nphusl._lch_to_rgb(lch_2)
    img_2 = transform.ensure_rgb_int(img_2)
    assert _diff(img_2, img, diff=1)


@try_optimizations(Opt.numexpr)
def test_luv_to_xyz():
    img = _img()
    xyz = _nphusl._rgb_to_xyz(img)
    luv = _nphusl._xyz_to_luv(xyz)
    xyz_2 = _nphusl._luv_to_xyz(luv)
    assert _diff(xyz_2, xyz)


@try_optimizations(Opt.numexpr)
def test_lch_to_luv():
    img = _img()
    lch = _nphusl._rgb_to_lch(img)
    luv = _nphusl._lch_to_luv(lch)  # we're testing this
    xyz = _nphusl._rgb_to_xyz(img)
    luv_2 = _nphusl._xyz_to_luv(xyz)
    lch_2 = _nphusl._luv_to_lch(luv_2)
    assert _diff(lch_2, lch)  # just a sanity check on RGB -> LCH
    assert _diff(luv, luv_2)


@try_optimizations(Opt.numexpr)
def test_from_light():
    val_a = 8 + 1.5
    val_b = 8 - 3.5
    _f_inv = _nphusl._from_light
    assert husl.f_inv(val_a) == _f_inv(np.array([val_a]))[0]
    assert husl.f_inv(val_b) == _f_inv(np.array([val_b]))[0]


### Tests for convenience functions, utilities


def test_channel():
    a = np.zeros((40, 40, 3))
    a[:] = (20, 30, 40)  # r = 20, b = 30, g = 40
    assert np.all(_channel(a, 0) == 20)
    assert np.all(_channel(a, 1) == 30)
    assert np.all(_channel(a, 2) == 40)


def test_channel_assignment():
    a = np.zeros((40, 40, 3))
    a[:] = (20, 30, 40)  # r = 20, b = 30, g = 40
    _channel(a, 0)[:] = 1
    _channel(a, 1)[:] = 2
    _channel(a, 2)[:] = 3
    assert np.all(_channel(a, 0) == 1)
    assert np.all(_channel(a, 1) == 2)
    assert np.all(_channel(a, 2) == 3)


def test_chunk():
    """Ensures that the chunk iterator breaks an image into NxN squares"""
    img = _img().copy()
    assert not np.all(img[:10, :10] == 0)
    for i, (chunk, _) in enumerate(transform.chunk_img(img, 10)):
        chunk[:] = i
    for i, (chunk, _) in enumerate(transform.chunk_img(img, 10)):
        assert np.all(chunk == i)
    assert np.all(img[:10, :10] == 0)


def test_chunk_transform():
    img = _img()
    assert np.sum(img == 0) != 0  # assert that there are some 0,0,0 pixels
    zero_at = np.where(img == 0)

    def trans_fn(chunk):
        chunk[chunk == 0] = 100
        return chunk

    chunks = transform.chunk_img(img, 10)
    transform.chunk_apply(trans_fn, chunks, out=img)
    assert np.sum(img == 0) == 0
    assert np.all(img[zero_at] == 100)


def test_transform_rgb():
    img = _img()
    as_husl = nphusl.to_husl(img)
    chunk_husl = transform.in_chunks(img, nphusl.to_husl, 10)
    assert _diff(as_husl, chunk_husl)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_hue_2d():
    img = _img()[:, 14]  # 2D RGB
    as_husl = nphusl.to_husl(img)
    just_hue = nphusl.to_hue(img)
    assert _diff(as_husl[..., 0], just_hue)


@try_optimizations(Opt.cython, Opt.numexpr)
def test_to_hue_3d():
    img = _img()  # 3D
    as_husl = _nphusl._rgb_to_husl(img / 255.0)
    just_hue = nphusl.to_hue(img)
    assert _diff(as_husl[..., 0], just_hue)


def test_handle_rgba():
    rgb = _img()
    rgba = np.zeros(shape=rgb.shape[:-1] + (4,), dtype=rgb.dtype)
    rgba[..., :3] = rgb
    alpha = 0x80  # 50%
    rgba[..., 3] = alpha
    ratio = alpha / 255.0
    do_nothing = lambda img: img
    to_rgb = transform.handle_rgba(do_nothing)
    new_rgb = to_rgb(rgba)
    should_be = np.round(rgb * ratio).astype(np.uint8)
    assert _diff(new_rgb, should_be)


@try_optimizations()
def test_to_husl_rgba():
    rgb = _img()
    rgba = np.zeros(shape=rgb.shape[:-1] + (4,), dtype=rgb.dtype)
    rgba[..., :3] = rgb
    alpha = 0x80  # 50%
    rgba[..., 3] = alpha
    ratio = alpha / 255.0
    new_rgb = np.round(rgb * ratio).astype(np.uint8)
    hsl_from_rgba = nphusl.to_husl(rgba)
    hsl_from_rgb = nphusl.to_husl(new_rgb)
    assert _diff_husl(hsl_from_rgba, hsl_from_rgb)


def test_cython_max_chroma():
    from nphusl import _cython_opt
    husl_chroma = husl.max_chroma_for_LH(0.25, 40.0)
    cyth_chroma = _cython_opt._test_max_chroma(0.25, 40.0)
    assert abs(husl_chroma - cyth_chroma) < 0.001


def test_ensure_float_input():
    @transform.float_input
    def go(inp, expect_dtype):
        assert inp.dtype == expect_dtype
    go(np.arange(10, dtype=np.float32), np.float32)
    go(np.arange(10, dtype=np.float64), np.float64)
    go(np.arange(10, dtype=np.uint8), np.float64)


def test_ensure_int_input():
    @transform.int_input
    def go(inp, expect_dtype):
        assert inp.dtype == expect_dtype
    go(np.arange(10, dtype=np.int64), np.int64)
    go(np.arange(10, dtype=np.uint8), np.uint8)
    go(np.arange(10, dtype=np.float32), np.int64)


def test_ensure_rgb_float_input():
    @transform.rgb_float_input
    def go(inp):
        assert inp.dtype == np.float64
        return inp
    # case 1: base type doesn't match (decorator scales down input arr)
    inp = go(np.arange(10, dtype=np.int64))
    assert np.all(inp == np.arange(10) / 255)
    # case 2: base type matches (decorator converts type, but doesn't scale)
    inp = go(np.arange(10, dtype=np.float32))
    assert np.all(inp == np.arange(10))
    # case 3: base type matches, no scaling or conversion happens
    orig = np.arange(10, dtype=np.float64)
    inp = go(orig)
    assert inp is orig
    assert np.all(inp == np.arange(10))


def test_ensure_rgb_int_input():
    @transform.rgb_int_input
    def go(inp):
        assert inp.dtype == np.uint8
        return inp
    # case 1: base type doesn't match (decorator scales up input arr)
    inp = go(np.asarray([0.1, 0.15, 0.66, 0.33], dtype=np.float64))
    assert np.all(inp == [26, 38, 168, 84])  # round to 0-255 scale
    # case 2: base type matches (decorator converts type, but doesn't scale)
    inp = go(np.arange(10, dtype=np.int32))
    assert np.all(inp == np.arange(10))
    # case 3: base type matches, no scaling or conversion happens
    orig = np.arange(0, 255, dtype=np.uint8)
    inp = go(orig)
    assert inp is orig
    assert np.all(inp == np.arange(0, 255))


def test_ensure_squeezed_output():
    @transform.squeeze_output
    def go():
        return np.asarray([[1,2,3]])
    assert list(go()) == [1,2,3]
    assert go().ndim == 1
    assert go().shape == (3,)


def test_ensure_image_input():
    @transform.ensure_image_input
    def go(inp):
        return inp
    # case 1: input is an RGB triplet, but it's not a numpy array
    assert type(go([1,2,3])) == np.ndarray
    # case 2: input is 1D, but size isn't a multiple of 3
    with pytest.raises(ValueError):
        go(list(range(10)))
    # case 3: input is 1D and size is a multiple of 3
    assert type(go(list(range(9)))) == np.ndarray
    # case 4: input is a numpy array, but its size isn't a multiple of 3
    with pytest.raises(ValueError):
        go(np.arange(10))
    # case 5: input is a numpy array, but it's 1D
    assert go(np.arange(9)).shape == (3, 3)
    # case 6: input as a numpy array, and it's already well formed
    orig = np.arange(9).reshape((3, 3))
    assert go(orig) is orig
    assert go(orig).shape == (3, 3)
    # case 7: input looks like an RGBA quadruplet
    assert go([1,2,3,4]).shape == (1, 4)
    assert type(go([1,2,3,4])) == np.ndarray
    assert go(list(range(8))).shape == (2, 4)


def _ref_to_husl(rgb):
    asfloat = (c/255.0 for c in rgb)
    return husl.rgb_to_husl(*asfloat)


def _ref_to_lch(rgb):
    asfloat = (c/255.0 for c in rgb)
    return husl.rgb_to_lch(*asfloat)


def _diff(a, b, diff=0.20):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.all(np.abs(a - b) <= diff)


def _diff_husl(a, b, max_rgb_diff=1):
    """Checks HUSL conversion by converting HUSL to RGB.
    Both HUSL triplets/arrays should produce the same RGB
    triplet/array."""
    rgb_a = husl.husl_to_rgb(*a) if len(a) == 3 else nphusl.to_rgb(a)
    rgb_b = husl.husl_to_rgb(*b) if len(b) == 3 else nphusl.to_rgb(b)
    return _diff(rgb_a, rgb_b, diff=max_rgb_diff)


IMG_CACHED = [None]


def _img():
    if IMG_CACHED[0] is None:
        i = imread.imread("images/gelface.jpg")
        i[50:100, 50:100] = np.random.rand(50, 50, 3) * 255
        i = i[::4, ::4]
        i[0] = 0  # ensure we get all black
        i[1] = 255  # ensure we get all white
        IMG_CACHED[0] = np.ascontiguousarray(i)
    IMG_CACHED[0][10:20, 10:20] = (
        np.random.rand(10, 10, 3) * 255).astype(np.uint8) / 255.0
    return IMG_CACHED[0].copy().astype(np.uint8)


# For functions with @try_optimizations decorator, remove
# the original function. Remaining functions will have names like
# fn__with_simd and fn__with_numpy.
for name in _optimized:
    del globals()[name]


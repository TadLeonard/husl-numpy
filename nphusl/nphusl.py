import math
import numpy as np
from numpy import ndarray
import warnings
from . import constants


### Optimization hooks

try:
    from . import _nphusl_expr as expr
except ImportError:
    expr = None
try:
    from . import _nphusl_cython as cyth
except ImportError:
    cyth = None


_NUMEXPR_ENABLED = True
_CYTHON_ENABLED = True
STANDARD = {}  # normal cpython/numpy fns
NUMEXPR = {}  # numexpr fns
CYTHON = {}  # cython extension fns


def optimized(fn):
    STANDARD[fn.__name__] = fn
    expr_fn = getattr(expr, fn.__name__, None) if _NUMEXPR_ENABLED else None
    cython_fn = getattr(cyth, fn.__name__, None) if _CYTHON_ENABLED else None
    opt_fn = cython_fn or expr_fn  # prefer cython
    result_fn = opt_fn or fn
    if cython_fn:
        CYTHON[fn.__name__] = cython_fn
    if expr_fn:
        NUMEXPR[fn.__name__] = expr_fn
    return result_fn


### Conversions in the direction of RGB -> HUSL


L_MAX = 99.99  # max lightness from original husl.py
L_MIN =  0.01


@optimized
def rgb_to_husl(rgb_nd: ndarray) -> ndarray:
    """Convert a float (0 <= i <= 1.0) RGB image to an `ndarray`
    of HUSL values"""
    return lch_to_husl(rgb_to_lch(rgb_nd))


@optimized
def rgb_to_hue(rgb: ndarray) -> ndarray:
    """Convenience function to return JUST the HUSL hue values
    for a given RGB image"""
    lch = luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))
    return _channel(lch, 2)


@optimized
def lch_to_husl(lch_nd: ndarray) -> ndarray:
    flat_shape = (lch_nd.size // 3, 3)
    lch_flat = lch_nd.reshape(flat_shape)
    _L, C, _H = (_channel(lch_flat, n) for n in range(3))
    hsl_flat = np.zeros(flat_shape, dtype=np.float)
    H, S, L = (_channel(hsl_flat, n) for n in range(3))
    H[:] = _H
    L[:] = _L

    # handle lightness extremes
    light = _L > L_MAX
    dark = _L < L_MIN
    S[light] = 0.0
    L[light] = 100.0
    S[dark] = 0.0
    L[dark] = 0.0

    # compute saturation for pixels that aren't too light or dark
    remaining = ~np.logical_or(light, dark)
    mx = _max_lh_chroma(lch_flat[remaining])
    S[remaining] = (C[remaining] / mx) * 100.0

    return hsl_flat.reshape(lch_nd.shape)


_2pi = math.pi * 2


@optimized
def _max_lh_chroma(lch: ndarray) -> ndarray:
    L, H = (_channel(lch, n) for n in (0, 2))
    hrad = (H / 360.0) * _2pi
    lengths = np.ndarray((lch.shape[0],), dtype=np.float)
    lengths[:] = np.inf
    for line in _bounds(L):
        lens = _ray_length(hrad, line)
        lens[np.isnan(lens)] = np.inf
        lens[lens < 0] = np.inf
        np.minimum(lens, lengths, out=lengths)
    return lengths


M_CONSTS = np.asarray(constants.M)
M1, M2, M3 = (M_CONSTS[..., n] for n in range(3))
TOP1_SCALAR = 284517.0 * M1 - 94839.0 * M3
TOP2_SCALAR = 838422.0 * M3 + 769860.0 * M2 + 731718.0 * M1
TOP2_L_SCALAR = 769860.0
BOTTOM_SCALAR = (632260.0 * M3 - 126452.0 * M2)
BOTTOM_CONST = 126452.0


@optimized
def _bounds(l_nd: ndarray) -> iter:
    sub1 = l_nd + 16.0
    np.power(sub1, 3, out=sub1)
    np.divide(sub1, 1560896.0, out=sub1)
    sub2 = sub1.flatten()  # flat copy
    lt_epsilon = sub2 < constants.EPSILON
    sub2[lt_epsilon] = (l_nd.flat[lt_epsilon] / constants.KAPPA)
    del lt_epsilon  # free NxM X sizeof(bool) memory?
    sub2 = sub2.reshape(sub1.shape)

    # The goal here is to compute "lines" for each lightness value
    # Since we can be dealing with LOTS of lightness values (i.e. 4,000 x
    # 6,000), this is implemented as an iterator. Raspberry Pi and other small
    # machines can't keep too many huge arrays in memory.
    for t1, t2, b in zip(TOP1_SCALAR, TOP2_SCALAR, BOTTOM_SCALAR):
        for t in (0, 1):
            top1 = sub2 * t1
            top2 = l_nd * sub2 * t2
            if t:
                top2 -= (l_nd * TOP2_L_SCALAR)
            bottom = sub2 * b
            if t:
                bottom += BOTTOM_CONST
            b1, b2 = top1 / bottom, top2 / bottom
            yield b1, b2


@optimized
def _ray_length(theta: ndarray, line: list) -> ndarray:
    m1, b1 = line
    length = b1 / (np.sin(theta) - m1 * np.cos(theta))
    return length


def rgb_to_lch(rgb: ndarray) -> ndarray:
    return luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))


@optimized
def luv_to_lch(luv_nd: ndarray) -> ndarray:
    uv_nd = _channel(luv_nd, slice(1, 2))
    uv_nd[uv_nd == -0.0] = 0.0   # -0.0 screws up atan2
    lch_nd = luv_nd.copy()
    U, V = (_channel(luv_nd, n) for n in range(1, 3))
    C, H = (_channel(lch_nd, n) for n in range(1, 3))
    C[:] = (U ** 2 + V ** 2) ** 0.5
    hrad = np.arctan2(V, U)
    H[:] = np.degrees(hrad)
    H[H < 0.0] += 360.0
    return lch_nd


@optimized
def xyz_to_luv(xyz_nd: ndarray) -> ndarray:
    flat_shape = (xyz_nd.size // 3, 3)
    luv_flat = np.zeros(flat_shape, dtype=np.float)  # flattened luv n-dim array
    xyz_flat = xyz_nd.reshape(flat_shape)
    X, Y, Z = (_channel(xyz_flat, n) for n in range(3))

    with np.errstate(invalid="ignore"):  # ignore divide by zero
        U_var = (4 * X) / (X + (15 * Y) + (3 * Z))
        V_var = (9 * Y) / (X + (15 * Y) + (3 * Z))
    U_var[np.isinf(U_var)] = 0  # correct divide by zero
    V_var[np.isinf(V_var)] = 0  # correct divide by zero

    L, U, V = (_channel(luv_flat, n) for n in range(3))
    L[:] = _f(Y)
    luv_flat[L == 0] = 0
    U[:] = L * 13 * (U_var - constants.REF_U)
    V[:] = L * 13 * (V_var - constants.REF_V)
    luv_flat = np.nan_to_num(luv_flat)
    return luv_flat.reshape(xyz_nd.shape)


def rgb_to_xyz(rgb_nd: ndarray) -> ndarray:
    rgbl = _to_linear(rgb_nd)
    return _dot_product(constants.M_INV, rgbl)


@optimized
def _f(y_nd: ndarray) -> ndarray:
    y_flat = y_nd.flatten()
    f_flat = np.zeros(y_flat.shape, dtype=np.float)
    gt = y_flat > constants.EPSILON
    f_flat[gt] = (y_flat[gt] / constants.REF_Y) ** (1.0 / 3.0) * 116 - 16
    f_flat[~gt] = (y_flat[~gt] / constants.REF_Y) * constants.KAPPA
    return f_flat.reshape(y_nd.shape)


@optimized
def _to_linear(rgb_nd: ndarray) -> ndarray:
    a = 0.055  # mysterious constant used in husl.to_linear
    xyz_nd = np.zeros(rgb_nd.shape, dtype=np.float)
    gt = rgb_nd > 0.04045
    xyz_nd[gt] = ((rgb_nd[gt] + a) / (1 + a)) ** 2.4
    xyz_nd[~gt] = rgb_nd[~gt] / 12.92
    return xyz_nd


@optimized
def _dot_product(scalars: list, rgb_nd: ndarray) -> ndarray:
    scalars = np.asarray(scalars, dtype=np.float)
    sum_axis = len(rgb_nd.shape) - 1
    x = np.sum(scalars[0] * rgb_nd, sum_axis)
    y = np.sum(scalars[1] * rgb_nd, sum_axis)
    z = np.sum(scalars[2] * rgb_nd, sum_axis)
    return np.dstack((x, y, z)).squeeze()


def _channel(data: ndarray, last_dim_idx) -> ndarray:
    return data[..., last_dim_idx]


### Conversions in the direction of HUSL -> RGB


@optimized
def husl_to_rgb(husl_nd: ndarray) -> ndarray:
    return lch_to_rgb(husl_to_lch(husl_nd))


def lch_to_rgb(lch_nd: ndarray) -> ndarray:
    return xyz_to_rgb(luv_to_xyz(lch_to_luv(lch_nd)))


def xyz_to_rgb(xyz_nd: ndarray) -> ndarray:
    xyz_dot = _dot_product(constants.M, xyz_nd)
    return _from_linear(xyz_dot)


def lch_to_luv(lch_nd: ndarray) -> ndarray:
    luv_nd = np.zeros(lch_nd.shape, dtype=np.float)
    _L, C, H = (_channel(lch_nd, n) for n in range(3))
    L, U, V  = (_channel(luv_nd, n) for n in range(3))
    hrad = np.radians(H)
    U[:] = np.cos(hrad) * C
    V[:] = np.sin(hrad) * C
    L[:] = _L
    return luv_nd


def _from_linear(xyz_nd: ndarray) -> ndarray:
    rgb_nd = np.zeros(xyz_nd.shape, dtype=np.float)
    lt = xyz_nd <= 0.0031308
    rgb_nd[lt] = 12.92 * xyz_nd[lt]
    rgb_nd[~lt] = 1.055 * (xyz_nd[~lt] ** (1 / 2.4)) - 0.055
    return rgb_nd


def husl_to_lch(husl_nd: ndarray) -> ndarray:
    flat_shape = (husl_nd.size // 3, 3)
    lch_flat = np.zeros(flat_shape, dtype=np.float)
    husl_flat = husl_nd.reshape(flat_shape)
    _H, S, _L = (_channel(husl_flat, n) for n in range(3))
    L, C, H = (_channel(lch_flat, n) for n in range(3))
    L[:] = _L
    H[:] = _H

    # compute max chroma for lightness and hue
    mx = _max_lh_chroma(lch_flat)
    C[:] = mx / 100.0 * S

    # handle lightness extremes
    light= L > L_MAX
    dark = L < L_MIN
    L[light] = 100
    C[light] = 0
    L[dark] = 0
    C[dark] = 0
    return lch_flat.reshape(husl_nd.shape)


def luv_to_xyz(luv_nd: ndarray) -> ndarray:
    flat_shape = (luv_nd.size // 3, 3)
    xyz_flat = np.zeros(flat_shape, dtype=np.float)  # flattened xyz array
    luv_flat = luv_nd.reshape(flat_shape)
    L, U, V = (_channel(luv_flat, n) for n in range(3))
    X, Y, Z = (_channel(xyz_flat, n) for n in range(3))

    Y_var = _f_inv(L)
    L13 = 13.0 * L
    with np.errstate(divide="ignore", invalid="ignore"):  # ignore divide by zero
        U_var = U / L13 + constants.REF_U
        V_var = V / L13 + constants.REF_V
    U_var[np.isinf(U_var)] = 0  # correct divide by zero
    V_var[np.isinf(V_var)] = 0  # correct divide by zero

    Y[:] = Y_var * constants.REF_Y
    with np.errstate(invalid="ignore"):
        X[:] = -(9 * Y * U_var) / ((U_var - 4.0) * V_var - U_var * V_var)
        Z[:] = (9.0 * Y - (15.0 * V_var * Y) - (V_var * X)) / (3.0 * V_var)
    xyz_flat[L == 0] = 0
    xyz_flat = np.nan_to_num(xyz_flat)
    return xyz_flat.reshape(luv_nd.shape)


def _f_inv(l_nd: ndarray) -> ndarray:
    l_flat = l_nd.flatten()
    large = l_nd > 8
    small = ~large
    out = np.zeros(l_flat.shape, dtype=np.float)
    out[large] = constants.REF_Y * (((l_nd[large] + 16) / 116) ** 3.0)
    out[small] = constants.REF_Y * l_nd[small] / constants.KAPPA
    return out.reshape(l_nd.shape)


### convenience functions


def handle_grayscale(fn):
    """Decorator for handling 1-channel RGB (grayscale) images"""
    def wrapped(rgb: ndarray, *args, **kwargs):
        if len(rgb.shape) == 3 and rgb.shape[-1] == 1:
            rgb = np.squeeze(rgb)
        if len(rgb.shape) == 2:
            _rgb = np.ndarray(rgb.shape + (3,), dtype=rgb.dtype)
            _rgb[:] = rgb[..., None]
            rgb = _rgb
        return fn(rgb, *args, **kwargs)
    return wrapped


def handle_rgba(fn):
    """Decorator for handling 4-channel RGBA images"""
    def wrapped(rgb: ndarray, *args, **kwargs):
        if len(rgb.shape) == 3 and rgb.shape[-1] == 4:
            # assume background is white because I said so
            warnings.warn("Assuming white RGBA background!")
            _rgb = rgb[..., :3]
            _a = rgb[..., 3]
            r, g, b, a = (rgb[..., n] for n in range(4))
            _rgb[:] = (_a[..., None] / 255.0) * _rgb
            rgb = np.round(_rgb).astype(np.uint8)
        return fn(rgb, *args, **kwargs)
    return wrapped


@handle_rgba
@handle_grayscale
def to_hue(rgb_img: ndarray, chunksize: int = None) -> ndarray:
    """Convert an RGB image of integers to a 2D array of HUSL hues"""
    out = np.zeros(rgb_img.shape[:2], dtype=np.float)
    out = transform_rgb(rgb_img, rgb_to_hue, chunksize, out)
    return out


def to_rgb(husl_img: ndarray, chunksize: int = None) -> ndarray:
    """Convert a 3D HUSL array of floats to a 3D RGB array of integers"""
    out = np.zeros(husl_img.shape, dtype=np.uint8)
    chunks = chunk_img(husl_img, chunksize)

    def transform(chunk):
        float_rgb = husl_to_rgb(chunk)
        return np.round(float_rgb * 255)  # to be cast to uint8

    chunk_transform(transform, chunks, out)
    return out


@handle_rgba
@handle_grayscale
def to_husl(rgb_img: ndarray, chunksize: int = None) -> ndarray:
    """Convert an RGB image of integers to a 3D array of HSL values"""
    out = np.zeros(rgb_img.shape, dtype=np.float)
    out = transform_rgb(rgb_img, rgb_to_husl, chunksize, out)
    return out


@handle_rgba
@handle_grayscale
def transform_rgb(rgb_img: ndarray, transform,
                  chunksize: int = None, out: ndarray = None) -> ndarray:
    """Transform an `np.ndarray` of RGB ints to some other
    float represntation (i.e. HUSL)"""
    chunks = chunk_img(rgb_img, chunksize)
    if out is None:
        out = np.zeros(rgb_img.shape, dtype=np.float)

    def trans(chunk: ndarray) -> ndarray:
        return transform(chunk / 255.0)

    chunk_transform(trans, chunks, out)
    return out


def chunk_transform(transform, chunks,
                    out: ndarray) -> None:
    """Transform chunks of an image and write the result to `out`"""
    for chunk, dims in chunks:
        (rstart, rend), (cstart, cend) = dims
        out[rstart: rend, cstart: cend] = transform(chunk)


def chunk_img(img: ndarray, chunksize: int = None):
    rows, cols = img.shape[:2]
    if chunksize:
        for row_start, row_end in chunk(rows, chunksize):
            for col_start, col_end in chunk(cols, chunksize):
                img_slice = img[row_start: row_end, col_start: col_end]
                yield img_slice, ((row_start, row_end), (col_start, col_end))
    else:
        yield img, ((0, rows), (0, cols))


def _chunk_no_idx(img: ndarray, chunksize: int = None):
    yield from (c[0] for c in chunk_img(img, chunksize))


def chunk_many(*arrays, chunksize: int = None):
    chunk_gens = [_chunk_no_idx(a, chunksize) for a in arrays]
    return zip(*chunk_gens)


def chunk(end: int, chunksize: int):
    """Generate tuples of (start_idx, end_idx) that breaks a sequence into
    slices of `chunksize` length"""
    _start = 0
    if end > chunksize:
        for _end in range(chunksize, end, chunksize or 1):
            yield _start, _end
            _start = _end
    yield _start, end


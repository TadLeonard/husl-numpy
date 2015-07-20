import math
import numpy as np  # type: ignore
from numpy import ndarray  # type: ignore
import husl
from typing import List, Iterator, Tuple, Callable, Union


# Constants used in the original husl.py for L channel comparison
L_MAX = 99.9999999
L_MIN =  0.0000001


def rgb_to_husl(rgb_nd: ndarray) -> ndarray:
    return lch_to_husl(rgb_to_lch(rgb_nd))


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

   
M_CONSTS = np.asarray(husl.m)
M1, M2, M3 = (M_CONSTS[..., n] for n in range(3))
TOP1_SCALAR = 284517.0 * M1 - 94839.0 * M3
TOP2_SCALAR = 838422.0 * M3 + 769860.0 * M2 + 731718.0 * M1
TOP2_L_SCALAR = 769860.0
BOTTOM_SCALAR = (632260.0 * M3 - 126452.0 * M2)
BOTTOM_CONST = 126452.0


def _bounds(l_nd: ndarray) -> Iterator:
    sub1 = l_nd + 16.0
    np.power(sub1, 3, out=sub1)
    np.divide(sub1, 1560896.0, out=sub1)
    sub2 = sub1.flatten()  # flat copy
    lt_epsilon = sub2 < husl.epsilon
    sub2[lt_epsilon] = (l_nd.flat[lt_epsilon] / husl.kappa)
    del lt_epsilon  # free NxM X sizeof(bool) memory?
    sub2 = sub2.reshape(sub1.shape)
    
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


def _ray_length(theta: ndarray, line: list) -> ndarray:
    m1, b1 = line
    length = b1 / (np.sin(theta) - m1 * np.cos(theta))
    return length 


def rgb_to_lch(rgb: ndarray) -> ndarray:
    return luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))


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


def xyz_to_luv(xyz_nd: ndarray) -> ndarray:
    flat_shape = (xyz_nd.size // 3, 3)
    luv_flat = np.zeros(flat_shape, dtype=np.float)  # flattened xyz n-dim array
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
    U[:] = L * 13 * (U_var - husl.refU)
    V[:] = L * 13 * (V_var - husl.refV)
    luv_flat = np.nan_to_num(luv_flat)
    return luv_flat.reshape(xyz_nd.shape)


def rgb_to_xyz(rgb_nd: ndarray) -> ndarray:
    rgbl = _to_linear(rgb_nd)
    return _dot_product(husl.m_inv, rgbl)


def _f(y_nd: ndarray) -> ndarray:
    y_flat = y_nd.flatten()
    f_flat = np.zeros(y_flat.shape, dtype=np.float)
    gt = y_flat > husl.epsilon
    f_flat[gt] = (y_flat[gt] / husl.refY) ** (1.0 / 3.0) * 116 - 16
    f_flat[~gt] = (y_flat[~gt] / husl.refY) * husl.kappa
    return f_flat.reshape(y_nd.shape)


def _to_linear(rgb_nd: ndarray) -> ndarray:
    a = 0.055  # mysterious constant used in husl.to_linear
    xyz_nd = np.zeros(rgb_nd.shape, dtype=np.float)
    gt = rgb_nd > 0.04045
    xyz_nd[gt] = ((rgb_nd[gt] + a) / (1 + a)) ** 2.4
    xyz_nd[~gt] = rgb_nd[~gt] / 12.92
    return xyz_nd
    

def _dot_product(scalars: List, rgb_nd: ndarray) -> ndarray:
    scalars = np.asarray(scalars, dtype=np.float)
    sum_axis = len(rgb_nd.shape) - 1
    x = np.sum(scalars[0] * rgb_nd, sum_axis)
    y = np.sum(scalars[1] * rgb_nd, sum_axis)
    z = np.sum(scalars[2] * rgb_nd, sum_axis)
    return np.dstack((x, y, z)).squeeze()


def _channel(data: ndarray, last_dim_idx: Union[int, slice]) -> ndarray:
    return data[..., last_dim_idx]


def transform_rgb(rgb_img: ndarray,
                  transform: Callable[[ndarray], ndarray],
                  chunksize: int = 200) -> ndarray:
    """Transform an `np.ndarray` of RGB ints to some other
    float represntation (i.e. HUSL)"""
    chunks = chunk_img(rgb_img, chunksize)
    out = np.zeros(rgb_img.shape, dtype=np.float)
    
    def trans(chunk: ndarray) -> ndarray:
        return transform(chunk / 255.0)
    
    chunk_transform(trans, chunks, out)
    return out


def chunk_transform(transform: Callable, chunks: Iterator,
                    out: ndarray) -> None:
    """Transform chunks of an image and write the result to `out`"""
    for chunk, dims in chunks:
        (rstart, rend), (cstart, cend) = dims
        out[rstart: rend, cstart: cend] = transform(chunk)


def chunk_img(img: ndarray, chunksize: int = 200) -> Iterator[ndarray]:
    """Break an image into squares of length `chunksize`"""
    rows, cols = img.shape[:2]
    for row_start, row_end in _chunk(rows, chunksize):
        for col_start, col_end in _chunk(cols, chunksize):
            img_slice = img[row_start: row_end, col_start: col_end]
            yield img_slice, ((row_start, row_end), (col_start, col_end))


def _chunk(end: int, chunksize: int) -> Iterator[Tuple[int, int]]:
    _start = 0
    if end > chunksize:
        for _end in range(chunksize, end, chunksize):
            yield _start, _end
            _start = _end
    yield _start, end 


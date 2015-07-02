import math
import numpy as np
from numpy import ndarray
import husl


# Constants used in the original husl.py for L channel comparison
L_MAX = 99.9999999
L_MIN =  0.0000001


def rgb_to_husl(rgb_nd: ndarray) -> ndarray:
    return lch_to_husl(rgb_to_lch(rgb_nd))


def lch_to_husl(lch_nd: ndarray) -> ndarray:
    _L, C, _H = (_channel(lch_nd, n) for n in range(3))
    hsl_nd = np.zeros(lch_nd.shape, dtype=float)
    H, S, L = (_channel(hsl_nd, n) for n in range(3))
    H[:] = _H
    L[:] = _L
    
    # handle lightness extremes
    hsl_nd[L > L_MAX][..., 1:] = (0.0, 100.0)
    hsl_nd[L < L_MIN][..., 1:] = 0.0
    
    # compute saturation
    mx = _max_lh_chroma(lch_nd)
    S = C / mx * 100.0
    _channel(hsl_nd, 1)[:] = S
    return hsl_nd


def _max_lh_chroma(lch: ndarray) -> ndarray:
    H_vals = _channel(lch, 2)
    hrad = H_vals / 360.0 * math.pi * 2.0
    lengths = np.ndarray((6,) + lch.shape[:-1])
    L = _channel(lch, 0)
    for i, line in enumerate(_bounds(L)):
        ray_len = _ray_length(hrad, line)
        lengths[i] = _ray_length(hrad, line)
    lengths[lengths < 0] = np.nan  # we don't want to consider subzero lens
    lengths[~np.isfinite(lengths)] = np.nan  # nor inifite lens
    return np.nanmin(lengths)


def _ray_length(theta: ndarray, line: list) -> ndarray:
    m1, b1 = line
    length = b1 / (np.sin(theta) - m1 * np.cos(theta))
    return length 


def _bounds(l_nd: ndarray) -> list:
    sub1 = ((l_nd + 16.0) ** 3.0) / 1560896.0
    sub2 = sub1.flatten()  # flat copy
    lt_epsilon = sub2 < husl.epsilon
    sub2[lt_epsilon] = (l_nd.flat[lt_epsilon] / husl.kappa)
    sub2 = sub2.reshape(sub1.shape)
    bounds = []
    for m1, m2, m3 in husl.m:
        for t in (0, 1):
            top1 = sub2 * (284517.0 * m1 - 94839.0 * m3)
            top2 = l_nd * sub2 * (838422.0 * m3 + 769860.0 * m2 + 731718.0 * m1)\
                   - ( l_nd * 769860.0 * t)
            bottom = sub2 * (632260.0 * m3 - 126452.0 * m2) + 126452.0 * t
            with np.errstate(invalid="ignore"):  # ignore zero division
                b1, b2 = top1 / bottom, top2 / bottom
            b1[~np.isfinite(b1)] = 0
            b2[~np.isfinite(b2)] = 0
            bounds.append((b1, b2))
    return bounds
        

def rgb_to_lch(rgb: ndarray) -> ndarray:
    return luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))


def luv_to_lch(luv_nd: ndarray) -> ndarray:
    lch_nd = luv_nd.copy()
    U = _channel(luv_nd, 1)
    V = _channel(luv_nd, 2)
    C_vals = _channel(lch_nd, 1)
    C_vals[:] = (U ** 2 + V ** 2) ** 0.5
    hrad = np.arctan2(V, U)
    H_vals = _channel(lch_nd, 2)
    H_vals[:] = np.degrees(hrad)
    H_vals[H_vals < 0.0] += 360
    return lch_nd


def xyz_to_luv(xyz_nd: ndarray) -> ndarray:
    flat_shape = (xyz_nd.size // 3, 3)
    luv_flat = np.zeros(flat_shape)  # flattened xyz n-dim array
    xyz_flat = xyz_nd.reshape(flat_shape)
    X, Y, Z = (_channel(xyz_flat, n) for n in range(3))

    with np.errstate(invalid="ignore"):  # ignore divide by zero
        U_var = (4 * X) / (X + (15 * Y) + (3 * Z))
        V_var = (9 * Y) / (X + (15 * Y) + (3 * Z))
    U_var[~np.isfinite(U_var)] = 0  # correct divide by zero
    V_var[~np.isfinite(V_var)] = 0  # correct divide by zero
    
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
    f_flat = np.zeros(y_flat.shape)
    gt = y_flat > husl.epsilon
    f_flat[gt] = (y_flat[gt] / husl.refY) ** (1.0 / 3.0) * 116 - 16
    f_flat[~gt] = (y_flat[~gt] / husl.refY) * husl.kappa
    return f_flat.reshape(y_nd.shape)


def _to_linear(rgb_nd: ndarray) -> ndarray:
    a = 0.055  # mysterious constant used in husl.to_linear
    xyz_nd = np.zeros(rgb_nd.shape)
    gt = rgb_nd > 0.04045
    xyz_nd[gt] = ((rgb_nd[gt] + a) / (1 + a)) ** 2.4
    xyz_nd[~gt] = rgb_nd[~gt] / 12.92
    return xyz_nd
    

def _dot_product(scalars, rgb_nd: ndarray) -> ndarray:
    scalars = np.asarray(scalars)
    assert rgb_nd.shape[-1] == 3
    assert scalars.shape == (3, 3)
    sum_axis = len(rgb_nd.shape) - 1
    x = np.sum(scalars[0] * rgb_nd, sum_axis)
    y = np.sum(scalars[1] * rgb_nd, sum_axis)
    z = np.sum(scalars[2] * rgb_nd, sum_axis)
    return np.dstack((x, y, z)).squeeze()


def _channel(data: ndarray, last_dim_idx: int) -> ndarray:
    return data[..., last_dim_idx]
 

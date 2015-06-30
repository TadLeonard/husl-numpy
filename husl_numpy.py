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
    husl_nd = lch_nd.copy()
    L_vals = _channel(lch_nd, 0)
    L_large = L_vals > L_MAX
    L_small = L_vals < L_MIN
    husl_L_large = husl_nd[L_large]
    _channel(husl_L_large, 0)[:] = _channel(husl_L_large, 2)
    _channel(husl_L_large, 2)[:] = 100.0
    husl_L_small = husl_nd[L_small]
    _channel(husl_L_small, 0)[:] = _channel(husl_L_small, 2)
    mx = _max_lh_chroma(lch_nd) 
    C_vals =  _channel(lch_nd, 1)
    S = C_vals / mx * 100.0
    _channel(husl_nd, 1)[:] = S
    return husl_nd


def _max_lh_chroma(lch: ndarray) -> ndarray:
    H_vals = _channel(lch, 2)
    hrad = H_vals / 360.0 * math.pi * 2.0
    lengths = np.ndarray((6,) + lch.shape[:-1])
    L_vals = _channel(lch, 0)
    for i, line in enumerate(_bounds(L_vals)):
        lengths[i] = _ray_length(hrad, line)
    return np.nanmin(lengths)


def _ray_length(theta: ndarray, line: list) -> ndarray:
    m1, b1 = line
    length = b1 / (np.sin(theta) - m1 * np.sin(theta))
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
            bounds.append((top1 / bottom, top2 / bottom))
    return bounds
        

def rgb_to_lch(rgb: ndarray) -> ndarray:
    return luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))


def luv_to_lch(luv_nd: ndarray) -> ndarray:
    lch_nd = luv_nd.copy()
    U_vals = _channel(luv_nd, 1)
    V_vals = _channel(luv_nd, 2)
    C_vals = _channel(lch_nd, 1)
    C_vals[:] = (U_vals ** 2 + V_vals ** 2) ** 0.5
    hrad = np.arctan2(V_vals, U_vals)
    H_vals = _channel(lch_nd, 2)
    H_vals[:] = np.degrees(hrad)
    H_vals[H_vals < 0.0] += 360
    return lch_nd


def xyz_to_luv(xyz_nd: ndarray) -> ndarray:
    luv_nd = xyz_nd.copy()
    X_vals = _channel(xyz_nd, 0)
    Y_vals = _channel(xyz_nd, 1)
    Z_vals = _channel(xyz_nd, 2)
    U_var = (4 * X_vals) / (X_vals + (15 * Y_vals) + (3 * Z_vals))
    V_var = (9 * Y_vals) / (X_vals + (15 * Y_vals) + (3 * Z_vals))
    L_vals = _channel(luv_nd, 0)
    L_vals[:] = f(Y_vals)
    luv_nd[L_vals == 0] = 0
    U_vals = _channel(luv_nd, 1)
    U_vals[:] = L_vals * 13 * (U_var - husl.refU)
    V_vals = _channel(luv_nd, 2)
    V_vals[:] = L_vals * 13 * (V_var - husl.refV)
    return luv_nd


def rgb_to_xyz(rgb_nd: ndarray) -> ndarray:
    a = 0.055  # mysterious constant used in husl.to_linear
    xyz_nd = np.zeros(rgb_nd.shape)
    gt = rgb_nd > 0.04045
    xyz_nd[gt] = ((rgb_nd[gt] + a) / (1 + a)) ** 2.4
    xyz_nd[~gt] = rgb_nd[~gt] / 12.92
    return xyz_nd


def f(y_nd: ndarray) -> ndarray:
    y_flat = y_nd.flatten()
    f_flat = np.zeros(y_flat.shape)
    gt = y_flat > husl.epsilon
    f_flat[gt] = (y_flat[gt] / husl.refY) ** (1.0 / 3.0) * 116 - 16
    f_flat[~gt] = (y_flat[~gt] / husl.refY) * husl.kappa
    return f_flat.reshape(y_nd.shape)
    

def _channel(data: ndarray, last_dim_idx: int) -> ndarray:
    return data[..., last_dim_idx]
    

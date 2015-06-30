import math
import numpy as np
from numpy import ndarray
import husl


# Constants used in the original husl.py for L channel comparison
L_MAX = 99.9999999
L_MIN =  0.0000001


def rgb_to_husl(rgb: ndarray) -> ndarray:
    return lch_to_husl(rgb_to_lch(rgb))


def lch_to_husl(lch_nd: ndarray) -> ndarray:
    husl_nd = lch_nd.copy()
    L_vals = _channel(lch_nd, 0)
    L_large = L_vals > L_MAX
    L_small = L_vals < L_MIN
    husl_L_large = husl_nd[L_large]
    _channel(husl_L_large, 0)[:] = _channel(husl_nd, 2)
    _channel(husl_L_large, 2)[:] = 100.0
    _channel(husl_L_small, 0)[:] = _channel(husl_nd, 2)
    mx = _max_lh_chroma(lch_nd) 
    C_vals =  _channel(lch_nd, 1)
    S = C_vals / mx * 100.0
    _channel(husl_nd, 1)[:] = S


def _max_lh_chroma(lch: ndarray) -> ndarray:
    H_vals = _channel(lch, 2)
    hrad = H_vals / 360.0 * math.pi * 2.0
    lengths = np.ndarray((6, lch.shape[0]))
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
    sub2 = sub1.copy()
    sub2[sub2 < husl.epsilon] = (l_nd / husl.kappa)
    bounds = []
    for m1, m2, m3 in husl.m:
        for t in (0, 1):
            top1 = sub2 * (284517.0 * m1 - 94839.0 * m3)
            top2 = L * sub2 * (838422.0 * m3 + 769860.0 * m2 + 731718.0 * m1)\
                   - ( L * 769860.0 * t)
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
    U_var = 4 * 


def f(t_nd: ndarray) -> ndarray:
    pass 


def _channel(data: ndarray, last_dim_idx: int) -> ndarray:
    return data[..., last_dim_idx]
    

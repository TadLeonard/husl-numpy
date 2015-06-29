import numpy as np
from numpy import ndarray
import husl


# Constants used in the original husl.py for L channel comparison
L_MAX = 99.9999999
L_MIN =  0.0000001


def rgb_to_husl(rgb: ndarray) -> ndarray:
    return lch_to_husl(rgb_to_lch(rgb))


def lch_to_husl(lch: ndarray) -> ndarray:
    nd_husl = np.zeros(lch.shape, dtype=float)
    L_vals = _channel(lch, 0)
    L_large = L_vals > L_MAX
    L_small = L_vals < L_MIN
    husl_L_large = nd_husl[L_large]
    _channel(husl_L_large, 0)[:] = _channel(nd_husl, 2)
    _channel(husl_L_large, 2)[:] = 100.0
    _channel(husl_L_small, 0)[:] = _channel(nd_husl, 2)


def _max_lh_chroma(lch: ndarray) -> ndarray:
    hrad = _channel(lch, 2) / 360.0 * math.pi * 2.0


def _get_bounds(l_nd: ndarray) -> ndarray:
    pass 
     

def rgb_to_lch(rgb: ndarray) -> ndarray:
    return luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))


def luv_to_lch(rgb: ndarray) -> ndarray:
    pass


def _channel(data: ndarray, last_dim_idx: int) -> ndarray:
    return data[..., last_dim_idx]
    

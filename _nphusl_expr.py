import numpy as np
from numpy import ndarray
import numexpr as ne
import husl


ne.set_num_threads(4)
#profile = lambda fn: fn


@profile
def _to_linear(rgb_nd: ndarray) -> ndarray:
    a = 0.055  # mysterious constant used in husl.to_linear
    xyz_nd = np.zeros(rgb_nd.shape, dtype=np.float)
    gt = rgb_nd > 0.04045
    lt = ~gt
    rgb_gt = rgb_nd[gt]
    rgb_lt = rgb_nd[lt]
    xyz_nd[gt] = ne.evaluate("((rgb_gt + a) / (1 + a)) ** 2.4")
    xyz_nd[lt] = rgb_lt / 12.92  # this gives bad values in numexpr!!
    return xyz_nd


@profile
def _f(y_nd: ndarray) -> ndarray:
    y_flat = y_nd.flatten()
    f_flat = np.zeros(y_flat.shape, dtype=np.float)
    gt = y_flat > husl.epsilon
    y_flat_gt = y_flat[gt]
    ref_y = hus.refY
    f_flat[gt] = (y_flat[gt] / husl.refY) ** (1.0 / 3.0) * 116 - 16
    f_flat[gt] = ne.evaluate("(y_flat_gt / ref_y) ** (1.0 / 3.0) * 116 - 16")
    f_flat[~gt] = (y_flat[~gt] / husl.refY) * husl.kappa
    return f_flat.reshape(y_nd.shape)




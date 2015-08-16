import numpy as np
from numpy import ndarray
import numexpr as ne
import husl


M_CONSTS = np.asarray(husl.m)
M1, M2, M3 = (M_CONSTS[..., n] for n in range(3))
TOP1_SCALAR = 284517.0 * M1 - 94839.0 * M3
TOP2_SCALAR = 838422.0 * M3 + 769860.0 * M2 + 731718.0 * M1
TOP2_L_SCALAR = 769860.0
BOTTOM_SCALAR = (632260.0 * M3 - 126452.0 * M2)
BOTTOM_CONST = 126452.0


#profile = lambda fn: fn


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


def _f(y_nd: ndarray) -> ndarray:
    y_flat = y_nd.flatten()
    f_flat = np.zeros(y_flat.shape, dtype=np.float)
    gt = y_flat > husl.epsilon
    lt = ~gt
    y_flat_gt = y_flat[gt]
    y_flat_lt = y_flat[lt]
    ref_y = husl.refY
    kappa = husl.kappa
    f_flat[gt] = ne.evaluate("(y_flat_gt / ref_y) ** (1. / 3.) * 116 - 16")
    f_flat[lt] = ne.evaluate("(y_flat_lt / ref_y) * kappa")
    return f_flat.reshape(y_nd.shape)


@profile
def _bounds(l_nd: ndarray) -> iter:
    sub1 = ne.evaluate("((l_nd + 16.0) ** 3) / 1560896.0")
    sub2 = sub1.flatten()  # flat copy
    lt_epsilon = sub2 < husl.epsilon
    sub2[lt_epsilon] = (l_nd.flat[lt_epsilon] / husl.kappa)
    del lt_epsilon  # free NxM X sizeof(bool) memory?
    sub2 = sub2.reshape(sub1.shape)
    
    # The goal here is to compute "lines" for each lightness value
    # Since we can be dealing with LOTS of lightness values (i.e. 4,000 x
    # 6,000), this is implemented as an iterator. Raspberry Pi and other small
    # machines can't keep too many huge arrays in memory.
    for t1, t2, b in zip(TOP1_SCALAR, TOP2_SCALAR, BOTTOM_SCALAR):
        bottom = sub2 * b
        top1 = sub2 * t1
        top2 = l_nd * sub2 * t2
        yield top1 / bottom, top2 / bottom
        top2 -= (l_nd * TOP2_L_SCALAR)
        bottom += BOTTOM_CONST
        yield top1 / bottom, top2 / bottom



import numpy as np
from numpy import ndarray
import numexpr as ne
from . import constants

# Constants used in the original husl.py for L channel comparison
L_MAX = 99.9999999
L_MIN =  0.0000001
M_CONSTS = np.asarray(constants.M)
M1, M2, M3 = (M_CONSTS[..., n] for n in range(3))
TOP1_SCALAR = 284517.0 * M1 - 94839.0 * M3
TOP2_SCALAR = 838422.0 * M3 + 769860.0 * M2 + 731718.0 * M1
TOP2_L_SCALAR = 769860.0
BOTTOM_SCALAR = (632260.0 * M3 - 126452.0 * M2)
BOTTOM_CONST = 126452.0


def lch_to_husl(lch_nd: ndarray) -> ndarray:
    flat_shape = (lch_nd.size // 3, 3)
    lch_flat = lch_nd.reshape(flat_shape)
    _L, C, _H = (lch_flat[..., n] for n in range(3))
    hsl_flat = np.zeros(flat_shape, dtype=np.float)
    H, S, L = (hsl_flat[..., n] for n in range(3))
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
    c_rem = C[remaining]
    S[remaining] = ne.evaluate("(c_rem / mx) * 100.0")

    return hsl_flat.reshape(lch_nd.shape)



def luv_to_lch(luv_nd: ndarray) -> ndarray:
    uv_nd = luv_nd[..., slice(1, 2)]
    uv_nd[uv_nd == -0.0] = 0.0   # -0.0 screws up atan2
    lch_nd = luv_nd.copy()
    U, V = luv_nd[..., 1], luv_nd[..., 2]
    C, H = lch_nd[..., 1], lch_nd[..., 2]
    C[:] = ne.evaluate("(U ** 2 + V ** 2) ** 0.5")
    H[:] = np.degrees(ne.evaluate("arctan2(V, U)"))
    H[H < 0.0] += 360.0
    return lch_nd


def xyz_to_luv(xyz_nd: ndarray) -> ndarray:
    flat_shape = (xyz_nd.size // 3, 3)
    luv_flat = np.zeros(flat_shape, dtype=np.float)  # flattened luv n-dim array
    xyz_flat = xyz_nd.reshape(flat_shape)
    X, Y, Z = (xyz_flat[..., n] for n in range(3))

    with np.errstate(invalid="ignore"):  # ignore divide by zero
        U_var = ne.evaluate("(4 * X) / (X + (15 * Y) + (3 * Z))")
        V_var = ne.evaluate("(9 * Y) / (X + (15 * Y) + (3 * Z))")
    U_var[np.isinf(U_var)] = 0  # correct divide by zero
    V_var[np.isinf(V_var)] = 0  # correct divide by zero

    L, U, V = (luv_flat[..., n] for n in range(3))
    L[:] = _f(Y)
    ref_u, ref_v = constants.REF_U, constants.REF_V
    U[:] = ne.evaluate("L * 13 * (U_var - ref_u)")
    V[:] = ne.evaluate("L * 13 * (V_var - ref_v)")
    luv_flat[np.isnan(luv_flat)] = 0
    return luv_flat.reshape(xyz_nd.shape)


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
    gt = y_flat > constants.EPSILON
    lt = ~gt
    y_flat_gt = y_flat[gt]
    y_flat_lt = y_flat[lt]
    ref_y = constants.REF_Y
    kappa = constants.KAPPA
    f_flat[gt] = ne.evaluate("(y_flat_gt / ref_y) ** (1. / 3.) * 116 - 16")
    f_flat[lt] = ne.evaluate("(y_flat_lt / ref_y) * kappa")
    return f_flat.reshape(y_nd.shape)


def _bounds(l_nd: ndarray) -> iter:
    sub1 = ne.evaluate("((l_nd + 16.0) ** 3) / 1560896.0")
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
        bottom = sub2 * b
        top1 = sub2 * t1
        top2 = ne.evaluate("l_nd * sub2 * t2")
        yield top1 / bottom, top2 / bottom
        bottom += BOTTOM_CONST
        yield top1 / bottom, ne.evaluate(
                "(top2 - (l_nd * TOP2_L_SCALAR)) / bottom")


def _ray_length(theta: ndarray, line: list) -> ndarray:
    m1, b1 = line
    length = ne.evaluate("b1 / (sin(theta) - m1 * cos(theta))")
    return length


_2pi = np.pi * 2


def _max_lh_chroma(lch: ndarray) -> ndarray:
    L, H = lch[..., 0], lch[..., 2]
    hrad = ne.evaluate("(H / 360.0) * _2pi")
    lengths = np.ndarray((lch.shape[0],), dtype=np.float)
    lengths.fill(np.inf)
    for line in _bounds(L):
        lens = _ray_length(hrad, line)
        lens[np.isnan(lens)] = np.inf
        lens[lens < 0] = np.inf
        np.minimum(lens, lengths, out=lengths)
    return lengths


def _dot_product(scalars: list, rgb_nd: ndarray) -> ndarray:
    scalars = np.asarray(scalars, dtype=np.float)
    xyz_nd = np.ndarray(rgb_nd.shape, dtype=np.float)
    x, y, z = (xyz_nd[..., n] for n in range(3))
    sum_axis = len(rgb_nd.shape) - 1
    _a, _b, _c = scalars

    # Why use this silly string formatting? Because numexpr
    # has some strange bug where, in "sum(blah, axis=somevar)"`somevar`,
    # the `somevar` local variable is seen as being "less than 0" or "False"
    # and it suggests using bitwise operators intead of "not".
    # So that nonsensical bevaior is avoided by giving a literal int in the
    # expression string.
    expr = "sum({{const}} * rgb_nd, axis={axis})".format(axis=sum_axis)
    x[:] = ne.evaluate(expr.format(const="_a"))
    y[:] = ne.evaluate(expr.format(const="_b"))
    z[:] = ne.evaluate(expr.format(const="_c"))
    return xyz_nd



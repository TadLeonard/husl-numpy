"""Convenience functions and decorators for handling the
shape and dtypes of image array inputs and outputs"""

import warnings
from functools import wraps, partial
from collections import namedtuple
from enum import Enum

import numpy as np
from numpy import ndarray


type_tuple = namedtuple("type_tuple", "base exact convert exact_required")


def to_rgb_int(dtype, arr: ndarray):
    if not np.issubdtype(arr.dtype, dtype.base):
        # scale float arrays in interval [0,1] to [0,255]
        arr = scale_up_rgb(arr)
    return to_dtype(dtype, arr)


def to_rgb_float(dtype, arr: ndarray):
    if not np.issubdtype(arr.dtype, dtype.base):
        # scale int arrays in interval [0,255] to [0,1]
        arr = scale_down_rgb(arr)
    return to_dtype(dtype, arr)


def scale_up_rgb(rgb: ndarray):
    """Convert RGB up to [0, 255] range"""
    return np.round(rgb*255.0)


def scale_down_rgb(rgb: ndarray):
    """Convert RGB down to [0, 1.0] range"""
    return rgb/255.0


def to_dtype(dtype, arr: ndarray):
    return arr.astype(dtype.exact)


_int_type = type_tuple(np.integer, np.uint8, to_dtype, False)
_float_type = type_tuple(np.float, np.float64, to_dtype, False)
_rgb_int_type = type_tuple(np.integer, np.uint8, to_rgb_int, True)
_rgb_float_type = type_tuple(np.float, np.float64, to_rgb_float, True)


class Dtype(type_tuple, Enum):
    int = _int_type
    float = _float_type
    rgb_int = _rgb_int_type
    rgb_float = _rgb_float_type


def ensure_dtype(dtype: Dtype, arr: ndarray):
    required_type = dtype.exact if dtype.exact_required else dtype.base
    if not np.issubdtype(arr.dtype, required_type):
        arr = dtype.convert(dtype, arr)
    return arr


ensure_int = partial(ensure_dtype, Dtype.int)
ensure_float = partial(ensure_dtype, Dtype.float)
ensure_rgb_int = partial(ensure_dtype, Dtype.rgb_int)
ensure_rgb_float = partial(ensure_dtype, Dtype.rgb_float)


def ensure_input_dtype(dtype: Dtype):
    """Decorator for ensuring that a function gets an array of `dtype`.
    If a float array is passed when an int is desired, the float array
    will be converted with something like `(input*255).astype(int)`."""
    def decorated(fn):
        @wraps(fn)
        def wrapped(arr: ndarray, *args, **kwargs):
            arr = ensure_dtype(dtype, arr)
            return fn(arr, *args, **kwargs)
        return wrapped
    return decorated


int_input = ensure_input_dtype(Dtype.int)
float_input = ensure_input_dtype(Dtype.float)
rgb_int_input = ensure_input_dtype(Dtype.rgb_int)
rgb_float_input = ensure_input_dtype(Dtype.rgb_float)


def ensure_output_dtype(dtype: Dtype):
    """Like `ensure_rgb_input_dtype`, but this decorator ensures
    that the *output* of a function has a specific dtype"""
    def decorated(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return ensure_dtype(dtype, out)
        return wrapped
    return decorated


int_output = ensure_output_dtype(Dtype.int)
float_output = ensure_output_dtype(Dtype.float)
rgb_int_output = ensure_output_dtype(Dtype.rgb_int)
rgb_float_output = ensure_output_dtype(Dtype.rgb_float)


def ensure_numpy_input(fn):
    """Ensures that we're working with an np.ndarray
    with at least two dimensions"""
    @wraps(fn)
    def wrapped(arr: ndarray, *args, **kwargs):
        if not isinstance(arr, (np.ndarray, np.generic)):
            arr = np.ascontiguousarray(arr)
        if arr.ndim == 1:
            size = arr.size
            if size in (3, 4):
                # force [r, g, b] to [[r, g, b]]
                # or [r, g, b, a] to [[r, g, b, a]]
                arr = arr[None, :]
            else:
                # force [r, g, b, r, g, b, ...] to [[r, g, b], ...]
                # assumes an array of triplets (no RGBA allowed)
                arr = arr.reshape((int(size/3), 3))
        return fn(arr, *args, **kwargs)
    return wrapped


def handle_grayscale(fn):
    """Decorator for handling 1-channel RGB (grayscale) images"""
    @wraps(fn)
    def wrapped(rgb: ndarray, *args, **kwargs):
        if rgb.shape[-1] == 1:
            rgb = np.squeeze(rgb)
        if len(rgb.shape) == 2 and rgb.shape[-1] != 3:
            # 1D grayscale needed squeezing
            _rgb = np.ndarray(rgb.shape + (3,), dtype=rgb.dtype)
            _rgb[:] = rgb[..., None]
            rgb = _rgb
        return fn(rgb, *args, **kwargs)
    return wrapped


def handle_rgba(fn):
    """Decorator for handling 4-channel RGBA images"""
    @wraps(fn)
    def wrapped(arr: ndarray, *args, **kwargs):
        if len(arr.shape) == 3 and arr.shape[-1] == 4:
            # assume background is white because I said so
            warnings.warn("Assuming white RGBA background!")
            rgb = arr[..., :3]
            a = arr[..., 3]
            ratio = (a / 255.0)
            rgb = np.round(rgb * ratio[..., None]).astype(np.uint8)
        else:
            rgb = arr
        return fn(rgb, *args, **kwargs)
    return wrapped


### Functions for applying transformations to images in chunks

def in_chunks(img: ndarray, transform: callable,
                  out: ndarray = None, chunksize: int = None) -> ndarray:
    """Transform an image with `transform`, optionally in chunks
    of `chunksize`, and optionally place results into `out` array."""
    if chunksize and out is None:
        raise ValueError("`out` array required if `chunksize` is given")
    if chunksize:
        chunks = chunk_img(img, chunksize)
        chunk_trans = chunk_apply_1d if len(out.shape) == 1 else \
                      chunk_apply
        chunk_trans(transform, chunks, out)
        return out
    else:
        return transform(img)


def chunk_apply(transform, chunks, out: ndarray) -> None:
    """Transform chunks of an image and write the result to `out`"""
    for chunk, dims in chunks:
        (rstart, rend), (cstart, cend) = dims
        out[rstart: rend, cstart: cend] = transform(chunk)


def chunk_apply_1d(transform, chunks, out: ndarray) -> None:
    for chunk, dims in chunks:
        (rstart, rend), _ = dims
        out[rstart: rend] = transform(chunk)


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


"""Convenience functions/decorators for handling the
shape and dtypes of image inputs and outputs"""

from functools import wraps


def ensure_input_dtype(check_for, replace_with=None):
    """Decorator for handling input array float/integer dtypes when
    one or the other is preferred. The `check_for` argument is
    the NumPy base class to check, the `replace_with` argument is the
    dtype to use should `np.issubtype(check_for` return False.
    `replace_with` defaults to `check_for`."""
    if replace_with is None:
        replace_with = check_for
    def decorated(fn):
        @wraps(fn)
        def wrapped(arr: ndarray, *args, **kwargs):
            if not np.issubdtype(arr.dtype, check_for):
                arr = arr.astype(replace_with)
            return fn(arr, *args, **kwargs)
        return wrapped
    return decorated


def ensure_output_dtype(check_for, replace_with=None):
    """Decorator for handling output array float/integer dtypes when
    one or the other is preferred. The `check_for` argument is
    the NumPy base class to check, the `replace_with` argument is the
    dtype to use should `np.issubtype(check_for` return False.
    `replace_with` defaults to `check_for`."""
    if replace_with is None:
        replace_with = check_for
    def decorated(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            out = fn(arr, *args, **kwargs)
            if not np.issubdtype(out, check_for):
                out = out.astype(replace_with)
            return out
        return wrapped
    return decorated


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


### Functions for applying transformations to images in chunks

def in_chunks(img: ndarray, transform: callable,
                  out: ndarray = None, chunksize: int = None) -> ndarray:
    """Transform an image with `transform`, optionally in chunks
    of `chunksize`, and optionally place results into `out` array."""
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


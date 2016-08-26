import sys
import timeit

from collections import defaultdict

import imageio
import pytest
import numpy as np
import nphusl
import husl


default_impls = "simd cython numexpr numpy".split()


# ASCII formatting codes
BOLD = "\033[1m"
FAINT = "\033[2m"
END = "\033[0m"


@pytest.fixture
def impls(request):
    methods = request.config.getoption("--impls")
    assert all(m in default_impls for m in methods)
    return methods


@pytest.fixture
def iters(request):
    return request.config.getoption("--iters")


class CachedImg:
    rgb = None
    hsl = None
    path = None


@pytest.fixture
def img(request):
    if CachedImg.rgb is None:
        path = request.config.getoption("--img")
        CachedImg.path = path
        CachedImg.rgb = imageio.imread(path)
        CachedImg.hsl = nphusl.to_husl(CachedImg.rgb)
    return CachedImg


def test_perf_husl_to_rgb(impls, iters, img):
    fn = "nphusl.to_rgb"
    _test_all(fn, img.hsl, locals(), impls, iters)


def test_perf_rgb_to_husl(impls, iters, img):
    fn = "nphusl.to_husl"
    _test_all(fn, img.rgb, locals(), impls, iters)


def test_perf_rgb_to_husl_one_triplet(impls, iters):
    fn = "nphusl.to_husl"
    rgb_triplet = list((np.random.rand(3) * 255).astype(np.uint8))
    _test_all(fn , rgb_triplet, locals(), impls, iters)


def test_perf_rgb_to_hue(impls, iters, img):
    fn = "nphusl.to_hue"
    _test_all(fn, img.rgb, locals(), impls, iters)


def _test_all(fn, img, env, impls, iters):
    env = {**globals(), **locals()}
    print("\n\n{}({})".format(fn, "img"), end="")
    if len(img) > 3:
        dims = "x".join(str(i) for i in img.shape)
    else:
        dims = "3x1"
    print(" -- best of {} with dims {} (source: {})".format(
          iters, dims, CachedImg.path))
    times = {}
    for impl in impls:
        enable = getattr(nphusl, "{}_enabled".format(impl))
        with enable():
            runs = timeit.repeat("{}(img)".format(fn),
                                 repeat=iters, number=1, globals=env)
        times[impl] = min(runs)
    worst = max(times.values())
    very_best = min(times.values())
    sorted_times = sorted(times.items(), key=lambda x: x[1])
    for i, (impl, best) in enumerate(sorted_times):
        spaces = len("numexpr") - len(impl)
        chart_bar = "\N{BLACK RIGHT-POINTING TRIANGLE}" * int(40*best/worst)
        tformat = "{:2.2e} s" if very_best < 0.001 else "{:2.4f} s"
        times_slower = best/very_best
        slower = "(fastest)" if times_slower == 1 else \
                 "({:0.2f}x slower)".format(times_slower)
        t = tformat.format(best)
        pps = np.asarray(img).size / 3 / best
        print((BOLD if i == 0 else FAINT) +
              "{}. {}: {}[{}] [{:0.2e} pixels/s] {:15s} {:20s}".format(
              i, impl, " "*spaces, t, pps, slower, chart_bar) + END)
    print()


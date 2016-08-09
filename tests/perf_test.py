import sys
import timeit

from collections import defaultdict

import imageio
import pytest
import numpy as np
import nphusl
import husl


default_impls = "simd cython numexpr numpy".split()


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


@pytest.fixture
def img(request):
    if CachedImg.rgb is None:
        path = request.config.getoption("--img")
        CachedImg.rgb = imageio.imread(path)
        CachedImg.hsl = nphusl.to_husl(CachedImg.rgb)
    return CachedImg


def _test_all(fn, arg, env, impls, iters):
    env = {**globals(), **env}
    print("\n\n{}({}) -- best of {}".format(fn, arg, iters))
    times = {}
    for impl in impls:
        enable = getattr(nphusl, "{}_enabled".format(impl))
        with enable():
            runs = timeit.repeat("{}({})".format(fn, arg),
                                 repeat=iters, number=1, globals=env)
        times[impl] = min(runs)
    worst = max(times.values())
    very_best = min(times.values())
    sorted_times = sorted(times.items(), key=lambda x: x[1])
    for impl, best in sorted_times:
        spaces = len("standard") - len(impl)
        chart_bar = "|" * int(40*best/worst)
        tformat = "{:10.4e} s" if very_best < 0.001 else "{:10.4f}"
        times_slower = best/very_best
        slower = "fastest" if times_slower == 1 else \
                 "{:0.2f}x slower".format(times_slower)
        t = tformat.format(best)
        print("  {}: {}{}   {} ({})".format(
              impl, " "*spaces, t, chart_bar, slower))
    print()


def test_perf_husl_to_rgb(impls, iters, img):
    fn = "nphusl.to_rgb"
    _test_all(fn, "img.hsl", locals(), impls, iters)


def test_perf_rgb_to_husl(impls, iters, img):
    fn = "nphusl.to_husl"
    _test_all(fn, "img.rgb", locals(), impls, iters)


def test_perf_rgb_to_husl_one_triplet(impls, iters):
    fn = "nphusl.to_husl"
    rgb_triplet = (np.random.rand(3) * 255).astype(np.uint8)
    rgb_triplet = list(rgb_triplet)
    _test_all(fn , "rgb_triplet", locals(), impls, iters)


def test_perf_rgb_to_hue(impls, iters, img):
    fn = "nphusl.to_hue"
    _test_all(fn, "img.rgb", locals(), impls, iters)


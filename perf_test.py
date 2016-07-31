import sys
import timeit

import pytest
import numpy as np
import nphusl
import husl


default_impls = "simd cython numexpr standard".split()


@pytest.fixture
def impls(request):
    methods = request.config.getoption("--impls")
    assert all(m in default_impls for m in methods)
    return methods


def _test_all(fn, arg, env, impls):
    env = {**globals(), **env}
    print("\n\n{}({}) ====".format(fn, arg))
    for impl in impls:
        enable = getattr(nphusl, "{}_enabled".format(impl))
        with enable():
            t = timeit.timeit("{}({})".format(fn, arg), number=1, globals=env)
        spaces = len("standard") - len(impl)
        print("  {}: {}{:0.4f}".format(impl, " "*spaces, t))
    print()


def test_perf_husl_to_rgb(impls):
    hsl = np.random.rand(1920, 1080, 3) * 100
    fn = "nphusl.husl_to_rgb"
    _test_all(fn, "hsl", locals(), impls)


def test_perf_rgb_to_husl(impls):
    rgb = np.random.rand(1920, 1080, 3)
    fn = "nphusl.rgb_to_husl"
    _test_all(fn, "rgb", locals(), impls)


def test_perf_rgb_to_hue(impls):
    rgb = np.random.rand(1920, 1080, 3)
    fn = "nphusl.rgb_to_hue"
    _test_all(fn, "rgb", locals(), impls)


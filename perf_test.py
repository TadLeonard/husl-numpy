import numpy as np
import nphusl
import husl
import timeit


def _test_all(fn, arg, env):
    env = {**globals(), **env}
    print("\n\n{}({}) ====".format(fn, arg))
    for method in "simd cython numexpr standard".split():
        enable = getattr(nphusl, "enable_{}_fns".format(method))
        nphusl.enable_standard_fns()
        enable()
        t = timeit.timeit("{}({})".format(fn, arg), number=1, globals=env)
        spaces = len("standard") - len(method)
        print("  {}: {}{:0.4f}".format(method, " "*spaces, t))
    print()


def test_perf_husl_to_rgb():
    hsl = np.random.rand(1920, 1080, 3) * 100
    fn = "nphusl.husl_to_rgb"
    _test_all(fn, "hsl", locals())


def test_perf_rgb_to_husl():
    rgb = np.random.rand(1920, 1080, 3)
    fn = "nphusl.rgb_to_husl"
    _test_all(fn, "rgb", locals())


def test_perf_rgb_to_hue():
    rgb = np.random.rand(1920, 1080, 3)
    fn = "nphusl.rgb_to_hue"
    _test_all(fn, "rgb", locals())


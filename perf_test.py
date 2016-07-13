import numpy as np
import nphusl
import _nphusl_cython as cy
import husl
import timeit


def test_perf_cython_husl_to_rgb():
    hsl = np.random.rand(1920, 1080, 3) * 100
    nphusl.enable_cython_fns()
    go_cy = cy.husl_to_rgb
    t_cy = timeit.timeit("go_cy(hsl)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    nphusl.enable_numexpr_fns()
    go_ex = nphusl.husl_to_rgb
    t_ex = timeit.timeit("go_ex(hsl)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    go_np = nphusl.husl_to_rgb
    t_np = timeit.timeit("go_np(hsl)", number=1, globals=locals())
    print("\n1080p image HUSL->RGB: "
          "\nCython: {}s\nNumExpr: {}s\nNumpy: {}s".format(
            t_cy, t_ex, t_np))
    assert t_cy < t_ex < t_np
    assert t_np / t_cy > 2


def test_perf_to_rgb():
    hsl = np.random.rand(1920, 1080, 3) * 100
    import nphusl
    nphusl.enable_standard_fns()
    cs = None
    t1 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    nphusl.enable_numexpr_fns()
    t2 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    nphusl.enable_cython_fns()
    t3 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    print("to_rgb()\nCython: {}\nNumExpr: {}\nNumpy: {}".format(
          t3, t2, t1))
    assert t3 < t2 < t1
    assert t1 / t3 > 2


def test_perf_to_rgb_2d():
    hsl = np.random.rand(19200, 3) * 100
    import nphusl
    nphusl.enable_standard_fns()
    cs = None
    t1 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    nphusl.enable_numexpr_fns()
    t2 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    nphusl.enable_cython_fns()
    t3 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    print("to_rgb()\nCython: {}\nNumExpr: {}\nNumpy: {}".format(
          t3, t2, t1))
    assert t3 < t2 < t1


def test_perf_rgb_to_husl():
    rgb = np.random.rand(1920, 1080, 3)
    nphusl.enable_cython_fns()
    go_cy = cy.rgb_to_husl
    t_cy = timeit.timeit("go_cy(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    nphusl.enable_numexpr_fns()
    go_ex = nphusl.rgb_to_husl
    t_ex = timeit.timeit("go_ex(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    go_np = nphusl.rgb_to_husl
    t_np = timeit.timeit("go_np(rgb)", number=1, globals=locals())
    print("\n1080p image RGB->HUSL: "
          "\nCython: {}s\nNumExpr: {}s\nNumpy: {}s".format(
            t_cy, t_ex, t_np))
    assert t_cy < t_ex < t_np
    assert t_np / t_cy > 2


def test_perf_rgb_to_husl_2d():
    rgb = np.random.rand(19200, 3)
    nphusl.enable_cython_fns()
    go_cy = cy.rgb_to_husl
    t_cy = timeit.timeit("go_cy(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    nphusl.enable_numexpr_fns()
    go_ex = nphusl.rgb_to_husl
    t_ex = timeit.timeit("go_ex(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    go_np = nphusl.rgb_to_husl
    t_np = timeit.timeit("go_np(rgb)", number=1, globals=locals())
    print("\n2D RGB->HUSL: "
          "\nCython: {}s\nNumExpr: {}s\nNumpy: {}s".format(
            t_cy, t_ex, t_np))
    assert t_cy < t_ex and t_cy < t_np


def test_perf_rgb_to_hue():
    rgb = np.random.rand(1920, 1080, 3)
    nphusl.enable_cython_fns()
    go_cy = cy.rgb_to_hue
    t_cy = timeit.timeit("go_cy(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    nphusl.enable_numexpr_fns()
    go_ex = nphusl.rgb_to_hue
    t_ex = timeit.timeit("go_ex(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    go_np = nphusl.rgb_to_hue
    t_np = timeit.timeit("go_np(rgb)", number=1, globals=locals())
    print("\n1080p image RGB->HUE: "
          "\nCython: {}s\nNumExpr: {}s\nNumpy: {}s".format(
            t_cy, t_ex, t_np))
    assert t_cy < t_ex < t_np
    assert t_np / t_cy > 2


def test_perf_rgb_to_hue_2d():
    rgb = np.random.rand(19200, 3)
    nphusl.enable_cython_fns()
    go_cy = cy.rgb_to_hue
    t_cy = timeit.timeit("go_cy(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    nphusl.enable_numexpr_fns()
    go_ex = nphusl.rgb_to_hue
    t_ex = timeit.timeit("go_ex(rgb)", number=1, globals=locals())
    nphusl.enable_standard_fns()
    go_np = nphusl.rgb_to_hue
    t_np = timeit.timeit("go_np(rgb)", number=1, globals=locals())
    print("\n1080p image RGB->HUE: "
          "\nCython: {}s\nNumExpr: {}s\nNumpy: {}s".format(
            t_cy, t_ex, t_np))
    assert t_cy < t_ex < t_np
    assert t_np / t_cy > 2



import numpy as np
import nphusl
import _nphusl_cython as cy
import husl
import timeit


def test_perf_cython_max_chroma():
    go_cyth = cy._grind_max_chroma
    go_husl = husl.max_chroma_for_LH
    go_nump = nphusl._max_lh_chroma
    lch = np.zeros(shape=(1000, 3), dtype=np.float)
    lch[:, 0] = 0.25
    lch[:, 2] = 40.0
    t_cyth = timeit.timeit("go(10000, 0.25, 40.0)", number=1, globals={"go": go_cyth})
    t_cyth /= 1
    t_husl = timeit.timeit("go(0.25, 40.0)", number=10000,
                       globals={"go": go_husl})
    t_nump = timeit.timeit("go(lch)", number=10, globals={"go": go_nump, "lch": lch})
    print("\nCython: {} speedup".format(t_husl/t_cyth))
    print("Numpy: {} speedup".format(t_husl/t_nump))
    assert (t_husl / t_cyth) > 80  # cython version should be better than 90x speedup


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
    assert t_np / t_cy > 3


def test_perf_to_rgb_cython():
    hsl = np.random.rand(1920, 1080, 3) * 100
    import nphusl
    nphusl.enable_standard_fns()
    cs = None
    t1 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    nphusl.enable_numexpr_fns()
    t2 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    nphusl.enable_cython_fns()
    t3 = timeit.timeit("nphusl.to_rgb(hsl, chunksize=cs)", number=1, globals=locals())
    print("to_rgb()\nwith numpy: {}\nwith NumExpr: {}\nwith Cython: {}".format(
          t1, t2, t3))


def test_perf_cython_rgb_to_husl():
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
    assert t_np / t_cy > 3




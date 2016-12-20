"""Tests accuracy of RGB <-> HUSL conversions.

For RGB -> HUSL, the goal is for a RGB -> HUSL -> RGB round trip
to produce an RGB triplet that's not off by more than TWO (2) on any channel.

For HUSL -> RGB, the goal is for a HUSL -> RGB -> HUSL round trip
to produce an HSL tripet that's not off by more than 1% on any channel."""


import imageio
import numpy as np
import pytest
import nphusl
import tabulate


# ASCII formatting codes
BOLD = "\033[1m"
FAINT = "\033[2m"
END = "\033[0m"


np.set_printoptions(threshold=5, precision=3,
                    formatter={"float_kind": "{:7.3f}".format})


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


def test_accuracy(img):
    with nphusl.simd_enabled():
        hsl = nphusl.to_husl(img.rgb)
    with nphusl.numpy_enabled():
        rgb = nphusl.to_rgb(hsl)
        hsl_ref = nphusl.to_husl(img.rgb)
    size = hsl.shape[0] * hsl.shape[1]
    hsl_flat = hsl.reshape((size, 3))
    rgb_flat = rgb.reshape((size, 3))
    hsl_ref_flat = hsl_ref.reshape((size, 3))
    rgb_ref_flat = img.rgb.reshape((size, 3))
    rgb_diff = np.abs(rgb_flat.astype(int) - rgb_ref_flat)
    hsl_diff = np.abs(hsl_flat - hsl_ref_flat)
    h_err, s_err, l_err = (hsl_diff[..., n] for n in range(3))
    h, s, l = (hsl_flat[..., n] for n in range(3))
    percentiles = [0, 25, 50, 90, 95, 96, 97, 98,
                   99.5, 99.6, 99.7, 99.8, 99.9, 100]

    def print_err_tables():
        fields = "Percentile", "Red error", "Green error", "Blue error", " "
        print(BOLD + "\nIMG->HUSL->RGB roundtrip error" + END)
        print(_error_table(rgb_diff, percentiles, fields))
        fields = "Percentile", "Hue error", "Sat error", "Light error", " "
        print(BOLD + "\nIMG->HUSL error vs. reference impl." + END)
        print(_error_table(hsl_diff, percentiles, fields))
        for i, name in enumerate("hue saturation lightness".split()):
            c = hsl_flat[..., i]
            c_err = hsl_diff[..., i]
            err_99 = np.percentile(c_err, 99)
            print(BOLD + "\nTypical RGB for {} error above 99th "
                  "percentile: ".format(name) + END)
            print(img.rgb[c_err.reshape(rgb.shape[:-1]) > err_99])
            print(BOLD + "\nTypical HUSL for {} error above 99th "
                  "percentile: ".format(name) + END)
            print(hsl[c_err.reshape(rgb.shape[:-1]) > err_99])

    print(BOLD + "\nAll RGB & HUSL errors")
    print(         "=====================" + END)
    print_err_tables()
    h_err[s < 0.1] = 0  # hue errors for low saturation have no meaning
    rgb_diff[s < 0.1] = 0
    s_err[l > 99.5] = 0  # saturation errors when very bright not meaningful
    rgb_diff[l > 99.5] = 0
    s_err[l < 1] = 0  # saturation errors when very dim not meaningful
    rgb_diff[l < 1] = 0
    print(BOLD + "\nPerceptible RGB & HUSL errors")
    print(         "=============================" + END)
    print_err_tables()

    max_err = max(np.percentile(rgb_diff, 100, axis=0))
    mask = np.any(rgb_diff == max_err, axis=1)
    print(BOLD + "\nMost challenging pixel")
    print(         "======================" + END)
    print("RGB input vs. output: {} -> {}".format(
        rgb_ref_flat[mask].squeeze(), rgb_flat[mask].squeeze()))
    print("HUSL ouput vs. reference impl.: {} vs. {}".format(
        hsl_flat[mask].squeeze(), hsl_ref_flat[mask].squeeze()))
    src = "_accuracy_test_source.png"
    rec = "_accuracy_test_recreated.png"
    print("\nWriting PNGs: {}, {}".format(src, rec))
    imageio.imwrite(src, img.rgb)
    imageio.imwrite(rec, rgb)


PCT_NOTES = {100: "(max)", 0: "(min)", 50: "(median)"}


def _error_table(flat_diff, percentiles, fields):
    rows = []
    is_int = np.issubdtype(flat_diff.dtype, np.integer)
    for pct in percentiles:
        chans = (np.percentile(flat_diff[..., n], pct)
                 for n in range(3))
        if is_int:
            chans = (int(c) for c in chans)
        a, b, c = chans
        rows.append([pct, a, b, c, PCT_NOTES.get(pct, "")])
    table = tabulate.tabulate(rows, headers=fields, tablefmt="simple",
                              floatfmt="6.2f")
    return table


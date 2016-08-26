"""Tests accuracy of RGB <-> HUSL conversions.

For RGB -> HUSL, the goal is for a RGB -> HUSL -> RGB round trip
to produce an RGB triplet that's not off by more than TWO (2) on any channel.

For HUSL -> RGB, the goal is for a HUSL -> RGB -> HUSL round trip
to produce an HSL tripet that's not off by more than 1% on any channel."""


import imageio
import numpy as np
import pytest
import nphusl


np.set_printoptions(threshold=np.inf, precision=3)


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
    #avg_rgb_diff = np.sum(np.sum(rgb_diff, axis=1), axis=0)/(diff.size/3)
    #max_rgb_diff = np.max(np.max(rgb_diff, axis=1), axis=0)
    #avg_hsl_diff = np.sum(np.sum(hsl_diff, axis=1), axis=0)/(diff.size/3)
    #max_hsl_diff = np.max(np.max(hsl_diff, axis=1), axis=0)
    print()
    print("==== RGB -> HUSL -> RGB error ====")
    #print("  Avg RGB->HUSL->RGB error:", avg_rgb_diff)
    #print("  Max RGB->HUSL->RGB error:", max_rgb_diff)
    for pct in range(0, 90, 30):
        print("{:3d}th percentile: {:s}".format(
              pct, str(np.percentile(rgb_diff, pct, axis=0))))
    for pct in range(90, 101, 1):
        print("{:3d}th percentile: {:s}".format(
              pct, str(np.percentile(rgb_diff, pct, axis=0))))
    print("==== RGB -> HUSL error ====")
    for pct in range(0, 90, 30):
        print("{:3d}th percentile: {:s}".format(
              pct, str(np.percentile(hsl_diff, pct, axis=0))))
    for pct in range(90, 101, 1):
        print("{:3d}th percentile: {:s}".format(
              pct, str(np.percentile(hsl_diff, pct, axis=0))))
    #print("  Avg RGB->HUSL error:", avg_hsl_diff)
    #print("  Max RGB->HUSL error:", max_hsl_diff)
    #where_large = np.where(np.any(diff >= 9, axis=2))
    #print(rgb[where_large])
    #print(img.rgb[where_large])
    #print(hsl_ref[where_large])
    #print(hsl[where_large])
    #print(hsl_ref[where_large] - hsl[where_large])
    imageio.imwrite("HORK.jpg", rgb, quality=100)


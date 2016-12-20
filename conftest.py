
import pytest


default_impls = "simd cython numexpr numpy".split()


def pytest_addoption(parser):
    parser.addoption("--impls", action="store", nargs="*",
                     default=default_impls)
    parser.addoption("--iters", action="store", type=int, default=1)
    parser.addoption("--img", action="store",
                     default="tests/rand_4million.jpg")



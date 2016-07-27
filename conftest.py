
import pytest


default_impls = "simd cython numexpr standard".split()


def pytest_addoption(parser):
    parser.addoption("--impls", action="store", nargs="*",
                     default=default_impls)



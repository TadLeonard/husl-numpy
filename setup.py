from distutils.core import setup
from nphusl import __version__


setup(name='nphusl',
      version=__version__,
      py_modules=['nphusl', "_nphusl_expr"],
)

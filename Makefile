
all:
	python setup.py build_ext --inplace --cythonize

cython:
	python setup.py build_ext --inplace --cythonize --no-simd-ext

simd:
	python setup.py build_ext --inplace --cythonize --no-cython-ext

clean_all:
	rm -rf build/
	rm -f nphusl/*.so
	rm -f nphusl/_cython_opt.c
	rm -f nphusl/_simd_wrap.c

clean:
	rm -rf build/
	rm -f nphusl/*.so


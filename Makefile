
all:
	python setup.py build_ext --inplace --use-cython


clean:
	rm -rf build/
	rm -f nphusl/*.so
	rm -f nphusl/_cython_opt.c
	rm -f nphusl/_simd_wrap.c


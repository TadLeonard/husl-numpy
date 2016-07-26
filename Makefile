
all:
	python setup.py build_ext --inplace --use-cython


clean:
	rm -rf build/
	rm nphusl/*.so
	rm nphusl/_nphusl_cython.c


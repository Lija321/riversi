all:
	clear

pure:
	gcc -shared -o riversi_c_utils.so riversi_c_utils.c -lpython3.10 -I/usr/include/python3.10 -fPIC
	clear
cython:
	python3 setup.py build_ext --inplace
	clear

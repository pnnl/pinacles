gfortran -fPIC -O3 -c  -ffree-line-length-512 -freal-4-real-8 module_mp_p3.f95
gfortran -shared -fPIC -O3 -freal-4-real-8 module_mp_p3.o p3_wrapper.F90 -o lib_p3.so

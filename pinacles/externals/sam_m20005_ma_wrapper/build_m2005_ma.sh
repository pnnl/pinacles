gfortran -fPIC -O3 -c  -ffree-line-length-512 -freal-4-real-8 module_mp_2005_ma.f95
gfortran -shared -fPIC -O3 -freal-4-real-8 module_mp_m2005_ma.o m2005_ma_wrapper.F90 -o lib_m2005_ma.so

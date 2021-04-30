gfortran -fPIC -O3 -c  -fbounds-check -ffree-line-length-512 -freal-4-real-8 module_bl_acm.F90
gfortran -shared -fPIC -O3 -fbounds-check -freal-4-real-8 module_bl_acm.o acm_wrapper.F90 -o lib_acm.so

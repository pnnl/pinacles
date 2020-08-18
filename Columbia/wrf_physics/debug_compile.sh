#f2py -c module_mp_fast_sbm.F90 --verbose --opt='-fdefault-real-8 -fcheck=all -ftree-vectorize' -m module_mp_fast_sbm


gfortran -c module_mp_SBM_polar_radar.f90  
gfortran -c module_mp_fast_sbm_warm.F90 
f2py -c module_mp_fast_sbm_warm.F90 --verbose --opt='' -m module_mp_fast_sbm_warm

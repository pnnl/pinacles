#f2py -c module_mp_fast_sbm.F90 --verbose --opt='-fcheck=all -ftree-vectorize' -m module_mp_fast_sbm

f2py -c module_mp_fast_sbm.F90 --verbose  -m module_mp_fast_sbm

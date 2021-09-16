
# # build RRTMG_LW
gfortran -ffixed-line-length-none -freal-4-real-8 \
      -fno-range-check   -fPIC -O3 -c \
      parcel_model_ssp/constant.f90  \
      parcel_model_ssp/empirical.f90  \
      parcel_model_ssp/DVODE_F90_M.f90 \
      parcel_model_ssp/chem.f90 \
      parcel_model_ssp/aerospec.f90 \
      parcel_model_ssp/state.f90 \
      parcel_model_ssp/dropspec.f90 \
      parcel_model_ssp/dynam.f90 \
      parcel_model_ssp/deli.f90 \
      parcel_model_ssp/cloudspec.f90 \
      parcel_model_ssp/cond.f90 \
      parcel_model_ssp/cpm.f90

 
  
gfortran -ffixed-line-length-none -freal-4-real-8 -fno-range-check -O3 \
 -shared -fPIC \
 constant.o \
 empirical.o \
 DVODE_F90_M.o \
 chem.o \
 aerospec.o \
 state.o \
 dropspec.o \
 dynam.o \
 deli.o \
 cloudspec.o \
 cond.o \
 cpm.o  parcel_model_wrapper.f90 -o libparcel.so

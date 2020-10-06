

# # build RRTMG_LW
gfortran -ffixed-line-length-none -freal-4-real-8 -fno-range-check   -fPIC -O3 \
   -c lw/modules/parkind.f90  lw/modules/parrrtm.f90  \
  lw/modules/rrlw_cld.f90  lw/modules/rrlw_con.f90  lw/modules/rrlw_kg01.f90 lw/modules/rrlw_kg02.f90 lw/modules/rrlw_kg03.f90 lw/modules/rrlw_kg04.f90 \
  lw/modules/rrlw_kg05.f90 lw/modules/rrlw_kg06.f90 lw/modules/rrlw_kg07.f90 lw/modules/rrlw_kg08.f90 \
  lw/modules/rrlw_kg09.f90 lw/modules/rrlw_kg10.f90 lw/modules/rrlw_kg11.f90 lw/modules/rrlw_kg12.f90 \
  lw/modules/rrlw_kg13.f90 lw/modules/rrlw_kg14.f90 lw/modules/rrlw_kg15.f90 lw/modules/rrlw_kg16.f90 \
  lw/modules/rrlw_ncpar.f90 lw/modules/rrlw_ref.f90 lw/modules/rrlw_tbl.f90 lw/modules/rrlw_vsn.f90 \
  lw/modules/rrlw_wvn.f90 \
  lw/src/rrtmg_lw_cldprmc.f90 lw/src/rrtmg_lw_cldprop.f90 lw/src/rrtmg_lw_rtrn.f90 \
  lw/src/rrtmg_lw_rtrnmc.f90 lw/src/rrtmg_lw_rtrnmr.f90 lw/src/rrtmg_lw_setcoef.f90 \
  lw/src/rrtmg_lw_taumol.f90 lw/src/rrtmg_lw_k_g.f90 lw/src/rrtmg_lw_init.f90 \
  lw/src/rrtmg_lw_rad.nomcica.f90 
  
gfortran -ffixed-line-length-none -freal-4-real-8 -fno-range-check -O3 \
 -shared -fPIC parkind.o \
  parrrtm.o rrlw_cld.o rrlw_con.o \
  rrlw_kg01.o rrlw_kg02.o rrlw_kg03.o rrlw_kg04.o \
  rrlw_kg05.o rrlw_kg06.o rrlw_kg07.o rrlw_kg08.o \
  rrlw_kg09.o rrlw_kg10.o rrlw_kg11.o rrlw_kg12.o \
  rrlw_kg13.o rrlw_kg14.o rrlw_kg15.o rrlw_kg16.o \
  rrlw_ncpar.o rrlw_ref.o rrlw_tbl.o rrlw_vsn.o \
  rrlw_wvn.o \
  rrtmg_lw_cldprmc.o rrtmg_lw_cldprop.o rrtmg_lw_rtrn.o \
  rrtmg_lw_rtrnmc.o rrtmg_lw_rtrnmr.o rrtmg_lw_setcoef.o \
  rrtmg_lw_taumol.o rrtmg_lw_k_g.o rrtmg_lw_init.o \
  rrtmg_lw_rad.nomcica.o  wrapper_rrtmg_lw.f90 -o librrtmglw.so

# # build RRTMG_SW - Note that parkind.f90 is the same as RRTMG_LW
gfortran -ffixed-line-length-none -freal-4-real-8 -fno-range-check   -fPIC -O3 -c \
  sw/modules/parkind.f90 sw/modules/parrrsw.f90 sw/modules/rrsw_aer.f90 \
  sw/modules/rrsw_cld.f90 sw/modules/rrsw_con.f90 \
  sw/modules/rrsw_kg16.f90 sw/modules/rrsw_kg17.f90 sw/modules/rrsw_kg18.f90 sw/modules/rrsw_kg19.f90 \
  sw/modules/rrsw_kg20.f90 sw/modules/rrsw_kg21.f90 sw/modules/rrsw_kg22.f90 sw/modules/rrsw_kg23.f90 \
  sw/modules/rrsw_kg24.f90 sw/modules/rrsw_kg25.f90 sw/modules/rrsw_kg26.f90 sw/modules/rrsw_kg27.f90 \
  sw/modules/rrsw_kg28.f90 sw/modules/rrsw_kg29.f90 \
  sw/modules/rrsw_ncpar.f90 sw/modules/rrsw_ref.f90 sw/modules/rrsw_tbl.f90 sw/modules/rrsw_vsn.f90 \
  sw/modules/rrsw_wvn.f90 \
  sw/src/rrtmg_sw_cldprmc.f90 sw/src/rrtmg_sw_cldprop.f90 sw/src/rrtmg_sw_reftra.f90\
  sw/src/rrtmg_sw_vrtqdr.f90 sw/src/rrtmg_sw_taumol.f90 \
  sw/src/rrtmg_sw_spcvmc.f90 sw/src/rrtmg_sw_spcvrt.f90 sw/src/rrtmg_sw_setcoef.f90 \
  sw/src/rrtmg_sw_k_g.f90 sw/src/rrtmg_sw_init.f90 \
  sw/src/rrtmg_sw_rad.nomcica.f90 

gfortran -ffixed-line-length-none -freal-4-real-8 -fno-range-check  -shared -fPIC -O3 parkind.o \
  parrrtm.o parrrsw.o rrsw_aer.o rrsw_cld.o rrsw_con.o \
  rrsw_kg16.o rrsw_kg17.o rrsw_kg18.o rrsw_kg19.o \
  rrsw_kg20.o rrsw_kg21.o rrsw_kg22.o rrsw_kg23.o \
  rrsw_kg24.o rrsw_kg25.o rrsw_kg26.o rrsw_kg27.o \
  rrsw_kg28.o rrsw_kg29.o \
  rrsw_ncpar.o rrsw_ref.o rrsw_tbl.o rrsw_vsn.o \
  rrsw_wvn.o \
  rrtmg_sw_cldprmc.o rrtmg_sw_cldprop.o rrtmg_sw_reftra.o \
  rrtmg_sw_vrtqdr.o rrtmg_sw_taumol.o \
  rrtmg_sw_spcvmc.o rrtmg_sw_spcvrt.o rrtmg_sw_setcoef.o \
  rrtmg_sw_k_g.o rrtmg_sw_init.o \
  rrtmg_sw_rad.nomcica.o  \
  wrapper_rrtmg_sw.f90 -o librrtmgsw.so

from cffi import FFI
import numpy as np
import os
import pathlib

class AerosolModel:
    def __init__(self):
        self.ffi = FFI()
        path = pathlib.Path(__file__).parent.absolute()
        # Grab the shared library for which we are building an ffi 
        self._lib = self.ffi.dlopen(os.path.join(path, "libparcel.so"))

          # Define the interfaces
        self.ffi.cdef(
            "void c_parcel_model_init(int ichem, int nbins);",
            override=True,
        )

        self.ffi.cdef(
            "void c_parcel_model_main(int npts, int nbins, double dt_les, double dt_cpm,\
                double temp[], double press[], double qv[], double ql[], double density[], \
                double pmass_total_aero_m[], double pmass_total_m[]);",
            override=True,
        )

        return
    def init( self, ichem=2, nbins=20):
        self._lib.c_parcel_model_init(ichem,nbins)
        return
    
    def update(
        self,
        npts,
        nbins,
        dt_les,
        dt_cpm,
        temp,
        press,
        qv,
        ql,
        density,
        pmass_total_aero_m,
        pmass_total_m):
        
      
        self._lib.c_parcel_model_main(
            npts, 
            nbins, 
            dt_les, 
            dt_cpm,
            self.as_pointer(temp),
            self.as_pointer(press),
            self.as_pointer(qv),
            self.as_pointer(ql),
            self.as_pointer(density),
            self.as_pointer(pmass_total_aero_m),
            self.as_pointer(pmass_total_m))
       
        
        return

    def as_pointer(self, numpy_array):
        assert numpy_array.flags[
            "F_CONTIGUOUS"
        ], "array is not contiguous in memory (Fortran order)"
        return self.ffi.cast("double*", numpy_array.ctypes.data)    


# if __name__ == "__main__":
#     print("Hi!")
#     AM = AerosolModel()
#     ichem = 1
#     nbins = 20
#     AM.init(ichem,nbins)
#     npts =1
   
#     dt_les = 1.0
#     dt_cpm = 1.0e-3
#     temp = np.array([295.0],dtype=np.double, order='F')
#     press = np.array([1.0e5],dtype=np.double, order='F')
#     qv = np.array([0.013352],dtype=np.double, order='F')
#     density = np.array([1.15635],dtype=np.double, order='F')
#     ql = np.array([0.0],dtype=np.double, order='F')

#     pmass_total_v_1d = np.array([3.2284815857675971E-010,  
#      3.0282833501587459E-009,  
#      2.0574839180891118E-008,  
#      1.0125538641205929E-007,  
#      3.6094544610661970E-007,   
#      9.3198032455469988E-007,   
#      1.7430653363728750E-006,
#      2.3613607177184992E-006,   
#      2.3171425255086249E-006,   
#      1.6469670350005267E-006,   
#      8.4792767656735483E-007,   
#      3.1620917043412546E-007,   
#      8.5414549797212355E-008,   
#      1.6712098895213352E-008,   
#      2.3684917876976295E-009,   
#      2.4313893775602493E-010,
#      1.8079186806711397E-011,  
#      9.7374449386298368E-013,   
#      3.7988564519985590E-014,   
#      1.0735011073706130E-015]) * 1e3
#     pmass_total_v = pmass_total_v_1d[np.newaxis,:]
#     pmass_total_aero_v_1d  = np.array([9.9764977265640035E-020,   
#      1.7569629875207928E-018,   
#      2.2412425703226992E-017,  
#      2.0708891780274985E-016,   
#      1.3860107533911417E-015,  
#      6.7192080169112066E-015, 
#      2.3594529341406579E-014, 
#      6.0013132495927233E-014,  
#      1.1056636178112736E-013,   
#      1.4755077257212911E-013, 
#      1.4262698503376696E-013,   
#      9.9862782179249526E-014, 
#      5.0646258385355235E-014,   
#      1.8605134621048444E-014,
#      4.9506247959060571E-015,   
#      9.5417702255474596E-016,   
#      1.3321086274572905E-016,   
#      1.3470759216795379E-017,   
#      9.8670255152109356E-019,   
#      5.2350661596357304E-020]) * 1e3
#     pmass_total_aero_v = pmass_total_aero_v_1d[np.newaxis,:]
    

#     temp_pointer=AM.as_pointer(temp)
#     pmass_total_aero_m = np.divide(pmass_total_aero_v,density)
#     pmass_total_m =np.divide(pmass_total_v,density)
#     print(np.shape(pmass_total_m))
    

#     AM.update(
#         npts,
#         nbins,
#         dt_les,
#         dt_cpm,
#         temp,
#         press,
#         qv,
#         ql,
#         density,
#         pmass_total_aero_m,
#         pmass_total_m)
#     print(temp, qv)
#     print(pmass_total_m)
#     print("Bye!")


if __name__ == "__main__":
    print("Hi!")
    AM = AerosolModel()
    ichem = 1
    nbins = 20
    AM.init(ichem,nbins)
    npts =1
   
    dt_les = 1e-3
    dt_cpm = 1.0e-3
    temp = np.array([298.5],dtype=np.double, order='F')
    press = np.array([101268.4404536949],dtype=np.double, order='F')
    qv = np.array([0.016973076923076924],dtype=np.double, order='F')
    density = np.array([1.1749611764742434],dtype=np.double, order='F')
    ql = np.array([0.0],dtype=np.double, order='F')
    qv_init = np.copy(qv)
    temp_init = np.copy(temp)

    pmass_total_m_1d = np.array([2.747735950382197e-07 , 
                                        2.5773486519721743e-06 , 
                                        1.7511086445106992e-05 , 
                                        8.617767164952262e-05 , 
                                        0.0003071978447186066 ,
                                        0.0007932011163051141 , 
                                        0.0014835090948997413 , 
                                        0.0020097352674499002 , 
                                        0.0019721013860930473 , 
                                        0.001401720313332624 , 
                                        0.0007216643301611166 ,
                                        0.0002691230256648675 ,
                                        7.269561546503303e-05 ,
                                        1.4223527699828144e-05 , 
                                        2.0158033938863198e-06 , 
                                        2.069334819061005e-07 , 
                                        1.5387040706528158e-08 , 
                                        8.287455017873802e-10 , 
                                        3.2331734940576714e-11 , 
                                        9.136473089608889e-13]) 
    pmass_total_m = pmass_total_m_1d[np.newaxis,:]
    pmass_total_m_init = np.copy(pmass_total_m)
    pmass_total_aero_m_1d  = np.array([8.490919984697453e-17 , 
                                        1.4953375369157446e-15 , 
                                        1.9075040406728408e-14 , 
                                        1.7625175166913328e-13 , 
                                        1.1796228086869031e-12 , 
                                        5.7186645853426774e-12 , 
                                        2.008111542116689e-11 , 
                                        5.107669456116709e-11 , 
                                        9.410213563945272e-11 , 
                                        1.255792600317368e-10 , 
                                        1.213886561784742e-10 , 
                                        8.499238801096614e-11 , 
                                        4.310460838880605e-11 , 
                                        1.5834673759220534e-11 , 
                                        4.213434824116621e-12 , 
                                        8.120919113225409e-13 , 
                                        1.1337461709902588e-13 , 
                                        1.1464846011351839e-14 , 
                                        8.397738804821199e-16 , 
                                        4.455518419612693e-17 ]) 
    pmass_total_aero_m = pmass_total_aero_m_1d[np.newaxis,:]
    print(np.shape(pmass_total_aero_m))

    temp_pointer=AM.as_pointer(temp)
    
    print(np.shape(pmass_total_m))
    

    AM.update(
        npts,
        nbins,
        dt_les,
        dt_cpm,
        temp,
        press,
        qv,
        ql,
        density,
        pmass_total_aero_m,
        pmass_total_m)
    print('Printing from python side')
    # print(temp,temp_init, qv,qv_init)
    print(pmass_total_m-pmass_total_m_init)
    print(pmass_total_m-pmass_total_aero_m)
    # print(pmass_total_m_init)
    print('Changes balance?',np.sum(pmass_total_m)-np.sum(pmass_total_m_init),qv[0]-qv_init[0])
    print("Bye!")


from cffi import FFI
import numpy as np
import os 
import pathlib

class ACM:
    def __init__(self):
        self.ffi  = FFI()

        path = pathlib.Path(__file__).parent.absolute()

        # Grab the shared library we are building a ffi for
        self._lib_acm= self.ffi.dlopen(os.path.join(path, 'lib_acm.so'))

        # Define the interface for the 3D ACM subroutine
        #For now we define an interface assuming WRF_CHEM != 1
        self.ffi.cdef("void c_acmpbl( \
            int XTIME,  double DTPBL, double U3D[], double V3D[], \
            double PP3D[], double DZ8W[], double TH3D[], double T3D[], \
            double QV3D[], double QC3D[], double QI3D[], double RR3D[], \
            double UST[],  double HFX[], double QFX[], double TSK[],  \
            double PSFC[], double EP1, double G,  \
            double ROVCP, double RD, double CPD, \
            double PBLH[], int KPBL2D[], double EXCH_H[], double REGIME[],   \
            double GZ1OZ0[], double WSPD[], double PSIM[], double MUT[], double RMOL[],  \
            double RUBLTEN[], double RVBLTEN[], double RTHBLTEN[],         \
            double RQVBLTEN[], double RQCBLTEN[], double RQIBLTEN[],         \
            int ids, int ide, int jds, int jde, int kds, int kde, \
            int ims, int ime, int jms, int jme, int kms, int kme, \
            int its, int ite, int jts, int jte, int kts, int kte  \
        );", override=True )


        self.ffi.cdef("void test();", override=True)

    # additionally checks if the array is contiguous
    def as_pointer(self, numpy_array):
        assert numpy_array.flags['F_CONTIGUOUS'], \
            "array is not contiguous in memory (Fortran order)"
        return self.ffi.cast("double*", numpy_array.ctypes.data)

    # additionally checks if the array is contiguous
    def as_pointer_int(self, numpy_array):
        assert numpy_array.flags['F_CONTIGUOUS'], \
            "array is not contiguous in memory (Fortran order)"
        return self.ffi.cast("int*", numpy_array.ctypes.data)

def main():

    acm = ACM()

    ids = 0
    ims = 0
    its = 1 
    
    jds = 0
    jms = 0
    jts = 1
    
    kds = 0
    kms = 0 
    kts = 1 

    ide = 12
    ime = 12
    ite = 11
    
    jde = 12
    jme = 12
    jte = 11
    
    kde = 12
    kme = 12
    kte = 11

    XTIME = 1

    DTPBL = 1.0
    U3D = np.zeros((ide,jde,kde), dtype=np.double, order='F')
    V3D = np.zeros_like(U3D)
    PP3D = np.zeros_like(U3D)
    DZ8W = np.ones_like(U3D)
    TH3D = np.zeros_like(U3D)
    T3D = np.zeros_like(U3D)
    QV3D = np.zeros_like(U3D)
    QC3D = np.zeros_like(U3D)
    QI3D = np.zeros_like(U3D)
    RR3D = np.zeros_like(U3D)

    UST = np.zeros((ide, jde), dtype=np.double, order='F')
    HFX = np.zeros_like(UST)
    QFX = np.zeros_like(UST)
    TSK = np.zeros_like(UST)
    PSFC = np.zeros_like(UST)

    EP1 = 287.1/461.5
    G = 9.81
    ROVCP = 287.1/1004.0
    RD =  287.1
    CPD = 1004.0

    PBLH = np.zeros_like(UST)
    KPBL2D = np.zeros((ide,jde), dtype=np.int, order='F')
    EXCH_H = np.zeros_like(U3D)
    REGIME = np.zeros_like(UST)
    GZ10Z0 = np.zeros_like(UST)
    WSPD = np.zeros_like(UST)
    PSIM = np.zeros_like(UST)
    MUT = np.zeros_like(UST)
    RMOL = np.zeros_like(UST)

    RUBLTEN = np.zeros_like(U3D)
    RVBLTEN = np.zeros_like(U3D)
    RTHBLTEN = np.zeros_like(U3D)

    RQVBLTEN = np.zeros_like(U3D)
    RQCBLTEN = np.zeros_like(U3D)
    RQIBLTEN = np.zeros_like(U3D)


    acm._lib_acm.c_acmpbl(XTIME, DTPBL, acm.as_pointer(U3D), acm.as_pointer(V3D),
            acm.as_pointer(PP3D), acm.as_pointer(DZ8W), acm.as_pointer(TH3D), acm.as_pointer(T3D), 
            acm.as_pointer(QV3D), acm.as_pointer(QC3D), acm.as_pointer(QI3D), acm.as_pointer(RR3D), 
            acm.as_pointer(UST),  acm.as_pointer(HFX), acm.as_pointer(QFX), acm.as_pointer(TSK),  
            acm.as_pointer(PSFC), EP1, G,  
            ROVCP, RD, CPD, 
            acm.as_pointer(PBLH), acm.as_pointer_int(KPBL2D), acm.as_pointer(EXCH_H), acm.as_pointer(REGIME),   
            acm.as_pointer(GZ10Z0), acm.as_pointer(WSPD), acm.as_pointer(PSIM), acm.as_pointer(MUT), 
            acm.as_pointer(RMOL),  
            acm.as_pointer(RUBLTEN), acm.as_pointer(RVBLTEN), acm.as_pointer(RTHBLTEN),         
            acm.as_pointer(RQVBLTEN), acm.as_pointer(RQCBLTEN), acm.as_pointer(RQIBLTEN),         
            ids, ide, jds, jde, kds, kde, 
            ims, ime, jms, jme, kms, kme, 
            its, ite, jts, jte, kts, kte)

    return

if __name__ == '__main__':
    main()
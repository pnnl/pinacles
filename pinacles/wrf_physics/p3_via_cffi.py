from cffi import FFI
import numpy as np
import os
import pathlib

class P3:

    def __init__(self):
        self.ffi = FFI()


        path = pathlib.Path(__file__).parent.absolute()


        # Grab the shared library we are building a ffi for
        self._lib_p3= self.ffi.dlopen(os.path.join(path, 'lib_p3.so'))

        # Define the interfaces
        self.ffi.cdef("void c_p3_init(char dir_path_c[], int dir_path_len, int nCat);", override=True)

        self.ffi.cdef("void c_p3_main(int ids, int ide, int jds, int jde, int kds, int kde, \
            int ims, int ime, int jms, int jme, int kms, int kme, \
            int its, int ite, int jts, int jte, int kts, int kte, \
            double th_3d[], double qv_3d[], double qc_3d[], double qr_3d[], \
            double qnr_3d[], double diag_zdbz_3d[], double diag_effc_3d[], \
            double diag_effi_3d[], double diag_vmi_3d[], double diag_di_3d[], \
            double diag_rhopo_3d[], double th_old_3d[], double qv_old_3d[], \
            double qi1_3d[], double qni1_3d[], double qir1_3d[], double qib1_3d[], \
            double nc_3d[], double pii[], double p[], double dz[], double w[], \
            double RAINNC[], double RAINNCV[] ,double SR[], double SNOWNC[],double SNOWNCV[], \
            double dt, int itimestep, int n_iceCat  ); ", override=True)

        return

    def init(self, nCat=1):

        path = str(pathlib.Path(__file__).parent.absolute())
        path = path.encode('ascii')

        self._lib_p3.c_p3_init(self.ffi.new("char[]", path), len(path), nCat)

        return

    def update(self, ids, ide, jds, jde, kds, kde,
            ims, ime, jms, jme, kms, kme,
            its, ite, jts, jte, kts, kte,
            th_3d, qv_3d, qc_3d, qr_3d,
            qnr_3d, diag_zdbz_3d, diag_effc_3d,
            diag_effi_3d, diag_vmi_3d, diag_di_3d,
            diag_rhopo_3d, th_old_3d, qv_old_3d,
            qi1_3d, qni1_3d, qir1_3d, qib1_3d,
            nc_3d, pii, p, dz, w,
            RAINNC, RAINNCV ,SR, SNOWNC, SNOWNCV,
            dt, itimestep, n_iceCat):

        self._lib_p3.c_p3_main(
            ids, ide, jds, jde, kds, kde,
            ims, ime, jms, jme, kms, kme,
            its, ite, jts, jte, kts, kte,
            self.as_pointer(th_3d), self.as_pointer(qv_3d), self.as_pointer(qc_3d), self.as_pointer(qr_3d),
            self.as_pointer(qnr_3d), self.as_pointer(diag_zdbz_3d), self.as_pointer(diag_effc_3d),
            self.as_pointer(diag_effi_3d), self.as_pointer(diag_vmi_3d), self.as_pointer(diag_di_3d),
            self.as_pointer(diag_rhopo_3d), self.as_pointer(th_old_3d), self.as_pointer(qv_old_3d),
            self.as_pointer(qi1_3d), self.as_pointer(qni1_3d), self.as_pointer(qir1_3d), self.as_pointer(qib1_3d),
            self.as_pointer(nc_3d), self.as_pointer(pii), self.as_pointer(p), self.as_pointer(dz), self.as_pointer(w),
            self.as_pointer(RAINNC), self.as_pointer(RAINNCV) ,self.as_pointer(SR), self.as_pointer(SNOWNC), self.as_pointer(SNOWNCV),
            dt, itimestep, n_iceCat)

        return

    # function creates cdata variables of a type "double *" from a numpy array
    # additionally checks if the array is contiguous
    def as_pointer(self, numpy_array):
        assert numpy_array.flags['F_CONTIGUOUS'], \
            "array is not contiguous in memory (Fortran order)"
        return self.ffi.cast("double*", numpy_array.ctypes.data)



def main():

    p3 = P3()
    p3.init()


    ids = 1; jds = 1; kds = 1
    ide = 1; jde = 1; kde = 1
    ims=1; jms = 1; kms = 1
    ime=64; jme=64; kme=100
    its=1; jts=1; kts=1
    ite=ime; jte=jme; kte=kme

    shape_2d = (ime, jme)
    shape_3d = (ime, kme, jme)

    th_3d = np.zeros(shape_3d, dtype=np.double, order='F') + 300.0
    qv_3d = np.zeros_like(th_3d) + 1e-5
    qc_3d = np.zeros_like(th_3d)
    qr_3d = np.zeros_like(th_3d)
    qnr_3d = np.zeros_like(th_3d)
    diag_zdbz_3d = np.zeros_like(th_3d)
    diag_effc_3d = np.zeros_like(th_3d)
    diag_effi_3d = np.zeros_like(th_3d)
    diag_vmi_3d = np.zeros_like(th_3d)
    diag_di_3d = np.zeros_like(th_3d)
    diag_rhopo_3d = np.zeros_like(th_3d)
    th_old_3d = np.zeros_like(th_3d) + 300.0
    qv_old_3d = np.zeros_like(th_3d) + 1e-5
    qi1_3d = np.zeros_like(th_3d)
    qni1_3d = np.zeros_like(th_3d)
    qir1_3d = np.zeros_like(th_3d)
    qib1_3d = np.zeros_like(th_3d)
    nc_3d = np.zeros_like(th_3d)
    pii = np.zeros_like(th_3d) + 1.0
    p = np.zeros_like(th_3d) +  1e5
    dz = np.zeros_like(th_3d) + 5.0
    w = np.zeros_like(th_3d)

    RAINNC = np.zeros(shape_2d, dtype=np.double, order='F')
    RAINNCV = np.zeros_like(RAINNC)
    SR = np.zeros_like(RAINNC)
    SNOWNC = np.zeros_like(RAINNC)
    SNOWNCV = np.zeros_like(RAINNC)

    dt = 5.0
    itimestep= 0
    n_iceCat = 1


    p3.update(ids, ide, jds, jde, kds, kde,
                ims, ime, jms, jme, kms, kme,
                its, ite, jts, jte, kts, kte,
                th_3d, qv_3d, qc_3d, qr_3d,
                qnr_3d, diag_zdbz_3d, diag_effc_3d,
                diag_effi_3d, diag_vmi_3d, diag_di_3d,
                diag_rhopo_3d, th_old_3d, qv_old_3d,
                qi1_3d, qni1_3d, qir1_3d, qib1_3d,
                nc_3d, pii, p, dz, w,
                RAINNC, RAINNCV ,SR, SNOWNC, SNOWNCV,
                dt, itimestep, 1)

    return



if __name__ == '__main__':
    main()

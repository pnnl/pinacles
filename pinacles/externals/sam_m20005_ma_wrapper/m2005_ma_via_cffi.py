from cffi import FFI
import numpy as np
import os
import pathlib


class M2005_MA:
    def __init__(self):
        self.ffi = FFI()

        path = pathlib.Path(__file__).parent.absolute()

        # Grab the shared library we are building a ffi for
        self._lib_m2005_ma = self.ffi.dlopen(os.path.join(path, "lib_m2005_ma.so"))

        # Define the interfaces
        self.ffi.cdef(
            "void c_m2005_ma_init(char dir_path_c[], int dir_path_len, int nCat, \
            double aero_inv_rm1, double aero_sig1, double aero_nanew1, \
            double aero_inv_rm2, double aero_sig2, double aero_nanew2, double nccnst_in);",
            override=True,
        )

        # This function is for m2005_ma
        self.ffi.cdef(
            "void c_m2005_ma_main(int ids, int ide, int jds, int jde, int kds, int kde, \
            int ims, int ime, int jms, int jme, int kms, int kme, \
            int its, int ite, int jts, int jte, int kts, int kte, \
            double th_3d[], double qv_3d[], double qc_3d[], double qr_3d[], \
            double qnr_3d[], double diag_zdbz_3d[], double diag_effc_3d[], \
            double diag_effi_3d[], double diag_vmi_3d[], double diag_di_3d[], \
            double diag_rhopo_3d[], double th_old_3d[], double qv_old_3d[], \
            double qi1_3d[], double qni1_3d[], double qir1_3d[], double qib1_3d[], \
            int n_diag_3d, double diag_3d[], \
            double LIQUID_SEDIMENTATION[], double ICE_SEDIMENTATION[], \
            double nc_3d[], double pii[], double p[], double dz[], double w[], \
            double RAINNC[], double RAINNCV[] ,double SR[], double SNOWNC[],double SNOWNCV[], \
            double dt, int itimestep, int n_iceCat  ); ",
            override=True,
        )

        return

    def init(
        self,
        nCat=1,
        aero_inv_rm1=2.0e7,
        aero_sig1=2.0,
        aero_nanew1=300.0e6,
        aero_inv_rm2=7.6923076e5,
        aero_sig2=2.5,
        aero_nanew2=0,
        nccnst_in=200e6,
    ):

        path = str(pathlib.Path(__file__).parent.absolute())
        path = path.encode("ascii")

        self._lib_m2005_ma.c_m2005_ma_init(
            self.ffi.new("char[]", path),
            len(path),
            nCat,
            aero_inv_rm1,
            aero_sig1,
            aero_nanew1,
            aero_inv_rm2,
            aero_sig2,
            aero_nanew2,
            nccnst_in,
        )

        return

    def update(
        self,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        th_3d,
        s_3d,
        microfield_4d,
        th_old_3d,
        qv_old_3d,
        n_diag_3d,
        diag_3d,
        LIQUID_SEDIMENTATION,
        ICE_SEDIMENTATION,
        Nacc_sct_3d,
        Nait_sct_3d,
        Nait2a_ct_3d,
        Mait2a_ct_3d,
        relhum_3d,
        diag_effc_3d,
        diag_effr_3d,
        diag_effi_3d,
        diag_effs_3d,
        pii,
        p,
        dz,
        w,
        RAINNC,
        RAINNCV,
        SR,
        SNOWNC,
        SNOWNCV,
        dt,
        itimestep,
        n_iceCat,
        nmicrofields,
    ):

        self._lib_m2005_ma.c_m2005_ma_main(
            ids,
            ide,
            jds,
            jde,
            kds,
            kde,
            ims,
            ime,
            jms,
            jme,
            kms,
            kme,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
            self.as_pointer(th_3d),
            self.as_pointer(s_3d),
            self.as_pointer(microfield_4d),
            self.as_pointer(th_old_3d),
            self.as_pointer(qv_old_3d),
            n_diag_3d,
            self.as_pointer(diag_3d),
            self.as_pointer(LIQUID_SEDIMENTATION),
            self.as_pointer(ICE_SEDIMENTATION),
            self.as_pointer(Nacc_sct_3d),
            self.as_pointer(Nait_sct_3d),
            self.as_pointer(Nait2a_sct_3d),
            self.as_pointer(Mait2a_sct_3d),
            self.as_pointer(relhum_3d),
            self.as_pointer(diag_effc_3d),
            self.as_pointer(diag_effr_3d),
            self.as_pointer(diag_effi_3d),
            self.as_pointer(diag_effs_3d),
            self.as_pointer(pii),
            self.as_pointer(p),
            self.as_pointer(dz),
            self.as_pointer(w),
            self.as_pointer(RAINNC),
            self.as_pointer(RAINNCV),
            self.as_pointer(SR),
            self.as_pointer(SNOWNC),
            self.as_pointer(SNOWNCV),
            dt,
            itimestep,
            n_iceCat,
            nmicrofields,
        )

        return

    # function creates cdata variables of a type "double *" from a numpy array
    # additionally checks if the array is contiguous
    def as_pointer(self, numpy_array):
        assert numpy_array.flags[
            "F_CONTIGUOUS"
        ], "array is not contiguous in memory (Fortran order)"
        return self.ffi.cast("double*", numpy_array.ctypes.data)


def main():

    m2005_ma = M2005_MA()
    m2005_ma.init()

    ids = 1
    jds = 1
    kds = 1
    ide = 1
    jde = 1
    kde = 1
    ims = 1
    jms = 1
    kms = 1
    ime = 64
    jme = 64
    kme = 100
    its = 1
    jts = 1
    kts = 1
    ite = ime
    jte = jme
    kte = kme

    shape_2d = (ime, jme)
    shape_3d = (ime, kme, jme)

    th_3d = np.zeros(shape_3d, dtype=np.double, order="F") + 300.0
    s_3d = np.zeros_like(th_3d) + 300.0
    th_old_3d = np.zeros_like(th_3d) + 300.0
    qv_old_3d = np.zeros_like(th_3d) + 1e-5
    LIQUID_SEDIMENTATION = np.zeros_like(th_3d)
    ICE_SEDIMENTATION = np.zeros_like(th_3d)
    Nacc_sct_3d = np.zeros_like(th_3d)
    Nait_sct_3d = np.zeros_like(th_3d)
    Nait2a_sct_3d = np.zeros_like(th_3d)
    Mait2a_sct_3d = np.zeros_like(th_3d)
    relhum_3d = np.zeros_like(th_3d)
    diag_effc_3d = np.zeros_like(th_3d)
    diag_effr_3d = np.zeros_like(th_3d)
    diag_effi_3d = np.zeros_like(th_3d)
    diag_effs_3d = np.zeros_like(th_3d)
    pii = np.zeros_like(th_3d) + 1.0
    p = np.zeros_like(th_3d) + 1e5
    dz = np.zeros_like(th_3d) + 5.0
    w = np.zeros_like(th_3d)

    nmicrofields = 10
    microfield_4d = np.zeros(
        (shape_3d[0], shape_3d[1], shape_3d[2], nmicrofields), order="F", dtype=np.double
    )
    
    n_diag_3d = 5
    diag_3d = np.zeros(
        (shape_3d[0], shape_3d[1], shape_3d[2], n_diag_3d), order="F", dtype=np.double
    )

    RAINNC = np.zeros(shape_2d, dtype=np.double, order="F")
    RAINNCV = np.zeros_like(RAINNC)
    SR = np.zeros_like(RAINNC)
    SNOWNC = np.zeros_like(RAINNC)
    SNOWNCV = np.zeros_like(RAINNC)

    dt = 5.0
    itimestep = 0
    n_iceCat = 1

    m2005_ma.update(
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        th_3d,
        s_3d,
        microfield_4d,
        th_old_3d,
        qv_old_3d,
        n_diag_3d,
        diag_3d,
        LIQUID_SEDIMENTATION,
        ICE_SEDIMENTATION,
        Nacc_sct_3d,
        Nait_sct_3d,
        Nait2a_ct_3d,
        Mait2a_ct_3d,
        relhum_3d,
        diag_effc_3d,
        diag_effr_3d,
        diag_effi_3d,
        diag_effs_3d,
        pii,
        p,
        dz,
        w,
        RAINNC,
        RAINNCV,
        SR,
        SNOWNC,
        SNOWNCV,
        dt,
        itimestep,
        1,
        nmicrofields,
    )

    return


if __name__ == "__main__":
    main()

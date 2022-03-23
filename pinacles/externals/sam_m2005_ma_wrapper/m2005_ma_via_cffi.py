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
        nx,
        ny,
        nz,
        nmicrofields,
        nx_gl,
        ny_gl,
        my_rank,
        nsubdomains_x,
        nsubdomains_y,
        th_3d,
        s_3d,
        w,
        microfield_4d,
        n_diag_3d,
        diag_3d,
        z,
        p0,
        rho0,
        tabs0,
        zi,
        rhow,
        dx,
        dz,
        nrainy,
        nrmn,
        ncmn,
        total_water_prec,
        tlat,
        tlatqi,
        precflux,
        qpfall,
        fluxbq,
        fluxtq,
        u10arr,
        precsfc,
        prec_xy,
        dt,
        time,
        itimestep,
        LCOND,
        LSUB,
        CPD,
        RGAS,
        RV,
        G,
    ):

        self._lib_m2005_ma.c_m2005_ma_main(
            nx,
            ny,
            nz,
            nmicrofields,
            nx_gl,
            ny_gl,
            my_rank,
            nsubdomains_x,
            nsubdomains_y,
            self.as_pointer(th_3d),
            self.as_pointer(s_3d),
            self.as_pointer(w),
            self.as_pointer(microfield_4d),
            n_diag_3d,
            self.as_pointer(diag_3d),
            self.as_pointer(z),
            self.as_pointer(p0),
            self.as_pointer(rho0),
            self.as_pointer(tabs0),
            self.as_pointer(zi),
            self.as_pointer(rhow),
            dx,
            dz,
            self.as_pointer(nrainy),
            self.as_pointer(nrmn),
            self.as_pointer(ncmn),
            self.as_pointer(total_water_prec),
            self.as_pointer(tlat),
            self.as_pointer(tlatqi),
            self.as_pointer(precflux),
            self.as_pointer(qpfall),
            self.as_pointer(fluxbq),
            self.as_pointer(fluxtq),
            self.as_pointer(u10arr),
            self.as_pointer(precsfc),
            self.as_pointer(prec_xy),
            dt,
            time,
            itimestep,
            LCOND,
            LSUB,
            CPD,
            RGAS,
            RV,
            G,
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

    nx = 64
    ny = 64
    nz = 100
    
    shape_2d = (nx, ny)
    shape_3d = (nx, ny, nz)

    th_3d = np.zeros(shape_3d, dtype=np.double, order="F") + 300.0
    s_3d = np.zeros_like(th_3d) + 300.0
    w = np.zeros_like(th_3d)

    nmicrofields = 10
    microfield_4d = np.zeros(
        (shape_3d[0], shape_3d[1], shape_3d[2], nmicrofields), order="F", dtype=np.double
    )
    
    n_diag_3d = 5
    diag_3d = np.zeros(
        (shape_3d[0], shape_3d[1], shape_3d[2], n_diag_3d), order="F", dtype=np.double
    )

    fluxbq = np.zeros(shape_2d, dtype=np.double, order="F")
    fluxtq = np.zeros_like(fluxbq)
    u10arr = np.zeros_like(fluxbq)
    precsfc = np.zeros_like(fluxbq)
    prec_xy = np.zeros_like(fluxbq)

    dt = 5.0
    itimestep = 0
    n_iceCat = 1

    m2005_ma.update(
        nx,
        ny,
        nz,
        nmicrofields,
        nx_gl,
        ny_gl,
        my_rank,
        nsubdomains_x,
        nsubdomains_y,
        th_3d,
        s_3d,
        w,
        microfield_4d,
        n_diag_3d,
        diag_3d,
        z,
        p0,
        rho0,
        tabs0,
        zi,
        rhow,
        dx,
        dz,
        nrainy,
        nrmn,
        ncmn,
        total_water_prec,
        tlat,
        tlatqi,
        precflux,
        qpfall,
        fluxbq,
        fluxtq,
        u10arr,
        precsfc,
        prec_xy,
        dt,
        time,
        itimestep,
        LCOND,
        LSUB,
        CPD,
        RGAS,
        RV,
        G,
    )

    return


if __name__ == "__main__":
    main()

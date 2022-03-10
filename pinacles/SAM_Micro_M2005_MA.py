from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
    water_path_lasso,
)
from pinacles.externals.sam_m2005_ma_wrapper import m2005_ma_via_cffi
from pinacles import UtilitiesParallel
from pinacles.WRFUtil import (
    to_wrf_order,
    wrf_tend_to_our_tend,
    wrf_theta_tend_to_our_tend,
    to_our_order,
)
from pinacles import parameters
from mpi4py import MPI
import numba
import numpy as np


@numba.njit(fastmath=True)
def compute_w_from_q(rho0, rhod, p, qv, T, wv, wc, wr, wnc, wi1, wni1, wir1, wib1):

    shape = qv.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                qt = wv[i,j,k] + wc[i,j,k] + wr[i,j,k]
                pd  = p[i,j,k] * (1.0 - qt) / ( 1.0 - qt  + parameters.EPSVI * wv[i,j,k])
                
                rhod[i,j,k]= pd/(parameters.RD * T[i,j,k])
                
                factor = (rho0[i,j,k]) / rhod[i,j,k]

                wv[i, j, k] *= factor
                wc[i, j, k] *= factor
                wr[i, j, k] *= factor
                wnc[i, j, k] *= factor
                wi1[i, j, k] *= factor
                wni1[i, j, k] *= factor
                wir1[i, j, k] *= factor
                wib1[i, j, k] *= factor

    return


@numba.njit(fastmath=True)
def compute_q_from_w(rho0, rhod, qv, qc, qr, qnc, qi1, qni1, qir1, qib1):
    shape = qv.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):

                factor = rhod[i,j,k] / (rho0[i,j,k])

                qv[i, j, k] *= factor
                qc[i, j, k] *= factor
                qr[i, j, k] *= factor
                qnc[i, j, k] *= factor
                qi1[i, j, k] *= factor
                qni1[i, j, k] *= factor
                qir1[i, j, k] *= factor
                qib1[i, j, k] *= factor

    return


class MicroM2005_MA(MicrophysicsBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
    ):

        MicrophysicsBase.__init__(
            self,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )

        self._m2005_ma_cffi = m2005_ma_via_cffi.M2005_MA()
        
        try:
            aero_in = namelist["microphysics"]["aero"]
            inv_rm1 = aero_in["inv_rm1"]
            sig1 = aero_in["sig1"]
            nanew1 = aero_in["nanew1"]

            inv_rm2 = aero_in["inv_rm2"]
            sig2 = aero_in["sig2"]
            nanew2 = aero_in["nanew2"]

            self._m2005_ma_cffi.init(
                aero_inv_rm1=inv_rm1,
                aero_sig1=sig1,
                aero_nanew1=nanew1,
                aero_inv_rm2=inv_rm2,
                aero_sig2=sig2,
                aero_nanew2=nanew2,
            )

            if MPI.COMM_WORLD.Get_rank() == 0:
                print("\tM2005_MA: Initialized with custom aerosol distn")
                print(inv_rm1, sig1, nanew1)
                print(inv_rm2, sig2, nanew2)
        except:
            self._m2005_ma_cffi.init()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("\tM2005_MA: Initialized with default aerosol distn")

        # Allocate microphysical/thermodynamic variables
        self._ScalarState.add_variable(
            "qv",
            long_name="water vapor mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_v",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qc",
            long_name="cloud water mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_c",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qnc",
            long_name="cloud number concentration",
            units="# kg^{-1}",
            latex_name="q_{nc}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qad",
            long_name="dry aerosol mass mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{ad}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qad2",
            long_name="aitken mode mass mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{ad2}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qnad",
            long_name="dry aerosol number concentration",
            units="# kg^{-1}",
            latex_name="q_{nad}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qnad2",
            long_name="aitken mode mass number concentration",
            units="# kg^{-1}",
            latex_name="q_{nad2}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qaw",
            long_name="wet aerosol mass mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{aw}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qr",
            long_name="rain water mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{r}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qnr",
            long_name="rain number concentration",
            units="# kg^{-1}",
            latex_name="q_{nr}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qar",
            long_name="wet aerosol mass mixing ratio in rain",
            units="kg kg^{-1}",
            latex_name="q_{ar}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qci",
            long_name="cloud ice mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{i}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qnci",
            long_name="cloud ice number concentration",
            units="# kg^{-1}",
            latex_name="q_{nci}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qs",
            long_name="snow mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{s}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qns",
            long_name="snow number concentration",
            units="# kg^{-1}",
            latex_name="q_{ns}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qg",
            long_name="graupel mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{g}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qng",
            long_name="graupel number concentration",
            units="# kg^{-1}",
            latex_name="q_{ng}",
            limit=True,
        )

        self._DiagnosticState.add_variable(
            "qtot_sed",
            long_name="total liquid water sedimentation",
            units="",
        )
        self._DiagnosticState.add_variable(
            "qice_sed",
            long_name="ice sedimentation",
            units="",
        )
        self._DiagnosticState.add_variable(
            "Nacc_sct",
            long_name="Nacc self coagulation tendency",
            units="",
        )
        self._DiagnosticState.add_variable(
            "Nait_sct",
            long_name="Nait self coagulation tendency",
            units="",
        )
        self._DiagnosticState.add_variable(
            "Nait2a_ct",
            long_name="Nait2acc coagulation tendency",
            units="",
        )
        self._DiagnosticState.add_variable(
            "Mait2a_ct",
            long_name="Mait2acc coagulation tendency",
            units="",
        )
        self._DiagnosticState.add_variable(
            "relhum",
            long_name="relative humidity",
            units="",
            latex_name="relhum",
        )
        self._DiagnosticState.add_variable(
            "diag_effc_3d",
            long_name="cloud droplet effective radius",
            units="m",
            latex_name="r_{e,c}",
        )
        self._DiagnosticState.add_variable(
            "diag_effr_3d",
            long_name="rain droplet effective radius",
            units="m",
            latex_name="r_{e,r}",
        )
        self._DiagnosticState.add_variable(
            "diag_effi_3d",
            long_name="cloud ice effective radius",
            units="m",
            latex_name="r_{e,i}",
        )
        self._DiagnosticState.add_variable(
            "diag_effs_3d",
            long_name="snow effective radius",
            units="m",
            latex_name="r_{e,s}",
        )

        nhalo = self._Grid.n_halo
        self._our_dims = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        self._sam_dims = (
            self._our_dims[0] - 2 * nhalo[0],
            self._our_dims[1] - 2 * nhalo[1],
            self._our_dims[2] - 2 * nhalo[2],
        )

        self._itimestep = 0

        # For diagnostic variables

        self._diag_3d_vars = [
            "dBZCLRAD",
            "NARG1",
            "NARG2",
            "NACTRATE",
            "QACTRATE",
            "NACTDIFF",
            "NATRANS",
            "QATRANS",
            "ISACT",
            "DC1",
            "DC2",
            "DG1",
            "DG2",
            "SSPK",
            "NCPOSLM",
            "NCNEGLM",
        ]

        qc_units = "kg kg-1"
        qc_tend_units = "kg kg-1 s-1"
        n_tend_units = "kg-1"
        n_rate_units = "kg-1 s-1"
        diag_3d_long_names = {
            "dBZCLRAD": ("Cloud Radar Reflectivity", "dBZ"),
            "NARG1": ("A-R&G. Activated Number Accumulation", n_tend_units),
            "NARG2": ("A-R&G Activated Number Aitken", n_tend_units),
            "NACTRATE": ("Activation Rate Accumulation Aerosol", n_rate_units),
            "QACTRATE": ("Activation Rate Accumulation Aerosol", qc_tend_units),
            "NACTDIFF": ("Difference from A-R&G activ number and and old cloud number", n_tend_units),
            "NATRANS": ("Aitken to Accum Transfer Number", n_tend_units),
            "QATRANS": ("Aitken to Accum Transfer Mass", qc_units),
            "ISACT": ("ARG Activation run on particular point","None"),
            "DC1": ("Critical activation diameter mode 1", "m"),
            "DC2": ("Critical activation diameter mode 2", "m"),
            "DG1": ("Modal diameter 1", "m"),
            "DG2": ("Modal diameter 2", "m"),
            "SSPK": ("Peak activation supersaturation ARG", "fraction"),
            "NCPOSLM": ("Change in NC due to positive rate limiter", n_rate_units),
            "NCNEGLM": ("Mass Activation Rate Aitken Aerosol", n_rate_units),
        }

        self._n_diag_3d = len(self._diag_3d_vars)

        self._diag_3d = np.empty(
            (self._sam_dims[0], self._sam_dims[1], self._sam_dims[2], self._n_diag_3d),
            dtype=np.double,
            order="F",
        )

        # Add a diagnostic variable for each of the process rates
        for i, vn in enumerate(self._diag_3d_vars):
            self._DiagnosticState.add_variable(
                "m2005_ma_" + vn,
                long_name=diag_3d_long_names[vn][0],
                units=diag_3d_long_names[vn][1],
            )

        # Allocate sam arrays
        for v in [
            "s_sam",
            "rho_sam",
            "rhod_sam",
            "exner_sam",
            "p0_sam",
            "T_sam",
            "qv_sam",
            "qc_sam",
            "qr_sam",
            "qnr_sam",
            "nc_sam",
            "qi1_sam",
            "qni1_sam",
            "qir1_sam",
            "qib1_sam",
            "w_sam",
            "th_old",
            "qv_old",
            "ice_sed_sam",
            "liq_sed_sam",
            "reflectivity_sam",
            "diag_effc_3d_sam",
            "diag_effi_3d_sam",
            "diag_vmi",
            "diag_di",
            "diag_rhopo",
            "dz_sam",
            "z",
        ]:
            setattr(
                self,
                "_" + v,
                np.empty(tuple(self._sam_dims), order="F", dtype=np.double),
            )

        self._Timers.add_timer("MicroM2005_MA_update")
        return

    def update(self):
        self._Timers.start_timer("MicroM2005_MA_update")

        # Get variables from the model state
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")
        qc = self._ScalarState.get_field("qc")
        qnc = self._ScalarState.get_field("qnc")
        qad = self._ScalarState.get_field("qad")
        qad2 = self._ScalarState.get_field("qad2")
        qnad = self._ScalarState.get_field("qnad")
        qnad2 = self._ScalarState.get_field("qnad2")
        qaw = self._ScalarState.get_field("qaw")
        qr = self._ScalarState.get_field("qr")
        qnr = self._ScalarState.get_field("qnr")
        qar = self._ScalarState.get_field("qar")
        qi = self._ScalarState.get_field("qi")
        qnci = self._ScalarState.get_field("qnci")
        qs = self._ScalarState.get_field("qs")
        qns = self._ScalarState.get_field("qns")
        qg = self._ScalarState.get_field("qg")
        qng = self._ScalarState.get_field("qng")

        w = self._VelocityState.get_field("w")

        qtot_sed = self._DiagnosticState.get_field("qtot_sed")
        qice_sed = self._DiagnosticState.get_field("qice_sed")
        Nacc_sct = self._DiagnosticState.get_field("Nacc_sct")
        Nait_sct = self._DiagnosticState.get_field("Nait_sct")
        Nait2a_ct = self._DiagnosticState.get_field("Nait2a_ct")
        Mait2a_ct = self._DiagnosticState.get_field("Mait2a_ct")
        relhum = self._DiagnosticState.get_field("relhum")

        diag_effc_3d = self._DiagnosticState.get_field("diag_effc_3d")
        diag_effr_3d = self._DiagnosticState.get_field("diag_effr_3d")
        diag_effi_3d = self._DiagnosticState.get_field("diag_effi_3d")
        diag_effs_3d = self._DiagnosticState.get_field("diag_effs_3d")

        exner = self._Ref.exner
        p0 = self._Ref.p0
        nhalo = self._Grid.n_halo

        s_sam = self._s_sam
        rho_sam = self._rho_sam
        rhod_sam = self._rhod_sam
        exner_sam = self._exner_sam
        p0_sam = self._p0_sam
        T_sam = self._T_sam
        qv_sam = self._qv_sam
        qc_sam = self._qc_sam
        qr_sam = self._qr_sam
        qnr_sam = self._qnr_sam
        nc_sam = self._nc_sam
        qi1_sam = self._qi1_sam
        qni1_sam = self._qni1_sam
        qir1_sam = self._qir1_sam
        qib1_sam = self._qib1_sam
        w_sam = self._w_sam
        th_old = self._th_old
        qv_old = self._qv_old

        ice_sed_sam = self._ice_sed_sam
        liq_sed_sam = self._liq_sed_sam

        reflectivity_sam = self._reflectivity_sam
        diag_effc_3d_sam = self._diag_effc_3d_sam
        diag_effi_3d_sam = self._diag_effi_3d_sam
        diag_vmi = self._diag_vmi
        diag_di = self._diag_di
        diag_rhopo = self._diag_rhopo

        dz_sam = self._dz_sam
        z = self._z

        dz_sam.fill(self._Grid.dx[2])
        z[:, :, :] = self._Grid.z_global[np.newaxis, np.newaxis, nhalo[2] : -nhalo[2]]
        rho_sam[:, :, :] = self._Ref.rho0[np.newaxis, np.newaxis, nhalo[2] : -nhalo[2]]
        exner_sam[:, :, :] = exner[np.newaxis, np.newaxis, nhalo[2] : -nhalo[2]]
        p0_sam[:, :, :] = p0[np.newaxis, np.newaxis, nhalo[2] : -nhalo[2]]

        dt = self._TimeSteppingController.dt

        ids = 1
        jds = 1
        kds = 1
        ide = 1
        jde = 1
        kde = 1
        ims = 1
        jms = 1
        kms = 1
        ime = self._sam_dims[0]
        jme = self._sam_dims[1]
        kme = self._sam_dims[2]
        its = 1
        jts = 1
        kts = 1
        ite = ime
        jte = jme
        kte = kme

        np.copyto(th_old, T_sam, casting="no")
        np.copyto(qv_old, qv_sam, casting="no")

        # Do conversions between specific humidity and mixing ratio
        compute_w_from_q(
            rho_sam,
            rhod_sam, 
            p0_sam,
            qv_sam,
            T_sam,
            qv_sam,
            qc_sam,
            qr_sam,
            nc_sam,
            qi1_sam,
            qni1_sam,
            qir1_sam,
            qib1_sam,
        )

        n_iceCat = 1


        self._m2005_ma_cffi.update(
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
            T_sam,
            qv_sam,
            qc_sam,
            qr_sam,
            qnr_sam,
            reflectivity_sam,
            diag_effc_3d_sam,
            diag_effi_3d_sam,
            diag_vmi,
            diag_di,
            diag_rhopo,
            th_old,
            qv_old,
            qi1_sam,
            qni1_sam,
            qir1_sam,
            qib1_sam,
            self._n_diag_3d,
            self._diag_3d,
            liq_sed_sam,
            ice_sed_sam,
            nc_sam,
            exner_sam,
            p0_sam,
            dz_sam,
            w_sam,
            self._RAINNC,
            self._RAINNCV,
            self._SR,
            self._SNOWNC,
            self._SNOWNCV,
            dt,
            self._itimestep,
            n_iceCat,
        )


        # Do conversions between specific humidity and mixing ratio
        compute_q_from_w(
            rho_sam,
            rhod_sam,
            qv_sam,
            qc_sam,
            qr_sam,
            nc_sam,
            qi1_sam,
            qni1_sam,
            qir1_sam,
            qib1_sam,
        )


        # Update the energys (TODO Move this to numba)
        # T_sam *= self._Ref.exner[np.newaxis, np.newaxis, nhalo[2] : -nhalo[2]]
        self._update_static_energy(z, T_sam, s_sam, qc_sam, qr_sam, qi1_sam)

        # s_sam = (
        #    T_sam
        #    + (
        #        parameters.G * z
        #        - parameters.LV * (qc_sam + qr_sam)
        #        - parameters.LS * (qi1_sam)
        #    )
        #    * parameters.ICPD
        # )

        # Compute and apply sedimentation sources of static energy
        np.multiply(liq_sed, parameters.LV / parameters.CPD, out=s_tend_liq_sed)
        np.multiply(ice_sed, parameters.LS / parameters.CPD, out=s_tend_ice_sed)

        # Convert sedimentation sources to units of tendency
        np.multiply(liq_sed, 1.0 / self._TimeSteppingController.dt, out=liq_sed)
        np.multiply(ice_sed, 1.0 / self._TimeSteppingController.dt, out=ice_sed)
        np.multiply(
            s_tend_liq_sed, -1.0 / self._TimeSteppingController.dt, out=s_tend_liq_sed
        )
        np.multiply(
            s_tend_ice_sed, -1.0 / self._TimeSteppingController.dt, out=s_tend_ice_sed
        )

        self._itimestep += 1

        self._Timers.end_timer("MicroM2005_MA_update")
        return

    def io_initialize(self, nc_grp):
        timeseries_grp = nc_grp["timeseries"]
        profiles_grp = nc_grp["profiles"]

        v = timeseries_grp.createVariable("CF", np.double, dimensions=("time",))
        v.long_name = "Cloud Fraction"
        v.standard_name = "CF"
        v.units = ""

        v = timeseries_grp.createVariable("RF", np.double, dimensions=("time",))
        v.long_name = "Rain Fraction"
        v.standard_name = "RF"
        v.units = ""

        v = timeseries_grp.createVariable("LWP", np.double, dimensions=("time",))
        v.long_name = "Liquid Water Path"
        v.standard_name = "LWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("LWP_LASSO", np.double, dimensions=("time",))
        v.long_name = "LASSO Liquid Water Path"
        v.standard_name = "LWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("RWP", np.double, dimensions=("time",))
        v.long_name = "Rain Water Path"
        v.standard_name = "RWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("VWP", np.double, dimensions=("time",))
        v.long_name = "Water Vapor Path"
        v.standard_name = "VWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("RAINNC", np.double, dimensions=("time",))
        v.long_name = "accumulated surface precip"
        v.units = "mm"
        v.latex_name = "rainnc"

        v = timeseries_grp.createVariable("RAINNCV", np.double, dimensions=("time",))
        v.long_name = "one time step accumulated surface precip"
        v.units = "mm"
        v.latex_name = "rainncv"

        # Now add cloud fraction and rain fraction profiles
        v = profiles_grp.createVariable(
            "CF",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        v.long_name = "Cloud Fraction"
        v.standard_name = "CF"
        v.units = ""

        v = profiles_grp.createVariable(
            "RF",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        v.long_name = "Rain Fraction"
        v.standard_name = "RF"
        v.units = ""

        return

    def io_update(self, nc_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()

        n_halo = self._Grid.n_halo
        dz = self._Grid.dx[2]
        rho = self._Ref.rho0
        npts = self._Grid.n[0] * self._Grid.n[1]

        qc = self._ScalarState.get_field("qc")
        qv = self._ScalarState.get_field("qv")
        qr = self._ScalarState.get_field("qr")

        # First compute liqud water path
        lwp = water_path(n_halo, dz, npts, rho, qc)
        lwp = UtilitiesParallel.ScalarAllReduce(lwp)

        # First compute liqud water path
        lwp_lasso, npts_lasso = water_path_lasso(n_halo, dz, rho, qc + qr)
        lwp_lasso = UtilitiesParallel.ScalarAllReduce(lwp_lasso)
        npts_lasso = UtilitiesParallel.ScalarAllReduce(npts_lasso)
        if npts_lasso > 0:
            lwp_lasso /= npts_lasso

        rwp = water_path(n_halo, dz, npts, rho, qr)
        rwp = UtilitiesParallel.ScalarAllReduce(rwp)

        vwp = water_path(n_halo, dz, npts, rho, qv)
        vwp = UtilitiesParallel.ScalarAllReduce(vwp)

        # Compute cloud and rain fraction
        cf = water_fraction(n_halo, npts, qc, threshold=1e-5)
        cf = UtilitiesParallel.ScalarAllReduce(cf)

        cf_prof = water_fraction_profile(n_halo, npts, qc, threshold=1e-5)
        cf_prof = UtilitiesParallel.ScalarAllReduce(cf_prof)

        rf = water_fraction(n_halo, npts, qr)
        rf = UtilitiesParallel.ScalarAllReduce(rf)

        rainnc = np.sum(self._RAINNC) / npts
        rainnc = UtilitiesParallel.ScalarAllReduce(rainnc)

        rf_prof = water_fraction_profile(n_halo, npts, qr, threshold=1e-5)
        rf_prof = UtilitiesParallel.ScalarAllReduce(rf_prof)

        rainncv = np.sum(self._RAINNCV) / npts
        rainncv = UtilitiesParallel.ScalarAllReduce(rainncv)

        if my_rank == 0:
            timeseries_grp = nc_grp["timeseries"]
            profiles_grp = nc_grp["profiles"]

            timeseries_grp["CF"][-1] = cf
            timeseries_grp["RF"][-1] = rf
            timeseries_grp["LWP"][-1] = lwp
            timeseries_grp["LWP_LASSO"][-1] = lwp_lasso
            timeseries_grp["RWP"][-1] = rwp
            timeseries_grp["VWP"][-1] = vwp

            timeseries_grp["RAINNC"][-1] = rainnc
            timeseries_grp["RAINNCV"][-1] = rainncv

            profiles_grp["CF"][-1, :] = cf_prof[n_halo[2] : -n_halo[2]]
            profiles_grp["RF"][-1, :] = rf_prof[n_halo[2] : -n_halo[2]]

        return

    @staticmethod
    @numba.njit
    def _update_static_energy(z, T_sam, s_sam, qc_sam, qr_sam, qi1_sam):
        shape = T_sam.shape
        # Fortran ordered so we need to loop differently
        for k in range(shape[2]):
            for j in range(shape[1]):
                for i in range(shape[0]):
                    _T = T_sam[i, j, k]
                    T_sam[i, j, k] = _T
                    s_sam[i, j, k] = (
                        _T
                        + (
                            parameters.G * z[i, j, k]
                            - parameters.LV * (qc_sam[i, j, k] + qr_sam[i, j, k])
                            - parameters.LS * (qi1_sam[i, j, k])
                        )
                        * parameters.ICPD
                    )

        return

    def io_fields2d_update(self, nc_grp):

        start = self._Grid.local_start
        end = self._Grid._local_end
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        if nc_grp is not None:
            rainnc = nc_grp.createVariable(
                "RAINNC",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        send_buffer[start[0] : end[0], start[1] : end[1]] = self._RAINNC
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

        if nc_grp is not None:
            rainnc[:, :] = recv_buffer

        if nc_grp is not None:
            rainncv = nc_grp.createVariable(
                "RAINNCV",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = self._RAINNCV
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            rainncv[:, :] = recv_buffer

        # Compute and output the LWP
        if nc_grp is not None:
            lwp = nc_grp.createVariable(
                "LWP",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )
        nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        qc = self._ScalarState.get_field("qc")[nh[0] : -nh[0], nh[1] : -nh[1], :]
        lwp_compute = np.sum(
            qc * rho0[np.newaxis, np.newaxis, 0] * self._Grid.dx[2], axis=2
        )

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = lwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            lwp[:, :] = recv_buffer

        # Compute and output the LWP
        if nc_grp is not None:
            iwp = nc_grp.createVariable(
                "IWP",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        qc = self._ScalarState.get_field("qi1")[nh[0] : -nh[0], nh[1] : -nh[1], :]
        iwp_compute = np.sum(qc * rho0[np.newaxis, np.newaxis, 0], axis=2)

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = iwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            iwp[:, :] = recv_buffer

        if nc_grp is not None:
            nc_grp.sync()

        return

    def get_qc(self):
        return self._ScalarState.get_field("qc") + self._ScalarState.get_field("qr")

    def get_qcloud(self):
        return self._ScalarState.get_field("qc")

    def get_qi(self):
        return self._ScalarState.get_field("qi1")

    def get_reffc(self):
        return self._DiagnosticState.get_field("diag_effc_3d")

    def get_reffi(self):
        return self._DiagnosticState.get_field("diag_effi_3d")

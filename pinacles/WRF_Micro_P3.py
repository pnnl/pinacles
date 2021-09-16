from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
)
from pinacles.externals.wrf_p3_wrapper import p3_via_cffi
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


class MicroP3(MicrophysicsBase):
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

        # Determine which rain microphysics scheme to use
        self._rain_moment = 1
        try:
            self._rain_moment = namelist["microphysics"]["rain_moment"]
        except:
            pass
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\tP3: Using the " + str(self._rain_moment) + "-moment rain scheme")

        # Set the default droplet conencentration for 1-moment rain scheme.
        self._nccnst = 200.0e6
        try:
            self._nccnst = namelist["microphysics"]["nccnst"]
        except:
            pass
        if MPI.COMM_WORLD.Get_rank() == 0 and self._rain_moment == 1:
            print(
                "\t\tP3: Using fixed cloud droplet concentration of: ",
                str(self._nccnst),
                "m-3.",
            )

        self._p3_cffi = p3_via_cffi.P3()
        if self._rain_moment == 1:
            self._p3_cffi.init(nccnst_in=self._nccnst)
        else:
            try:
                aero_in = namelist["microphysics"]["aero"]
                inv_rm1 = aero_in["inv_rm1"]
                sig1 = aero_in["sig1"]
                nanew1 = aero_in["nanew1"]

                inv_rm2 = aero_in["inv_rm2"]
                sig2 = aero_in["sig2"]
                nanew2 = aero_in["nanew2"]

                self._p3_cffi.init(
                    aero_inv_rm1=inv_rm1,
                    aero_sig1=sig1,
                    aero_nanew1=nanew1,
                    aero_inv_rm2=inv_rm2,
                    aero_sig2=sig2,
                    aero_nanew2=nanew2,
                )

                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("\tP3: Initialized with custom aerosol distn")
                    print(inv_rm1, sig1, nanew1)
                    print(inv_rm2, sig2, nanew2)
            except:
                self._p3_cffi.init()
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("\tP3: Initialized with default aerosol distn")

        # Allocate microphysical/thermodyamic variables
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
            "qi1",
            long_name="total ice mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{i}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qni1",
            long_name="ice number concentration",
            units="# kg^{-1}",
            latex_name="q_{ni}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qir1",
            long_name="rime ice mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{ir}",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qib1",
            long_name="ice rime volume mixing ratio",
            units="m^{-3} kg^{-1}",
            latex_name="q_{ib}",
            limit=True,
        )

        self._DiagnosticState.add_variable("liq_sed")
        self._DiagnosticState.add_variable(
            "s_tend_liq_sed", long_name="s tend liquid water sedimentation", units=""
        )
        self._DiagnosticState.add_variable(
            "s_tend_ice_sed", long_name="s tend ice water sedimentation", units=""
        )
        self._DiagnosticState.add_variable("ice_sed")
        self._DiagnosticState.add_variable(
            "reflectivity",
            long_name="radar reflectivity",
            units="dBz",
            latex_name="reflectivity",
        )
        self._DiagnosticState.add_variable(
            "diag_effc_3d",
            long_name="cloud droplet effective radius",
            units="m",
            latex_name="r_e",
        )
        self._DiagnosticState.add_variable(
            "diag_effi_3d",
            long_name="cloud ice effective radius",
            units="m",
            latex_name="r_{e,i}",
        )

        nhalo = self._Grid.n_halo
        self._our_dims = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        self._wrf_dims = (
            self._our_dims[0] - 2 * nhalo[0],
            self._our_dims[2] - 2 * nhalo[2],
            self._our_dims[1] - 2 * nhalo[1],
        )

        self._itimestep = 0
        self._RAINNC = np.zeros(
            (self._wrf_dims[0], self._wrf_dims[2]), order="F", dtype=np.double
        )
        self._SR = np.zeros_like(self._RAINNC)
        self._RAINNCV = np.zeros_like(self._RAINNC)
        self._SNOWNC = np.zeros_like(self._RAINNC)
        self._SNOWNCV = np.zeros_like(self._RAINNC)

        # For diagnostic variables

        self._diag_3d_vars = [
            "qcacc",
            "qrevp",
            "qccon",
            "qcaut",
            "qcevp",
            "qrcon",
            "ncacc",
            "ncnuc",
            "ncslf",
            "ncautc",
            "qcnuc",
            "nrslf",
            "nrevp",
            "ncautr",
        ]

        qc_tend_units = "kg kg-1 s-1"
        n_tend_units = "kg-1"
        diag_3d_long_names = {
            "qcacc": ("cloud droplet accretion by rain", qc_tend_units),
            "qrevp": ("rain evaporation", qc_tend_units),
            "qccon": ("cloud droplet condensation", qc_tend_units),
            "qcaut": ("cloud droplet autoconversion to rain", qc_tend_units),
            "qcevp": ("cloud droplet evaporation", qc_tend_units),
            "qrcon": ("rain condensation", qc_tend_units),
            "ncacc": (
                "change in cloud droplet number from accretion by rain",
                n_tend_units,
            ),
            "ncnuc": (
                "change in cloud droplet number from activation of CCN",
                n_tend_units,
            ),
            "ncslf": (
                "change in cloud droplet number from self-collection",
                n_tend_units,
            ),
            "ncautc": (
                "change in cloud droplet number from autoconversion",
                n_tend_units,
            ),
            "qcnuc": ("activation of cloud droplets from CCN", qc_tend_units),
            "nrslf": ("change in rain number from self-collection", n_tend_units),
            "nrevp": ("change in rain number from evaporation", n_tend_units),
            "ncautr": (
                "change in rain number from autoconversion of cloud water",
                n_tend_units,
            ),
        }

        self._n_diag_3d = len(self._diag_3d_vars)

        self._diag_3d = np.empty(
            (self._wrf_dims[0], self._wrf_dims[1], self._wrf_dims[2], self._n_diag_3d),
            dtype=np.double,
            order="F",
        )

        # Add a diagnostic variable for each of the process rates
        for i, vn in enumerate(self._diag_3d_vars):
            self._DiagnosticState.add_variable(
                "p3_" + vn,
                long_name=diag_3d_long_names[vn][0],
                units=diag_3d_long_names[vn][1],
            )

        # Allocate wrf arrays
        for v in [
            "s_wrf",
            "rho_wrf",
            "exner_wrf",
            "p0_wrf",
            "T_wrf",
            "qv_wrf",
            "qc_wrf",
            "qr_wrf",
            "qnr_wrf",
            "nc_wrf",
            "qi1_wrf",
            "qni1_wrf",
            "qir1_wrf",
            "qib1_wrf",
            "w_wrf",
            "th_old",
            "qv_old",
            "ice_sed_wrf",
            "liq_sed_wrf",
            "reflectivity_wrf",
            "diag_effc_3d_wrf",
            "diag_effi_3d_wrf",
            "diag_vmi",
            "diag_di",
            "diag_rhopo",
            "dz_wrf",
            "z",
        ]:
            setattr(
                self,
                "_" + v,
                np.empty(tuple(self._wrf_dims), order="F", dtype=np.double),
            )

        self._Timers.add_timer("MicroP3_update")
        return

    def update(self):
        self._Timers.start_timer("MicroP3_update")

        # Get variables from the model state
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")
        qc = self._ScalarState.get_field("qc")
        qnc = self._ScalarState.get_field("qnc")
        qr = self._ScalarState.get_field("qr")

        qnr = self._ScalarState.get_field("qnr")
        qi1 = self._ScalarState.get_field("qi1")
        qni1 = self._ScalarState.get_field("qni1")
        qir1 = self._ScalarState.get_field("qir1")
        qib1 = self._ScalarState.get_field("qib1")

        w = self._VelocityState.get_field("w")

        reflectivity = self._DiagnosticState.get_field("reflectivity")
        diag_effc_3d = self._DiagnosticState.get_field("diag_effc_3d")
        diag_effi_3d = self._DiagnosticState.get_field("diag_effi_3d")

        liq_sed = self._DiagnosticState.get_field("liq_sed")
        ice_sed = self._DiagnosticState.get_field("ice_sed")
        s_tend_liq_sed = self._DiagnosticState.get_field("s_tend_liq_sed")
        s_tend_ice_sed = self._DiagnosticState.get_field("s_tend_ice_sed")

        exner = self._Ref.exner
        p0 = self._Ref.p0
        nhalo = self._Grid.n_halo

        s_wrf = self._s_wrf
        rho_wrf = self._rho_wrf
        exner_wrf = self._exner_wrf
        p0_wrf = self._p0_wrf
        T_wrf = self._T_wrf
        qv_wrf = self._qv_wrf
        qc_wrf = self._qc_wrf
        qr_wrf = self._qr_wrf
        qnr_wrf = self._qnr_wrf
        nc_wrf = self._nc_wrf
        qi1_wrf = self._qi1_wrf
        qni1_wrf = self._qni1_wrf
        qir1_wrf = self._qir1_wrf
        qib1_wrf = self._qib1_wrf
        w_wrf = self._w_wrf
        th_old = self._th_old
        qv_old = self._qv_old

        ice_sed_wrf = self._ice_sed_wrf
        liq_sed_wrf = self._liq_sed_wrf

        reflectivity_wrf = self._reflectivity_wrf
        diag_effc_3d_wrf = self._diag_effc_3d_wrf
        diag_effi_3d_wrf = self._diag_effi_3d_wrf
        diag_vmi = self._diag_vmi
        diag_di = self._diag_di
        diag_rhopo = self._diag_rhopo

        dz_wrf = self._dz_wrf
        z = self._z

        dz_wrf.fill(self._Grid.dx[2])
        z[:, :, :] = self._Grid.z_global[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]
        rho_wrf[:, :, :] = self._Ref.rho0[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]
        exner_wrf[:, :, :] = exner[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]
        p0_wrf[:, :, :] = p0[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]

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
        ime = self._wrf_dims[0]
        jme = self._wrf_dims[2]
        kme = self._wrf_dims[1]
        its = 1
        jts = 1
        kts = 1
        ite = ime
        jte = jme
        kte = kme

        # Reorder arrays
        to_wrf_order(nhalo, T / self._Ref.exner[np.newaxis, np.newaxis, :], T_wrf)
        to_wrf_order(nhalo, qv, qv_wrf)
        to_wrf_order(nhalo, qc, qc_wrf)
        to_wrf_order(nhalo, qr, qr_wrf)
        to_wrf_order(nhalo, w, w_wrf)
        to_wrf_order(nhalo, qnr, qnr_wrf)
        to_wrf_order(nhalo, qnc, nc_wrf)
        to_wrf_order(nhalo, qi1, qi1_wrf)
        to_wrf_order(nhalo, qni1, qni1_wrf)
        to_wrf_order(nhalo, qir1, qir1_wrf)
        to_wrf_order(nhalo, qib1, qib1_wrf)

        np.copyto(th_old, T_wrf, casting="no")
        np.copyto(qv_old, qv_wrf, casting="no")

        n_iceCat = 1

        if self._rain_moment == 2:
            self._p3_cffi.update(
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
                T_wrf,
                qv_wrf,
                qc_wrf,
                qr_wrf,
                qnr_wrf,
                reflectivity_wrf,
                diag_effc_3d_wrf,
                diag_effi_3d_wrf,
                diag_vmi,
                diag_di,
                diag_rhopo,
                th_old,
                qv_old,
                qi1_wrf,
                qni1_wrf,
                qir1_wrf,
                qib1_wrf,
                self._n_diag_3d,
                self._diag_3d,
                liq_sed_wrf,
                ice_sed_wrf,
                nc_wrf,
                exner_wrf,
                p0_wrf,
                dz_wrf,
                w_wrf,
                self._RAINNC,
                self._RAINNCV,
                self._SR,
                self._SNOWNC,
                self._SNOWNCV,
                dt,
                self._itimestep,
                n_iceCat,
            )
        else:
            self._p3_cffi.update_1mom(
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
                T_wrf,
                qv_wrf,
                qc_wrf,
                qr_wrf,
                qnr_wrf,
                reflectivity_wrf,
                diag_effc_3d_wrf,
                diag_effi_3d_wrf,
                diag_vmi,
                diag_di,
                diag_rhopo,
                th_old,
                qv_old,
                qi1_wrf,
                qni1_wrf,
                qir1_wrf,
                qib1_wrf,
                self._n_diag_3d,
                self._diag_3d,
                liq_sed_wrf,
                ice_sed_wrf,
                exner_wrf,
                p0_wrf,
                dz_wrf,
                w_wrf,
                self._RAINNC,
                self._RAINNCV,
                self._SR,
                self._SNOWNC,
                self._SNOWNCV,
                dt,
                self._itimestep,
                n_iceCat,
            )

        # Update prognosed fields
        to_our_order(nhalo, qv_wrf, qv)
        to_our_order(nhalo, qc_wrf, qc)
        to_our_order(nhalo, nc_wrf, qnc)
        to_our_order(nhalo, qr_wrf, qr)
        to_our_order(nhalo, qnr_wrf, qnr)
        to_our_order(nhalo, qi1_wrf, qi1)
        to_our_order(nhalo, qni1_wrf, qni1)

        # The accumulated liquid and ice sedimentation
        to_our_order(nhalo, liq_sed_wrf, liq_sed)
        to_our_order(nhalo, ice_sed_wrf, ice_sed)

        # Reorder the diagnostic arrays
        for i, vn in enumerate(self._diag_3d_vars):
            dv = self._DiagnosticState.get_field("p3_" + vn)
            to_our_order(nhalo, self._diag_3d[:, :, :, i], dv)

        # Update the energys (TODO Move this to numba)
        # T_wrf *= self._Ref.exner[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]
        self._update_static_energy(exner_wrf, z, T_wrf, s_wrf, qc_wrf, qr_wrf, qi1_wrf)

        # s_wrf = (
        #    T_wrf
        #    + (
        #        parameters.G * z
        #        - parameters.LV * (qc_wrf + qr_wrf)
        #        - parameters.LS * (qi1_wrf)
        #    )
        #    * parameters.ICPD
        # )
        to_our_order(nhalo, s_wrf, s)

        # Compute and apply sedimentation sources of static energy
        np.multiply(liq_sed, parameters.LV / parameters.CPD, out=s_tend_liq_sed)
        np.multiply(ice_sed, parameters.LS / parameters.CPD, out=s_tend_ice_sed)
        np.subtract(s, s_tend_liq_sed, out=s)
        np.subtract(s, s_tend_ice_sed, out=s)

        # Convert sedimentation sources to units of tendency
        np.multiply(liq_sed, 1.0 / self._TimeSteppingController.dt, out=liq_sed)
        np.multiply(ice_sed, 1.0 / self._TimeSteppingController.dt, out=ice_sed)
        np.multiply(
            s_tend_liq_sed, -1.0 / self._TimeSteppingController.dt, out=s_tend_liq_sed
        )
        np.multiply(
            s_tend_ice_sed, -1.0 / self._TimeSteppingController.dt, out=s_tend_ice_sed
        )

        to_our_order(nhalo, diag_effc_3d_wrf, diag_effc_3d)
        to_our_order(nhalo, diag_effi_3d_wrf, diag_effi_3d)
        to_our_order(nhalo, reflectivity_wrf, reflectivity)

        self._itimestep += 1

        self._Timers.end_timer("MicroP3_update")
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

        timeseries_grp.createVariable("RAINNCV", np.double, dimensions=("time",))
        v.long_name = "one time step accumulated surface precip"
        v.units = "mm"
        v.latex_name = "rainncv"

        # Now add cloud fraction and rain fraction profiles
        v = profiles_grp.createVariable("CF", np.double, dimensions=("time", "z",))
        v.long_name = "Cloud Fraction"
        v.standard_name = "CF"
        v.units = ""

        profiles_grp.createVariable("RF", np.double, dimensions=("time", "z",))
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
            timeseries_grp["RWP"][-1] = rwp
            timeseries_grp["VWP"][-1] = vwp

            timeseries_grp["RAINNC"][-1] = rainnc
            timeseries_grp["RAINNCV"][-1] = rainncv

            profiles_grp["CF"][-1, :] = cf_prof[n_halo[2] : -n_halo[2]]
            profiles_grp["RF"][-1, :] = rf_prof[n_halo[2] : -n_halo[2]]

        return

    @staticmethod
    @numba.njit
    def _update_static_energy(exner_wrf, z, T_wrf, s_wrf, qc_wrf, qr_wrf, qi1_wrf):
        shape = T_wrf.shape
        # Fortran ordered so we need to loop differently
        for k in range(shape[2]):
            for j in range(shape[1]):
                for i in range(shape[0]):
                    _T = T_wrf[i, j, k] * exner_wrf[i, j, k]
                    T_wrf[i, j, k] = _T
                    s_wrf[i, j, k] = (
                        _T
                        + (
                            parameters.G * z[i, j, k]
                            - parameters.LV * (qc_wrf[i, j, k] + qr_wrf[i, j, k])
                            - parameters.LS * (qi1_wrf[i, j, k])
                        )
                        * parameters.ICPD
                    )

        return

    def io_fields2d_update(self, nc_grp):

        rainnc = nc_grp.createVariable("RAINNC", np.double, dimensions=("X", "Y",))
        rainnc[:, :] = self._RAINNC

        rainncv = nc_grp.createVariable("RAINNCV", np.double, dimensions=("X", "Y"))
        rainncv[:, :] = self._RAINNCV

        nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        qc = self._ScalarState.get_field('qc')[nh[0]:-nh[0], nh[1]:-nh[1],:]
        lwp_compute = np.sum(qc * rho0[np.newaxis, np.newaxis,0] , axis=2)
        lwp = nc_grp.createVariable("lwp", np.double, dimensions=("X", "Y"))
        lwp[:, :] = lwp_compute
       
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
        effi = np.copy(self._DiagnosticState.get_field("diag_effi_3d"))
        return effi 

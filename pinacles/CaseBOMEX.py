import numpy as np
from mpi4py import MPI
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles import UtilitiesParallel


class SurfaceBOMEX(Surface.SurfaceBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):

        Surface.SurfaceBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        self._theta_flux = 8.0e-3  # K m/s
        self._qv_flux = 5.2e-5
        self._ustar = 0.28  # m/s
        self._theta_surface = 299.1  # K
        self.T_surface = 299.1
        self.bflux_from_thflux()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc) + self._buoyancy_flux
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc) + self._ustar

        self._Timers.add_timer("SurfaceBomex_update")

        return

    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp["timeseries"]

        # Add surface windspeed
        v = timeseries_grp.createVariable(
            "wind_horizontal", np.double, dimensions=("time",)
        )
        v.long_name = "Surface layer wind speed"
        v.unts = "m s^{-1}"
        v.standard_name = "surface wind"

        # Add surface stresses
        v = timeseries_grp.createVariable("ustar", np.double, dimensions=("time",))
        v.long_name = "friction velocity"
        v.units = "m s^{-1}"
        v.standard_name = "u^{\star}"

        v = timeseries_grp.createVariable("taux", np.double, dimensions=("time",))
        v.long_name = "surface shear stress x-component"
        v.unts = "m^2 s^{-2}"
        v.standard_name = "\tau{13}"

        v = timeseries_grp.createVariable("tauy", np.double, dimensions=("time",))
        v.long_name = "surface shear stress y-component"
        v.units = "m^2 s^{-2}"
        v.standard_name = "\tau{23}"

        # Add thermodynamic fluxes
        v = timeseries_grp.createVariable("tflx", np.double, dimensions=("time",))
        v.long_name = "surface temperature flux"
        v.units = "K m s^{-2}"
        v.standard_name = "surface temperature flux"

        v = timeseries_grp.createVariable("shf", np.double, dimensions=("time",))
        v.long_name = "surface sensible heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "shf"

        v = timeseries_grp.createVariable("lhf", np.double, dimensions=("time",))
        v.long_name = "surface latent heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "lhf"

        return

    def io_update(self, rt_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        windspeed = (
            np.sum(self._windspeed_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        windspeed = UtilitiesParallel.ScalarAllReduce(windspeed)

        taux = (
            np.sum(self._taux_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        taux = UtilitiesParallel.ScalarAllReduce(taux)

        tauy = (
            np.sum(self._taux_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        tauy = UtilitiesParallel.ScalarAllReduce(tauy)

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["wind_horizontal"][-1] = windspeed
            timeseries_grp["ustar"][-1] = self._ustar
            timeseries_grp["taux"][-1] = taux
            timeseries_grp["tauy"][-1] = tauy
            timeseries_grp["tflx"][-1] = (
                self._theta_flux * self._Ref.exner_edge[n_halo[2] - 1]
            )
            timeseries_grp["shf"][-1] = (
                self._theta_flux
                * self._Ref.exner_edge[n_halo[2] - 1]
                * parameters.CPD
                * self._Ref.rho0_edge[n_halo[2] - 1]
            )
            timeseries_grp["lhf"][-1] = (
                self._qv_flux * parameters.LV * self._Ref.rho0_edge[n_halo[2] - 1]
            )
        return

    def update(self):

        self._Timers.start_timer("SurfaceBomex_update")
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        rho0_edge = self._Ref.rho0_edge

        exner_edge = self._Ref.exner_edge

        # Get Fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        # Get Tendnecies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        utsfc = ut[:, :, nh[2]]
        vtsfc = vt[:, :, nh[2]]
        stsfc = st[:, :, nh[2]]
        qvtsfc = qvt[:, :, nh[2]]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )
        Surface_impl.tau_given_ustar(
            self._ustar_sfc,
            usfc,
            vsfc,
            self._Ref.u0,
            self._Ref.v0,
            self._windspeed_sfc,
            self._taux_sfc,
            self._tauy_sfc,
        )

        tflx = np.zeros_like(self._taux_sfc) + self._theta_flux * exner_edge[nh[2] - 1]
        qv_flx_sf = np.zeros_like(self._taux_sfc) + self._qv_flux
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, tflx, st
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, qv_flx_sf, qvt
        )

        self._Timers.end_timer("SurfaceBomex_update")
        return


class ForcingBOMEX(Forcing.ForcingBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):
        Forcing.ForcingBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        self._f = 0.376e-4

        zl = self._Grid.z_local

        # Set Geostrophic wind
        self._ug = np.zeros_like(self._Grid.z_global)
        for k in range(zl.shape[0]):
            self._ug[k] = -10.0 + (1.8e-3) * zl[k]
        self._vg = np.zeros_like(self._ug)

        # Set heating rate
        self._heating_rate = np.zeros_like(self._Grid.z_global)
        self._subsidence = np.zeros_like(self._Grid.z_global)

        # Convert given form of tendencies (theta) to temperature tendency
        for k in range(zl.shape[0]):
            if zl[k] <= 1500.0:
                self._heating_rate[k] = -2.0 / (3600 * 24.0)  # K/s
            if zl[k] > 1500.0:
                self._heating_rate[k] = -2.0 / (3600 * 24.0) + (zl[k] - 1500.0) * (
                    0.0 - -2.0 / (3600 * 24.0)
                ) / (3000.0 - 1500.0)
            if zl[k] <= 1500.0:
                self._subsidence[k] = 0.0 + zl[k] * (-0.65 / 100.0 - 0.0) / (
                    1500.0 - 0.0
                )
            if zl[k] > 1500.0 and zl[k] <= 2100.0:
                self._subsidence[k] = -0.65 / 100 + (zl[k] - 1500.0) * (
                    0.0 - -0.65 / 100.0
                ) / (2100.0 - 1500.0)

        self._Timers.add_timer("ForcingBomex_update")
        return

    def update(self):

        self._Timers.start_timer("ForcingBomex_update")
        exner = self._Ref.exner

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        st += (self._heating_rate * exner)[np.newaxis, np.newaxis, :]

        Forcing_impl.large_scale_pgf(
            self._ug, self._vg, self._f, u, v, self._Ref.u0, self._Ref.v0, ut, vt
        )

        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], s, st)
        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], qv, qvt)

        self._Timers.end_timer("ForcingBomex_update")
        return

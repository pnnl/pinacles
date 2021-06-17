import numpy as np
from mpi4py import MPI
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles.WRF_Micro_Kessler import compute_qvs
from pinacles import UtilitiesParallel


class SurfaceRICO(Surface.SurfaceBase):
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

        self._cm = 0.001229
        self._ch = 0.001094
        self._cq = 0.001133

        self._T0 = 299.8
        self._P0 = 1.0154e5
        self._qs0 = compute_qvs(self._T0, self._P0)

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.single)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._qvflx = np.zeros_like(self._windspeed_sfc)
        self._tflx = np.zeros_like(self._windspeed_sfc)

        self._Timers.add_timer("SurfaceRICO_update")

        return

    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp["timeseries"]

        # Add surface windspeed
        timeseries_grp.createVariable(
            "wind_horizontal", np.single, dimensions=("time",)
        )

        # Add surface stresses
        # timeseries_grp.createVariable('ustar', np.single, dimensions=('time',))
        v = timeseries_grp.createVariable("taux", np.single, dimensions=("time",))
        v.long_name = "surface shear stress x-component"
        v.unts = "m^2 s^{-2}"
        v.standard_name = "\tau{13}"

        v = timeseries_grp.createVariable("tauy", np.single, dimensions=("time",))
        v.long_name = "surface shear stress y-component"
        v.units = "m^2 s^{-2}"
        v.standard_name = "\tau{23}"

        # Add thermodynamic fluxes
        v = timeseries_grp.createVariable("tflx", np.single, dimensions=("time",))
        v.long_name = "surface temperature flux"
        v.units = "K m s^{-2}"
        v.standard_name = "surface temperature flux"

        v = timeseries_grp.createVariable("qvflx", np.single, dimensions=("time",))
        v.long_name = "surface water vapor flux"
        v.units = "kg kg^{-2} m s^{-2}"
        v.standard_name = "surface water vapor flux"

        v = timeseries_grp.createVariable("shf", np.single, dimensions=("time",))
        v.long_name = "surface sensible heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "shf"

        v = timeseries_grp.createVariable("lhf", np.single, dimensions=("time",))
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

        tauy = (
            np.sum(self._taux_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        tauy = UtilitiesParallel.ScalarAllReduce(tauy)

        tflx = np.sum(self._tflx[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]]) / npts
        tflx = UtilitiesParallel.ScalarAllReduce(tflx)

        qvflx = (
            np.sum(self._qvflx[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]]) / npts
        )
        qvflx = UtilitiesParallel.ScalarAllReduce(qvflx)

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["wind_horizontal"][-1] = windspeed
            # timeseries_grp['ustar'][-1] = self._ustar
            timeseries_grp["taux"][-1] = taux
            timeseries_grp["tauy"][-1] = tauy
            timeseries_grp["tflx"][-1] = tflx
            timeseries_grp["shf"][-1] = (
                tflx * parameters.CPD * self._Ref.rho0_edge[n_halo[2] - 1]
            )
            timeseries_grp["lhf"][-1] = (
                qvflx * parameters.LV * self._Ref.rho0_edge[n_halo[2] - 1]
            )
        return

    def update(self):

        self._Timers.start_timer("SurfaceRICO_update")

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
        qv = self._ScalarState.get_field("qv")
        s = self._ScalarState.get_field("s")
        # Get Tendnecies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        Ssfc = s[:, :, nh[2]]
        qvsfc = qv[:, :, nh[2]]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )
        # Surface_impl.tau_given_ustar(self._ustar_sfc, usfc, vsfc, self._Ref.u0, self._Ref.v0, self._windspeed_sfc, self._taux_sfc, self._tauy_sfc)

        # TODO Not not optimized code
        self._tflx = -self._ch * self._windspeed_sfc * (Ssfc - self._T0)
        self._qvflx = -self._cq * self._windspeed_sfc * (qvsfc - self._qs0)
        Surface_impl.momentum_bulk_aero(
            self._windspeed_sfc,
            self._cm,
            usfc,
            vsfc,
            self._Ref.u0,
            self._Ref.v0,
            self._taux_sfc,
            self._tauy_sfc,
        )
        self._taux_sfc = -self._cm * self._windspeed_sfc * (usfc + self._Ref.u0)
        self._tauy_sfc = -self._cm * self._windspeed_sfc * (vsfc + self._Ref.v0)

        # shf = np.zeros_like(self._taux_sfc) + self._theta_flux * exner_edge[nh[2]-1]
        # qv_flx_sf = np.zeros_like(self._taux_sfc) + self._qv_flux

        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tflx, st
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._qvflx, qvt
        )

        self._Timers.end_timer("SurfaceRICO_update")

        return


class ForcingRICO(Forcing.ForcingBase):
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
        self._subsidence = np.zeros_like(self._Grid.z_global)
        self._ls_mositure = np.zeros_like(self._Grid.z_global)

        for k in range(zl.shape[0]):
            self._ug[k] = -9.9 + 2.0e-3 * zl[k]
            if zl[k] <= 2260.0:
                self._subsidence[k] = -(0.005 / 2260.0) * zl[k]
            else:
                self._subsidence[k] = -0.005

            if zl[k] <= 2980.0:
                self._ls_mositure[k] = (
                    (-1.0 + 1.3456 / 2980.0 * zl[k]) / 86400.0 / 1000.0
                )
            else:
                self._ls_mositure[k] = 0.3456 / 86400.0 / 1000.0
        self._vg = np.zeros_like(self._ug) - 3.8

        # Set heating rate
        self._heating_rate = np.zeros_like(self._Grid.z_global) - 2.5 / 86400.0

        self._Timers.add_timer("ForcingRico_update")
        return

    def update(self):

        self._Timers.start_timer("ForcingRico_update")

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
        qvt += self._ls_mositure[np.newaxis, np.newaxis, :]

        Forcing_impl.large_scale_pgf(
            self._ug, self._vg, self._f, u, v, self._Ref.u0, self._Ref.v0, ut, vt
        )

        # Now ad large scale subsidence
        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], s, st)
        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], qv, qvt)

        self._Timers.end_timer("ForcingRico_update")
        return

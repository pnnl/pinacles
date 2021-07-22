import numpy as np
import numba 
import netCDF4 as nc
from mpi4py import MPI
from pinacles import Surface, Surface_impl, Forcing, Forcing_impl
from pinacles import parameters
from scipy import interpolate
from pinacles import UtilitiesParallel
from pinacles.WRF_Micro_Kessler import compute_qvs


class SurfaceMAGIC(Surface.SurfaceBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        VelocityState,
        ScalarState,
        DiagnosticState,
        TimeSteppingController,
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

        self._TimeSteppingController = TimeSteppingController

        file = namelist["magic"]["input_filepath"]
        data = nc.Dataset(file, "r")
        surface_data = data.groups["surface"]
        self._forcing_times = surface_data.variables["times"][:]
        self._forcing_skintemp = surface_data.variables["surface_temperature"][:]
        self._forcing_psurf = surface_data.variables["surface_pressure"][:]  #hPa

        data.close()

        self._z0 = 0.0      #CaseTestbed
        self._ustar = 0.30  #CaseATEX

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double) + self.gustiness
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc)
        self._shf = np.zeros_like(self._windspeed_sfc)
        self._lhf = np.zeros_like(self._windspeed_sfc)
        self._cm = np.zeros_like(self._windspeed_sfc)
        self._ch = np.zeros_like(self._windspeed_sfc)
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc) + self._ustar

        self._Timers.add_timer("SurfaceMAGIC_update")
        return 

    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp["timeseries"]

        # Add thermodynamic fluxes
        timeseries_grp.createVariable("shf", np.double, dimensions=("time",))
        timeseries_grp.createVariable("lhf", np.double, dimensions=("time",))
        timeseries_grp.createVariable("ustar", np.double, dimensions=("time",))
        timeseries_grp.createVariable("z0", np.double, dimensions=("time",))
        return

    def io_update(self, rt_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["shf"][-1] = self._shf[0]
            timeseries_grp["lhf"][-1] = self._lhf[0]

            timeseries_grp["ustar"][-1] = self._ustar
            timeseries_grp["z0"][-1] = self._z0

        return

    def update(self):
        current_time = self._TimeSteppingController.time
        self.T_surface = interpolate.interp1d(
            self._forcing_times,
            self._forcing_skintemp,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)

        self.P_surface = interpolate.interp1d(
            self._forcing_times,
            self._forcing_psurf,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)

        self.qs_surface = compute_qvs(self.T_surface, self.P_surface)

        # Get grid & reference profile info
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        exner_edge = self._Ref.exner_edge

        # Get fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        # Get tendencies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        Ssfc = s[:, :, nh[2]]
        qvsfc = qv[:, :, nh[2]]

        # Compute the surface stress & apply it
        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )

        # pycles 
        # Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
        # Ri = Nb2 * zb* zb/(windspeed[ij] * windspeed[ij])
        bvf = self._DiagnosticState.get_field("bvf")
        dz = self._Grid.dx[2] / 2.

        bvfsfc = bvf[:, :, nh[2]]
        wssfc = self._windspeed_sfc
        Ri = bvfsfc * dz * dz / (wssfc * wssfc)

        self._z0 = self._compute_z0(dz, wssfc)

        Surface_impl.compute_exchange_coefficients(Ri, dz, self._z0, self._cm, self._ch)
        shf = -self._ch * self._windspeed_sfc * (Ssfc - self.T_surface)    
        s_flx_sf = (
            np.zeros_like(self._taux_sfc)
            + shf * alpha0_edge[nh[2] - 1] / parameters.CPD
        )
        qv_flx_sf = -self._ch * self._windspeed_sfc * (qvsfc - self.qs_surface)  #self._cq = self._ch 

        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, s_flx_sf, st
        )
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, qv_flx_sf, qvt
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
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, self._taux_sfc, ut
        )
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, self._tauy_sfc, vt
        )

        # Store the surface fluxes for output
        self._shf = shf
        self._lhf = qv_flx_sf * parameters.LV / alpha0_edge[nh[2] - 1]
        return

    def _compute_z0(self, z1, wspd):
            kappa = 0.4
            c1 = (0.4 + 0.079 * wspd) / 1000.0
            return z1 * np.exp(-kappa / np.sqrt(c1))


class ForcingMAGIC(Forcing.ForcingBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        VelocityState,
        ScalarState,
        DiagnosticState,
        TimeSteppingController,
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
        self._TimeSteppingController = TimeSteppingController

        file = namelist["magic"]["input_filepath"]
        self._momentum_forcing_method = namelist["magic"]["momentum_forcing"]
        # Options: relaxation, mean_relaxation, geostrophic, none
        self._subsidence_forcing_method = namelist["magic"]["subsidence_forcing"]
        # Options: vertical tendency, subsidence, fixed_subsidence
        # Vertical tendency: provide time-height profiles for qv and temperature (not theta)
        # Subsidence: provide time-height profile of subsidence velocity and tendencies are computed
        # Fixed_subsidence: provide a large scale divergence D and get subsidence velocity as -D * z (constant in time)
        if self._subsidence_forcing_method == "fixed_subsidence":
            _ls_divergence = namelist["magic"]["largescale_divergence"]

        data = nc.Dataset(file, "r")
        forcing_data = data.groups["forcing"]
        lat = forcing_data.variables["latitude"][0]
        self._f = 2.0 * parameters.OMEGA * np.sin(lat * np.pi / 180.0)
        zl = self._Grid.z_local

        # Read in the data, we want to
        forcing_z = forcing_data.variables["z"][:]
        self._forcing_times = forcing_data.variables["times"][:]
        ntimes = np.shape(self._forcing_times)[0]
        if self._momentum_forcing_method == "geostrophic":
            raw_ug = forcing_data.variables["u_geostrophic"][:, :]
            raw_vg = forcing_data.variables["v_geostrophic"][:, :]
        if (
            self._momentum_forcing_method == "relaxation"
            or self._momentum_forcing_method == "mean_relaxation"
        ):
            raw_ur = forcing_data.variables["u_relaxation"][:, :]
            raw_vr = forcing_data.variables["v_relaxation"][:, :]

        raw_adv_qt = forcing_data.variables["qt_advection"][:, :]
        # If both theta and temperature advection are provided, take temperature
        # Otherwise, take theta
        try:
            raw_adv_temperature = forcing_data.variables["temperature_advection"][:, :]
            self._use_temperature_advection = True
        except:
            raw_adv_theta = forcing_data.variables["theta_advection"][:, :]
            self._use_temperature_advection = False

        # Get the subsidence/vertical advection tendency ifnromation
        # Assuming that vertical advection tendencies are provided for temperature, not theta
        # as this is what varanal provides
        if self._subsidence_forcing_method == "fixed_subsidence":
            raw_subsidence = (
                np.ones_like(raw_adv_qt)
                * np.tile(forcing_z[np.newaxis, :], (ntimes, 1))
                * _ls_divergence
            )
        else:
            raw_subsidence = forcing_data.variables["subsidence"][:, :]
        if self._subsidence_forcing_method == "vertical_tendency":
            raw_vtend_qt = forcing_data.variables["qt_vertical_tendency"][:, :]
            # temperature, not theta
            raw_vtend_temperature = forcing_data.variables[
                "temperature_vertical_tendency"
            ][:, :]

        data.close()

        if self._momentum_forcing_method == "geostrophic":
            self._ug = interpolate.interp1d(
                forcing_z, raw_ug, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)
            self._vg = interpolate.interp1d(
                forcing_z, raw_vg, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)
        if (
            self._momentum_forcing_method == "relaxation"
            or self._momentum_forcing_method == "mean_relaxation"
        ):
            self._ur = interpolate.interp1d(
                forcing_z, raw_ur, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)
            self._vr = interpolate.interp1d(
                forcing_z, raw_vr, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)

        self._subsidence = interpolate.interp1d(
            forcing_z,
            raw_subsidence,
            axis=1,
            fill_value="extrapolate",
            assume_sorted=True,
        )(zl)
        self._adv_qt = interpolate.interp1d(
            forcing_z, raw_adv_qt, axis=1, fill_value="extrapolate", assume_sorted=True
        )(zl)
        if self._use_temperature_advection:
            self._adv_temperature = interpolate.interp1d(
                forcing_z,
                raw_adv_temperature,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)
        else:
            self._adv_theta = interpolate.interp1d(
                forcing_z,
                raw_adv_theta,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)
        if self._subsidence_forcing_method == "vertical_tendency":
            self._vtend_temperature = interpolate.interp1d(
                forcing_z,
                raw_vtend_temperature,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)

            self._vtend_qt = interpolate.interp1d(
                forcing_z,
                raw_vtend_qt,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)

        z_top = self._Grid.l[2]
        # Performing relaxation nudging over the same depth as damping, this is an assumption to revisit
        _depth = namelist["damping"]["depth"]
        znudge = z_top - _depth
        # Assume nudging timescale of 1 hour, again this is an assumption that could be revisited
        self._compute_relaxation_coefficient(znudge, 3600.0)

        self._Timers.add_timer("ForcingMAGIC_update")

        return

    def update(self):

        self._Timers.start_timer("ForcingMAGIC_update")

        current_time = self._TimeSteppingController.time

        # interpolate in time
        if self._momentum_forcing_method == "geostrophic":
            current_ug = interpolate.interp1d(
                self._forcing_times,
                self._ug,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
            current_vg = interpolate.interp1d(
                self._forcing_times,
                self._vg,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        if (
            self._momentum_forcing_method == "relaxation"
            or self._momentum_forcing_method == "mean_relaxation"
        ):
            current_ur = interpolate.interp1d(
                self._forcing_times,
                self._ur,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
            current_vr = interpolate.interp1d(
                self._forcing_times,
                self._vr,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        current_subsidence = interpolate.interp1d(
            self._forcing_times,
            self._subsidence,
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        current_adv_qt = interpolate.interp1d(
            self._forcing_times,
            self._adv_qt,
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        if self._use_temperature_advection:
            current_adv_temperature = interpolate.interp1d(
                self._forcing_times,
                self._adv_temperature,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        else:
            current_adv_theta = interpolate.interp1d(
                self._forcing_times,
                self._adv_theta,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        if self._subsidence_forcing_method == "vertical_tendency":
            current_vtend_temperature = interpolate.interp1d(
                self._forcing_times,
                self._vtend_temperature,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
            current_vtend_qt = interpolate.interp1d(
                self._forcing_times,
                self._vtend_qt,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)

        exner = self._Ref.exner

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        if self._use_temperature_advection:
            st += (current_adv_temperature)[np.newaxis, np.newaxis, :]
        else:
            st += (current_adv_theta * exner)[np.newaxis, np.newaxis, :]
        qvt += (current_adv_qt)[np.newaxis, np.newaxis, :]
        if self._momentum_forcing_method == "geostrophic":
            Forcing_impl.large_scale_pgf(
                current_ug,
                current_vg,
                self._f,
                u,
                v,
                self._Ref.u0,
                self._Ref.v0,
                ut,
                vt,
            )
        if self._momentum_forcing_method == "relaxation":
            Forcing_impl.relax_velocities(
                current_ur,
                current_vr,
                u,
                v,
                self._Ref.u0,
                self._Ref.v0,
                ut,
                vt,
                self._relaxation_coefficient,
            )
        if self._momentum_forcing_method == "mean_relaxation":
            umean = self._VelocityState.mean("u")
            vmean = self._VelocityState.mean("v")
            Forcing_impl.relax_mean_velocities(
                current_ur,
                current_vr,
                umean,
                vmean,
                self._Ref.u0,
                self._Ref.v0,
                ut,
                vt,
                self._relaxation_coefficient,
            )

        if self._subsidence_forcing_method == "vertical_tendency":
            st += (
                current_vtend_temperature
                + current_subsidence * parameters.G * parameters.ICPD
            )[np.newaxis, np.newaxis, :]
            qvt += (current_vtend_qt)[np.newaxis, np.newaxis, :]
        else:
            Forcing_impl.apply_subsidence(current_subsidence, self._Grid.dxi[2], s, st)
            Forcing_impl.apply_subsidence(
                current_subsidence, self._Grid.dxi[2], qv, qvt
            )

        self._Timers.end_timer("ForcingMAGIC_update")

        return

    def _compute_relaxation_coefficient(self, znudge, timescale):

        self._relaxation_coefficient = np.zeros(self._Grid.ngrid[2], dtype=np.double)

        z = self._Grid.z_global

        for k in range(self._Grid.ngrid[2]):
            self._relaxation_coefficient[k] = 1.0 / timescale
            if z[k] < znudge:
                self._relaxation_coefficient[k] *= (
                    np.sin((np.pi / 2.0) * (1.0 - (znudge - z[k]) / znudge)) ** 2.0
                )

        return




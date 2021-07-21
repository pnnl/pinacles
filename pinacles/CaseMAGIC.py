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

        file = namelist["testbed"]["input_filepath"]
        data = nc.Dataset(file, "r")
        surface_data = data.groups["surface"]
        self._forcing_times = surface_data.variables["times"][:]
        self._forcing_skintemp = surface_data.variables["surface_temperature"][:]
        self._forcing_psurf = surface_data.variables["surface_pressure"][:]  #hPa

        data.close()

        self._z0 = 0.0      #CaseTestbed
        self._ustar = 0.30  #CaseATEX

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc)
        self._cm = np.zeros_like(self._windspeed_sfc)
        self._ch = np.zeros_like(self._windspeed_sfc)
        self._shf = np.zeros_like(self._windspeed_sfc)
        self._lhf = np.zeros_like(self._windspeed_sfc)
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

            timeseries_grp["shf"][-1] = self._shf
            timeseries_grp["lhf"][-1] = self._lhf

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
        dz = self._Grid.dx[2]

        bvfsfc = bvf[:, :, nh[2]]
        zb = dz[:, :, nh[2]]
        wssfc = self._windspeed_sfc
        Ri = bvfsfc * zb * zb / (wssfc * wssfc)

        Surface_impl.exchange_coefficients_byun(Ri, zb, self._z0, self._cm, self._ch)
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
        self._lhf = qv_flx_sf * parameters.Lv / alpha0_edge[nh[2] - 1]
        return


@numba.njit()
def radiative_transfer(dz, rho0, qc, st):

    shape = qc.shape

    lwp = np.zeros(shape, dtype=np.double)
    lw_flux = np.zeros(shape, dtype=np.double)

    for i in range(shape[0]):
        for j in range(shape[1]):
            # Compute the liquid path
            for k in range(shape[2] - 1, 1, -1):
                lwp[i, j, k - 1] = lwp[i, j, k] + rho0[k] * qc[i, j, k] * dz

            for k in range(shape[2]):
                lw_flux[i, j, k] = 74.0 * np.exp(-130.0 * lwp[i, j, k])

            # Now compute tendencies
            for k in range(1, shape[2]):
                st[i, j, k] -= (
                    (lw_flux[i, j, k] - lw_flux[i, j, k - 1]) / dz / parameters.CPD
                )

    return


class ForcingMAGIC(Forcing.ForcingBase):  #CaseATEX
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        Microphysics,
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
        self._Microphysics = Microphysics

        DiagnosticState.add_variable("radiation_temp_tend")

        self._f = 0.376e-4

        zl = self._Grid.z_local
        nhalo = self._Grid.n_halo
        exner = self._Ref.exner

        # Set the geostrophic wind
        self._ug = np.zeros_like(self._Grid.z_global)
        self._vg = np.zeros_like(self._ug)

        for k in range(self._ug.shape[0]):
            z = zl[k]
            if z <= 150.0:
                self._ug[k] = max(-11.0 + z * (-10.55 - -11.00) / 150.0, -8.0)
                self._vg[k] = -2.0 + z * (-1.90 - -2.0) / 150.0
            elif z > 150.0 and z <= 700.0:
                dz = 700.0 - 150.0
                self._ug[k] = max(-10.55 + (z - 150.0) * (-8.90 - -10.55) / dz, -8.0)
                self._vg[k] = -1.90 + (z - 150.0) * (-1.10 - -1.90) / dz
            elif z > 700.0 and z <= 750.0:
                dz = 750.0 - 700.0
                self._ug[k] = max(-8.90 + (z - 700.0) * (-8.75 - -8.90) / dz, -8.0)
                self._vg[k] = -1.10 + (z - 700.0) * (-1.00 - -1.10) / dz
            elif z > 750.0 and z <= 1400.0:
                dz = 1400.0 - 750.0
                self._ug[k] = max(-8.75 + (z - 750.0) * (-6.80 - -8.75) / dz, -8.0)
                self._vg[k] = -1.00 + (z - 750.0) * (-0.14 - -1.00) / dz
            elif z > 1400.0 and z <= 1650.0:
                dz = 1650.0 - 1400.0
                self._ug[k] = max(-6.80 + (z - 1400.0) * (-6.80 - -5.75) / dz, -8.0)
                self._vg[k] = -0.14 + (z - 1400.0) * (0.18 - -0.14) / dz
            elif z > 1650.0:
                dz = 4000.0 - 1650.0
                self._ug[k] = max(-5.75 + (z - 1650.0) * (1.00 - -5.75) / dz, -8.0)
                self._vg[k] = 0.18 + (z - 1650.0) * (2.75 - 0.18) / dz

        self._Timers.add_timer("ForcingMAGIC_update")
        return

    def update(self):

        self._Timers.start_timer("ForcingMAGIC_update")

        # Get grid and reference information
        zl = self._Grid.z_local
        exner = self._Ref.exner
        rho = self._Ref.rho0
        dxi = self._Grid.dxi
        dx = self._Grid.dx
        n_halo = self._Grid.n_halo

        # Read in fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        qc = self._Microphysics.get_qc()

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        radiation_temp_tend = self._DiagnosticState.get_field("radiation_temp_tend")

        # Compute zi
        qv_mean_prof = self._ScalarState.mean("qv")

        qv_above = np.where(qv_mean_prof > 6.5 / 1000.0)
        n_above = qv_above[0][-1]

        dqvdz = (qv_mean_prof[n_above + 1] - qv_mean_prof[n_above]) * dxi[2]
        extrap_z = ((6.5 / 1000.0) - qv_mean_prof[n_above]) / dqvdz

        zi = zl[n_above] + extrap_z

        # Apply pressure gradient
        Forcing_impl.large_scale_pgf(
            self._ug, self._vg, self._f, u, v, self._Ref.u0, self._Ref.v0, ut, vt
        )

        # Compute subsidence
        subsidence = np.zeros_like(qv_mean_prof)
        free_subsidence = -6.5 / 1000.0
        for k in range(subsidence.shape[0]):
            if zl[k] < zi:
                subsidence[k] = (free_subsidence / zi) * zl[k]
            else:
                subsidence[k] = free_subsidence

        Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], s, st)
        Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], qv, qvt)

        # Todo becareful about applying subsidence to velocity fields
        Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], u, ut)
        Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], v, vt)

        # Heating rates
        if self._TimeSteppingController.time > 5400.0:
            dqtdt = np.zeros_like(subsidence)
            dtdt = np.zeros_like(subsidence)
            for k in range(subsidence.shape[0]):
                dqtdt[k] = -1.58e-8 * (1.0 - zl[k] / zi)
                dtdt[k] = -1.1575e-5 * (3.0 - zl[k] / zi) * exner[k]

            qvt += dqtdt[np.newaxis, np.newaxis, :]
            st += dtdt[np.newaxis, np.newaxis, :]
        dz = dx[2]

        st_old = np.copy(st)

        radiative_transfer(dz, rho, qc, st)
        radiation_temp_tend[:, :, :] = st[:, :, :] - st_old[:, :, :]

        self._Timers.end_timer("ForcingMAGIC_update")

        return




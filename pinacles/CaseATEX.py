import numpy as np
import numba
from mpi4py import MPI
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles.WRF_Micro_Kessler import compute_qvs
from pinacles import parameters
from pinacles import UtilitiesParallel
import pinacles.ThermodynamicsDry_impl as DryThermo


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing ATEX Case")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    # Integrate the reference profile.
    Ref.set_surface(Psfc=1.0154e5, Tsfc=295.750, u0=-3.5, v0=0.38)
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    # Wind is uniform initially
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape
    temp = np.empty(shape[2], dtype=np.double)
    perts = np.random.uniform(-0.1, 0.1, (shape[0], shape[1], shape[2]))
    pertsqt = np.random.uniform(-0.025, 0.025, (shape[0], shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]
                if z <= 150.0:
                    t = 295.75
                    qv[i, j, k] = 13.0 + z * (12.50 - 13.0) / 150.0
                    u[i, j, k] = max(-11.0 + z * (-10.55 - -11.00) / 150.0, -8.0)
                    v[i, j, k] = -2.0 + z * (-1.90 - -2.0) / 150.0
                elif z > 150.0 and z <= 700.0:
                    dz = 700.0 - 150.0
                    t = 295.75
                    qv[i, j, k] = 12.50
                    u[i, j, k] = max(-10.55 + (z - 150.0) * (-8.90 - -10.55) / dz, -8.0)
                    v[i, j, k] = -1.90 + (z - 150.0) * (-1.10 - -1.90) / dz
                elif z > 700.0 and z <= 750.0:
                    dz = 750.0 - 700.0
                    t = 295.75 + (z - 700.0) * (296.125 - 295.75) / dz
                    qv[i, j, k] = 12.50 + (z - 700.0) * (11.50 - 12.50) / dz
                    u[i, j, k] = max(-8.90 + (z - 700.0) * (-8.75 - -8.90) / dz, -8.0)
                    v[i, j, k] = -1.10 + (z - 700.0) * (-1.00 - -1.10) / dz
                elif z > 750.0 and z <= 1400.0:
                    dz = 1400.0 - 750.0
                    t = 296.125 + (z - 750.0) * (297.75 - 296.125) / dz
                    qv[i, j, k] = 11.50 + (z - 750.0) * (10.25 - 11.50) / dz
                    u[i, j, k] = max(-8.75 + (z - 750.0) * (-6.80 - -8.75) / dz, -8.0)
                    v[i, j, k] = -1.00 + (z - 750.0) * (-0.14 - -1.00) / dz
                elif z > 1400.0 and z <= 1650.0:
                    dz = 1650.0 - 1400.0
                    t = 297.75 + (z - 1400.0) * (306.75 - 297.75) / dz
                    qv[i, j, k] = 10.25 + (z - 1400.0) * (4.50 - 10.25) / dz
                    u[i, j, k] = max(-6.80 + (z - 1400.0) * (-5.75 - -6.80) / dz, -8.0)
                    v[i, j, k] = -0.14 + (z - 1400.0) * (0.18 - -0.14) / dz
                elif z > 1650.0:
                    dz = 4000.0 - 1650.0
                    t = 306.75 + (z - 1650.0) * (314.975 - 306.75) / dz
                    qv[i, j, k] = 4.50
                    u[i, j, k] = max(-5.75 + (z - 1650.0) * (1.00 - -5.75) / dz, -8.0)
                    v[i, j, k] = 0.18 + (z - 1650.0) * (2.75 - 0.18) / dz
                temp[k] = qv[i, j, k]
                t *= exner[k]
                if zl[k] < 810.0:
                    t += perts[i, j, k]
                    qv[i, j, k] += pertsqt[i, j, k]
                s[i, j, k] = DryThermo.s(zl[k], t)

    u -= Ref.u0
    v -= Ref.v0

    # u.fill(0.0)
    qv /= 1000.0

    return


class SurfaceATEX(Surface.SurfaceBase):
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

        self._ustar = 0.30
        self._ch = (
            0.0013
            * (np.log(10.0 / 0.015e-2) / np.log((self._Grid.dx[2] / 2.0) / (0.015e-2)))
            ** 2.0
        )
        self._cq = self._ch

        self._T0 = 298.0
        self._P0 = 1.0154e5
        self._qs0 = compute_qvs(self._T0, self._P0)

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc) + self._ustar

        self._Timers.add_timer("SurfaceATEX_update")
        return

    def update(self):

        self._Timers.start_timer("SurfaceATEX_update")

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
        utsfc = ut[:, :, nh[2]]
        vtsfc = vt[:, :, nh[2]]
        stsfc = st[:, :, nh[2]]
        qvtsfc = qvt[:, :, nh[2]]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )
        # Surface_impl.tau_given_ustar(self._ustar_sfc, usfc, vsfc, self._Ref.u0, self._Ref.v0, self._windspeed_sfc, self._taux_sfc, self._tauy_sfc)

        # TODO Not not optimized code
        shf = -self._ch * self._windspeed_sfc * (Ssfc - self._T0)
        qv_flx_sfc = -self._cq * self._windspeed_sfc * (qvsfc - self._qs0)
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

        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, shf, st
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, qv_flx_sfc, qvt
        )

        self._Timers.end_timer("SurfaceATEX_update")

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


class ForcingATEX(Forcing.ForcingBase):
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
                self._ug[k] = -11.0 + z * (-10.55 - -11.00) / 150.0
                self._vg[k] = -2.0 + z * (-1.90 - -2.0) / 150.0
            elif z > 150.0 and z <= 700.0:
                dz = 700.0 - 150.0
                self._ug[k] = -10.55 + (z - 150.0) * (-8.90 - -10.55) / dz
                self._vg[k] = -1.90 + (z - 150.0) * (-1.10 - -1.90) / dz
            elif z > 700.0 and z <= 750.0:
                dz = 750.0 - 700.0
                self._ug[k] = -8.90 + (z - 700.0) * (-8.75 - -8.90) / dz
                self._vg[k] = -1.10 + (z - 700.0) * (-1.00 - -1.10) / dz
            elif z > 750.0 and z <= 1400.0:
                dz = 1400.0 - 750.0
                self._ug[k] = -8.75 + (z - 750.0) * (-6.80 - -8.75) / dz
                self._vg[k] = -1.00 + (z - 750.0) * (-0.14 - -1.00) / dz
            elif z > 1400.0 and z <= 1650.0:
                dz = 1650.0 - 1400.0
                self._ug[k] = -6.80 + (z - 1400.0) * (-5.75 - -6.80) / dz
                self._vg[k] = -0.14 + (z - 1400.0) * (0.18 - -0.14) / dz
            elif z > 1650.0:
                dz = 4000.0 - 1650.0
                self._ug[k] = -5.75 + (z - 1650.0) * (1.00 - -5.75) / dz
                self._vg[k] = 0.18 + (z - 1650.0) * (2.75 - 0.18) / dz

        self._Timers.add_timer("ForcingATEX_update")
        return

    def update(self):

        self._Timers.start_timer("ForcingATEX_update")

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

        zi = zl[n_above] + dxi[2] * ((6.5 / 1000.0) - qv_mean_prof[n_above]) / (
            qv_mean_prof[n_above + 1] - qv_mean_prof[n_above]
        )

        # Apply pressure gradient
        Forcing_impl.large_scale_pgf(
            self._ug, self._vg, self._f, u, v, self._Ref.u0, self._Ref.v0, ut, vt
        )

        if self._TimeSteppingController.time > 5400.0:

            # Compute subsidence
            subsidence = np.zeros_like(qv_mean_prof)
            dqtdt = np.zeros_like(subsidence)
            dtdt = np.zeros_like(subsidence)
            free_subsidence = -6.5 / 1000.0
            for k in range(subsidence.shape[0]):
                if zl[k] < zi:
                    subsidence[k] = (free_subsidence / zi) * zl[k]
                    dqtdt[k] = -1.58e-8 * (1.0 - zl[k] / zi)
                    dtdt[k] = -1.1575e-5 * (3.0 - zl[k] / zi) * exner[k]
                elif zl[k] < zi + 300.0:
                    subsidence[k] = free_subsidence * (300.0 - zl[k] + zi) / 300.0
                    dtdt[k] = -1.1575e-5 * 2.0 * exner[k] * (300.0 - zl[k] + zi) / 300.0

            Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], s, st)
            Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], qv, qvt)

            # Todo becareful about applying subsidence to velocity fields
            Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], u, ut)
            Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2], v, vt)

            # Heating rates

            qvt += dqtdt[np.newaxis, np.newaxis, :]
            st += dtdt[np.newaxis, np.newaxis, :]

        dz = dx[2]

        st_old = np.copy(st)

        radiative_transfer(dz, rho, qc, st)
        radiation_temp_tend[:, :, :] = st[:, :, :] - st_old[:, :, :]

        self._Timers.end_timer("ForcingATEX_update")

        return

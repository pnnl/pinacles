import numpy as np
import numba


@numba.njit(fastmath=True)
def compute_u_fluxes(
    n_halo,
    dx,
    dxi,
    rho0,
    rho0_edge,
    eddy_viscosity,
    dudx,
    dudy,
    dudz,
    dvdx,
    dwdx,
    s11,
    s12,
    s13,
    fluxx,
    fluxy,
    fluxz,
    ut,
):

    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))
    shape = ut.shape
    # Compute the fluxes
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                # s11 = dudx[i, j, k]
                # s12 = 0.5 * (dvdx[i, j, k] + dudy[i, j, k])
                # s13 = 0.5 * (dudz[i, j, k] + dwdx[i, j, k])

                fluxx[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s11[i, j, k]
                fluxy[i, j, k] = (
                    -0.5
                    * rho0[k]
                    * (
                        eddy_viscosity[i, j, k]
                        + eddy_viscosity[i + 1, j, k]
                        + eddy_viscosity[i, j + 1, k]
                        + eddy_viscosity[i + 1, j + 1, k]
                    )
                    * s12[i, j, k]
                )
                fluxz[i, j, k] = (
                    -0.5
                    * rho0_edge[k]
                    * (
                        eddy_viscosity[i, j, k]
                        + eddy_viscosity[i + 1, j, k]
                        + eddy_viscosity[i, j, k + 1]
                        + eddy_viscosity[i + 1, j, k + 1]
                    )
                    * s13[i, j, k]
                )

    # Compute the flux divergences
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):
                bc = 1.0
                if k <= n_halo[2]:
                    bc = 0.0

                # For u the x flux is in the correct location
                ut[i, j, k] -= (fluxx[i + 1, j, k] - fluxx[i, j, k]) * dxi[0] / rho0[k]
                ut[i, j, k] -= (fluxy[i, j, k] - fluxy[i, j - 1, k]) * dxi[1] / rho0[k]
                ut[i, j, k] -= (
                    (fluxz[i, j, k] - bc * fluxz[i, j, k - 1]) * dxi[2] / rho0[k]
                )

    return


@numba.njit(fastmath=True)
def compute_v_fluxes(
    n_halo,
    dx,
    dxi,
    rho0,
    rho0_edge,
    eddy_viscosity,
    dvdx,
    dvdy,
    dvdz,
    dudy,
    dwdy,
    s21,
    s22,
    s23,
    fluxx,
    fluxy,
    fluxz,
    vt,
):

    shape = vt.shape

    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))
    # Compute the fluxes
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                fluxx[i, j, k] = (
                    -0.5
                    * rho0[k]
                    * (
                        eddy_viscosity[i, j, k]
                        + eddy_viscosity[i + 1, j, k]
                        + eddy_viscosity[i, j + 1, k]
                        + eddy_viscosity[i + 1, j + 1, k]
                    )
                    * s21[i, j, k]
                )
                fluxy[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s22[i, j, k]
                fluxz[i, j, k] = (
                    -0.5
                    * rho0_edge[k]
                    * (
                        eddy_viscosity[i, j, k]
                        + eddy_viscosity[i, j + 1, k]
                        + eddy_viscosity[i, j, k + 1]
                        + eddy_viscosity[i, j + 1, k + 1]
                    )
                    * s23[i, j, k]
                )  # * kz_fact

    # Compute the flux divergences
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                bc = 1.0
                if k <= n_halo[2]:
                    bc = 0.0
                vt[i, j, k] -= (fluxx[i, j, k] - fluxx[i - 1, j, k]) * dxi[0] / rho0[k]
                vt[i, j, k] -= (fluxy[i, j + 1, k] - fluxy[i, j, k]) * dxi[1] / rho0[k]
                vt[i, j, k] -= (
                    (fluxz[i, j, k] - fluxz[i, j, k - 1] * bc) * dxi[2] / rho0[k]
                )
    return


@numba.njit(fastmath=True)
def compute_w_fluxes(
    n_halo,
    dx,
    dxi,
    rho0,
    rho0_edge,
    eddy_viscosity,
    dwdx,
    dwdy,
    dwdz,
    dudz,
    dvdz,
    s13,
    s23,
    s33,
    fluxx,
    fluxy,
    fluxz,
    wt,
):

    shape = wt.shape

    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))

    # Compute the fluxes
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                fluxx[i, j, k] = (
                    -0.5
                    * rho0_edge[k]
                    * (
                        eddy_viscosity[i, j, k]
                        + eddy_viscosity[i + 1, j, k]
                        + eddy_viscosity[i, j, k + 1]
                        + eddy_viscosity[i + 1, j, k + 1]
                    )
                    * s13[i, j, k]
                )
                fluxy[i, j, k] = (
                    -0.5
                    * rho0_edge[k]
                    * (
                        eddy_viscosity[i, j, k]
                        + eddy_viscosity[i, j + 1, k]
                        + eddy_viscosity[i, j, k + 1]
                        + eddy_viscosity[i, j + 1, k + 1]
                    )
                    * s23[i, j, k]
                )
                fluxz[i, j, k] = (
                    -2.0 * rho0[k + 1] * eddy_viscosity[i, j, k] * s33[i, j, k]
                )

    # Compute the flux divergences
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                wt[i, j, k] -= (
                    (fluxx[i, j, k] - fluxx[i - 1, j, k]) * dxi[0] / rho0_edge[k]
                )

                wt[i, j, k] -= (
                    (fluxy[i, j, k] - fluxy[i, j - 1, k]) * dxi[1] / rho0_edge[k]
                )

                bc = 1.0
                if k <= n_halo[2]:
                    bc = 0.0

                wt[i, j, k] -= (
                    (fluxz[i, j, k + 1] - bc * fluxz[i, j, k]) * dxi[2] / rho0_edge[k]
                )

    return


class MomentumDiffusion:
    def __init__(
        self, namelist, Timers, Grid, Ref, DiagnosticState, Kine, VelocityState
    ):

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._DiagnosticState = DiagnosticState
        self._VelocityState = VelocityState
        self._Kine = Kine

        self._fluxx = np.zeros(self._Grid.ngrid_local, dtype=np.double)
        self._fluxy = np.zeros_like(self._fluxx)
        self._fluxz = np.zeros_like(self._fluxx)

        # Add diagnostic fields
        self._DiagnosticState.add_variable(
            "uw_sgs",
            long_name="Vertical SGS flux of zonal momentum",
            loc='z',
            units="m^2 s^-2",
            latex_name=r"\overline{u'w'}_{sgs}",
        )
        self._DiagnosticState.add_variable(
            "vw_sgs",
            long_name="Vertical SGS flux of meridional momentum",
            loc='z',
            units="m^2 s^-2",
            latex_name=r"\overline{v'w'}_{sgs}",
        )

        self._Timers.add_timer("MomentumDiffusion_update")
        return

    def update(self):

        self._Timers.start_timer("MomentumDiffusion_update")

        dxi = self._Grid.dxi
        dx = self._Grid.dx
        n_halo = self._Grid.n_halo

        rho0 = self._Ref.rho0
        alpha0 = self._Ref.alpha0
        rho0_edge = self._Ref.rho0_edge

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        wt = self._VelocityState.get_tend("w")

        eddy_viscosity = self._DiagnosticState.get_field("eddy_viscosity")

        fluxx = self._fluxx
        fluxy = self._fluxy
        fluxz = self._fluxz

        dudx = self._Kine._dudx
        dudy = self._Kine._dudy
        dudz = self._Kine._dudz

        dvdx = self._Kine._dvdx
        dvdy = self._Kine._dvdy
        dvdz = self._Kine._dvdz

        dwdx = self._Kine._dwdx
        dwdy = self._Kine._dwdy
        dwdz = self._Kine._dwdz

        s11 = self._Kine._s11
        s12 = self._Kine._s12
        s13 = self._Kine._s13

        s22 = self._Kine._s22
        s23 = self._Kine._s23

        s33 = self._Kine._s33

        compute_u_fluxes(
            n_halo,
            dx,
            dxi,
            rho0,
            rho0_edge,
            eddy_viscosity,
            dudx,
            dudy,
            dudz,
            dvdx,
            dwdx,
            s11,
            s12,
            s13,
            fluxx,
            fluxy,
            fluxz,
            ut,
        )

        uw_sgs = self._DiagnosticState.get_field('uw_sgs')
        uw_sgs[:] = fluxz[:]

        compute_v_fluxes(
            n_halo,
            dx,
            dxi,
            rho0,
            rho0_edge,
            eddy_viscosity,
            dvdx,
            dvdy,
            dvdz,
            dudy,
            dwdy,
            s12,
            s22,
            s23,
            fluxx,
            fluxy,
            fluxz,
            vt,
        )

        vw_sgs = self._DiagnosticState.get_field('vw_sgs')
        vw_sgs[:] = fluxz[:]
        
        compute_w_fluxes(
            n_halo,
            dx,
            dxi,
            rho0,
            rho0_edge,
            eddy_viscosity,
            dwdx,
            dwdy,
            dwdz,
            dudz,
            dvdz,
            s13,
            s23,
            s33,
            fluxx,
            fluxy,
            fluxz,
            wt,
        )

        self._Timers.end_timer("MomentumDiffusion_update")

        return

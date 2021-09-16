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
    fluxx,
    fluxy,
    fluxz,
    ut,
):

    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))
    shape = ut.shape
    # Compute the fluxes
    for i in range(1, shape[0] -1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                s11 = dudx[i, j, k]
                s12 = 0.5 * (dvdx[i, j, k] + dudy[i, j, k])
                s13 = 0.5 * (dudz[i, j, k] +  dwdx[i, j, k])

                fluxx[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s11 
                fluxy[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s12
                fluxz[i, j, k] = (
                    -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s13
                )  # * kz_fact

    # Compute the flux divergences
    for i in range(1, shape[0] -1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                # For u the x flux is in the correct location
                ut[i, j, k] -= (fluxx[i + 1, j, k] - fluxx[i, j, k]) * dxi[0] / rho0[k]

                ut[i, j, k] -= (
                    0.25
                    * (
                        (
                            (
                                fluxy[i, j, k]
                                + fluxy[i + 1, j, k]
                                + fluxy[i, j + 1, k]
                                + fluxy[i + 1, j + 1, k]
                            )
                            - (
                                fluxy[i, j, k]
                                + fluxy[i + 1, j, k]
                                + fluxy[i, j - 1, k]
                                + fluxy[i + 1, j - 1, k]
                            )
                        )
                        * dxi[1]
                    )
                    / rho0[k]
                )

                ut[i, j, k] -= (
                    0.25
                    * (
                        (
                            (
                                fluxz[i, j, k]
                                + fluxz[i + 1, j, k]
                                + fluxz[i, j, k + 1]
                                + fluxz[i + 1, j, k + 1]
                            )
                            - (
                                fluxz[i, j, k]
                                + fluxz[i + 1, j, k]
                                + fluxz[i, j, k - 1]
                                + fluxz[i + 1, j, k - 1]
                            )
                        )
                        * dxi[2]
                    )
                    / rho0[k]
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
    fluxx,
    fluxy,
    fluxz,
    vt,
):

    shape = vt.shape

    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))
    # Compute the fluxes
    for i in range(1, shape[0] -1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                s22 = dvdy[i, j, k]
                s21 = 0.5 * (dvdx[i, j, k] + dudy[i, j, k])
                s23 = 0.5 * (dvdz[i, j, k] + dwdy[i, j, k])

                fluxx[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s21
                fluxy[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s22
                fluxz[i, j, k] = (
                    -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s23
                )  # * kz_fact

    # Compute the flux divergences
    for i in range(1, shape[0] -1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):
                vt[i, j, k] -= (
                    0.25
                    * (
                        (
                            (
                                fluxx[i, j, k]
                                + fluxx[i, j + 1, k]
                                + fluxx[i + 1, j, k]
                                + fluxx[i + 1, j + 1, k]
                            )
                            - (
                                fluxx[i, j, k]
                                + fluxx[i, j + 1, k]
                                + fluxx[i - 1, j, k]
                                + fluxx[i - 1, j + 1, k]
                            )
                        )
                        * dxi[0]
                    )
                    / rho0[k]
                )
                vt[i, j, k] -= (fluxy[i, j + 1, k] - fluxy[i, j, k]) * dxi[1] / rho0[k]
                vt[i, j, k] -= (
                    0.25
                    * (
                        (
                            (
                                fluxz[i, j, k]
                                + fluxz[i, j + 1, k]
                                + fluxz[i, j, k + 1]
                                + fluxz[i, j + 1, k + 1]
                            )
                            - (
                                fluxz[i, j, k]
                                + fluxz[i, j + 1, k]
                                + fluxz[i, j, k - 1]
                                + fluxz[i, j + 1, k - 1]
                            )
                        )
                        * dxi[2]
                    )
                    / rho0[k]
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
    fluxx,
    fluxy,
    fluxz,
    wt,
):

    shape = wt.shape

    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))

    # Compute the fluxes
    for i in range(1, shape[0] -1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                s33 = dwdz[i, j, k]
                s31 = 0.5 * (dwdx[i, j, k] + dudz[i, j, k])
                s23 = 0.5 * (dvdz[i, j, k] + dwdy[i, j, k])

                fluxx[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s31
                fluxy[i, j, k] = -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s23
                fluxz[i, j, k] = (
                    -2.0 * rho0[k] * eddy_viscosity[i, j, k] * s33
                )  # * kz_fact

    # Compute the flux divergences
    for i in range(1, shape[0] -1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                wt[i, j, k] -= (
                    0.25
                    * (
                        (
                            (
                                fluxx[i, j, k]
                                + fluxx[i, j, k + 1]
                                + fluxx[i + 1, j, k]
                                + fluxx[i + 1, j, k + 1]
                            )
                            - (
                                fluxx[i, j, k]
                                + fluxx[i, j, k + 1]
                                + fluxx[i - 1, j, k]
                                + fluxx[i - 1, j, k + 1]
                            )
                        )
                        * dxi[0]
                    )
                    / rho0_edge[k]
                )

                wt[i, j, k] -= (
                    0.25
                    * (
                        (
                            (
                                fluxy[i, j, k]
                                + fluxy[i, j, k + 1]
                                + fluxy[i, j + 1, k]
                                + fluxy[i, j + 1, k + 1]
                            )
                            - (
                                fluxy[i, j, k]
                                + fluxy[i, j, k + 1]
                                + fluxy[i, j - 1, k]
                                + fluxy[i, j - 1, k + 1]
                            )
                        )
                        * dxi[1]
                    )
                    / rho0_edge[k]
                )

                wt[i, j, k] -= (
                    (fluxz[i, j, k + 1] - fluxz[i, j, k]) * dxi[2] / rho0_edge[k]
                )
                # wt[i,j,k] -= ((fluxx[i+1,j,k] - fluxx[i,j,k])*dxi[0]
                #    + (fluxy[i,j+1,k] - fluxy[i,j,k])*dxi[1]
                #    + (fluxz[i,j,k+1] - fluxz[i,j,k])*dxi[2])/ rho0_edge[k]

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
            fluxx,
            fluxy,
            fluxz,
            ut,
        )

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
            fluxx,
            fluxy,
            fluxz,
            vt,
        )

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
            fluxx,
            fluxy,
            fluxz,
            wt,
        )

        self._Timers.end_timer("MomentumDiffusion_update")

        return

import numba
from ..interpolation_impl import interp_weno5


@numba.njit(fastmath=True)
def weno5_advection(rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz):
    phi_shape = phi.shape
    for i in range(2, phi_shape[0] - 3):
        for j in range(2, phi_shape[1] - 3):
            for k in range(2, phi_shape[2] - 3):
                # First compute x-advection
                if u[i, j, k] >= 0:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno5(
                            phi[i - 2, j, k],
                            phi[i - 1, j, k],
                            phi[i, j, k],
                            phi[i + 1, j, k],
                            phi[i + 2, j, k],
                        )
                    )
                else:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno5(
                            phi[i + 3, j, k],
                            phi[i + 2, j, k],
                            phi[i + 1, j, k],
                            phi[i, j, k],
                            phi[i - 1, j, k],
                        )
                    )

                # First compute y-advection
                if v[i, j, k] >= 0:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno5(
                            phi[i, j - 2, k],
                            phi[i, j - 1, k],
                            phi[i, j, k],
                            phi[i, j + 1, k],
                            phi[i, j + 2, k],
                        )
                    )
                else:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno5(
                            phi[i, j + 3, k],
                            phi[i, j + 2, k],
                            phi[i, j + 1, k],
                            phi[i, j, k],
                            phi[i, j - 1, k],
                        )
                    )

                # First compute y-advection
                if w[i, j, k] >= 0:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno5(
                            phi[i, j, k - 2],
                            phi[i, j, k - 1],
                            phi[i, j, k],
                            phi[i, j, k + 1],
                            phi[i, j, k + 2],
                        )
                    )
                else:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno5(
                            phi[i, j, k + 3],
                            phi[i, j, k + 2],
                            phi[i, j, k + 1],
                            phi[i, j, k],
                            phi[i, j, k - 1],
                        )
                    )

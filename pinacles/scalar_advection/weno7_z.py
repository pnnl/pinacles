import numpy as np
import numba
from ..interpolation_impl import interp_weno7_z


@numba.njit(fastmath=True)
def weno7_advection_z(rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz):
    phi_shape = phi.shape
    for i in range(3, phi_shape[0] - 4):
        for j in range(3, phi_shape[1] - 4):
            for k in range(3, phi_shape[2] - 4):
                # First compute x-advection
                if u[i, j, k] >= 0:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno7_z(
                            phi[i - 3, j, k],
                            phi[i - 2, j, k],
                            phi[i - 1, j, k],
                            phi[i, j, k],
                            phi[i + 1, j, k],
                            phi[i + 2, j, k],
                            phi[i + 3, j, k],
                        )
                    )
                else:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno7_z(
                            phi[i + 4, j, k],
                            phi[i + 3, j, k],
                            phi[i + 2, j, k],
                            phi[i + 1, j, k],
                            phi[i, j, k],
                            phi[i - 1, j, k],
                            phi[i - 2, j, k],
                        )
                    )

                # First compute y-advection
                if v[i, j, k] >= 0:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno7_z(
                            phi[i, j - 3, k],
                            phi[i, j - 2, k],
                            phi[i, j - 1, k],
                            phi[i, j, k],
                            phi[i, j + 1, k],
                            phi[i, j + 2, k],
                            phi[i, j + 3, k],
                        )
                    )
                else:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno7_z(
                            phi[i, j + 4, k],
                            phi[i, j + 3, k],
                            phi[i, j + 2, k],
                            phi[i, j + 1, k],
                            phi[i, j, k],
                            phi[i, j - 1, k],
                            phi[i, j - 2, k],
                        )
                    )

                # First compute y-advection
                if w[i, j, k] >= 0:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno7_z(
                            phi[i, j, k - 3],
                            phi[i, j, k - 2],
                            phi[i, j, k - 1],
                            phi[i, j, k],
                            phi[i, j, k + 1],
                            phi[i, j, k + 2],
                            phi[i, j, k + 3],
                        )
                    )
                else:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno7_z(
                            phi[i, j, k + 4],
                            phi[i, j, k + 3],
                            phi[i, j, k + 2],
                            phi[i, j, k + 1],
                            phi[i, j, k],
                            phi[i, j, k - 1],
                            phi[i, j, k - 2],
                        )
                    )

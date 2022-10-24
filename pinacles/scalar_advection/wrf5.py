import numba
from ..interpolation_impl import wrf_fifth


@numba.njit(fastmath=True)
def wrf5_advection(rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz):
    phi_shape = phi.shape
    for i in range(2, phi_shape[0] - 3):
        for j in range(2, phi_shape[1] - 3):
            for k in range(2, phi_shape[2] - 3):
                # First compute x-advection
                fluxx[i, j, k] = (
                    rho0[k]
                    * u[i, j, k]
                    * wrf_fifth(
                        u[i, j, k],
                        phi[i - 2, j, k],
                        phi[i - 1, j, k],
                        phi[i, j, k],
                        phi[i + 1, j, k],
                        phi[i + 2, j, k],
                        phi[i + 3, j, k],
                    )
                )

                # First compute y-advection
                fluxy[i, j, k] = (
                    rho0[k]
                    * v[i, j, k]
                    * wrf_fifth(
                        v[i, j, k],
                        phi[i, j - 2, k],
                        phi[i, j - 1, k],
                        phi[i, j, k],
                        phi[i, j + 1, k],
                        phi[i, j + 2, k],
                        phi[i, j + 3, k],
                    )
                )

                # First compute y-advection
                fluxz[i, j, k] = (
                    rho0_edge[k]
                    * w[i, j, k]
                    * wrf_fifth(
                        w[i, j, k],
                        phi[i, j, k - 2],
                        phi[i, j, k - 1],
                        phi[i, j, k],
                        phi[i, j, k + 1],
                        phi[i, j, k + 2],
                        phi[i, j, k + 3],
                    )
                )

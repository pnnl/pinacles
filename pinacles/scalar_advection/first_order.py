import numba


@numba.njit(fastmath=True)
def first_order(rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz):

    phi_shape = phi.shape
    for i in range(0, phi_shape[0] - 1):
        for j in range(0, phi_shape[1] - 1):
            for k in range(0, phi_shape[2] - 1):
                # First compute x-advection
                if u[i, j, k] >= 0:
                    fluxx[i, j, k] = rho0[k] * u[i, j, k] * phi[i, j, k]
                else:
                    fluxx[i, j, k] = rho0[k] * u[i, j, k] * phi[i + 1, j, k]

                # First compute y-advection
                if v[i, j, k] >= 0:
                    fluxy[i, j, k] = rho0[k] * v[i, j, k] * phi[i, j, k]
                else:
                    fluxy[i, j, k] = rho0[k] * v[i, j, k] * phi[i, j + 1, k]

                # First compute y-advection
                if w[i, j, k] >= 0:
                    fluxz[i, j, k] = rho0_edge[k] * w[i, j, k] * phi[i, j, k]
                else:
                    fluxz[i, j, k] = rho0_edge[k] * w[i, j, k] * phi[i, j, k + 1]

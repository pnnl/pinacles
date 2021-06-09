import numba


@numba.njit(fastmath=True)
def divergence(n_halo, dxs, rho0, rho0_edge, u, v, w, div):
    # Second order divergence
    shape = u.shape
    nh0 = n_halo[0]
    nh1 = n_halo[1]
    nh2 = n_halo[2]
    for i in range(nh0, shape[0] - nh0):
        for j in range(nh1, shape[1] - nh1):
            for k in range(nh2, shape[2] - nh2):
                div[i - nh0, j - nh1, k - nh2] = (
                    (u[i, j, k] - u[i - 1, j, k]) / dxs[0] * rho0[k]
                    + (v[i, j, k] - v[i, j - 1, k]) / dxs[1] * rho0[k]
                    + (w[i, j, k] * rho0_edge[k] - w[i, j, k - 1] * rho0_edge[k - 1])
                    / dxs[2]
                )

    return

@numba.njit()
def divergence_ghost(n_halo, dxs, rho0, rho0_edge, u, v, w, div):
    shape = u.shape
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            for k in range(1, shape[2]):
                div[i , j, k] = (
                    (u[i, j, k] - u[i - 1, j, k]) / dxs[0] * rho0[k]
                    + (v[i, j, k] - v[i, j - 1, k]) / dxs[1] * rho0[k]
                    + (w[i, j, k] * rho0_edge[k] - w[i, j, k - 1] * rho0_edge[k - 1])
                    / dxs[2]
                )


    return


@numba.njit(fastmath=True)
def fill_pressure(n_halo, pres, dynp):
    # Copy only the diagnosed real part of the pressure field
    shape = pres.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dynp[i + n_halo[0], j + n_halo[1], k + n_halo[2]] = pres[i, j, k].real

    return


@numba.njit(fastmath=True)
def apply_pressure(dxs, dynp, u, v, w):
    # Use the diagnosed pressure to enforce continuity
    shape = dynp.shape
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):
                u[i, j, k] -= (dynp[i + 1, j, k] - dynp[i, j, k]) / dxs[0]
                v[i, j, k] -= (dynp[i, j + 1, k] - dynp[i, j, k]) / dxs[1]
                w[i, j, k] -= (dynp[i, j, k + 1] - dynp[i, j, k]) / dxs[2]

    return

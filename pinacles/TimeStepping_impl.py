import numpy as np
import numba


@numba.njit
def rk2ssp_s0(state_0, tend_0, dt):
    shape = state_0.shape
    for n in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    state_0[n, i, j, k] += tend_0[n, i, j, k] * dt
                    tend_0[n, i, j, k] = 0.0

    return


@numba.njit
def rk2ssp_s1(state_0, state_1, tend_1, dt):
    shape = state_0.shape
    for n in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    state_1[n, i, j, k] = (
                        state_0[n, i, j, k]
                        + 0.5 * ( (state_1[n, i, j, k]  - state_0[n,i,j,k]) + tend_1[n, i, j, k] * dt)
                    )
                    tend_1[n, i, j, k] = 0.0
    return


@numba.njit
def comput_local_cfl_max(nhalo, dxi, u, v, w):
    shape = u.shape
    cfl_max = -1e6
    umax = -1e6
    vmax = -1e6
    wmax = -1e6
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                xcfl = np.abs(u[i, j, k] * dxi[0])
                ycfl = np.abs(v[i, j, k] * dxi[1])
                zcfl = np.abs(w[i, j, k] * dxi[2])
                cfl_max = max(cfl_max, xcfl + ycfl + zcfl)
                umax = max(umax, np.abs(u[i, j, k]))
                vmax = max(vmax, np.abs(v[i, j, k]))
                wmax = max(wmax, np.abs(w[i, j, k]))

    return cfl_max, umax, vmax, wmax


@numba.njit(fastmath=True)
def compute_local_diff_num_max(nhalo, dxi, dt, Km):
    """Returns number diffusion number following
    Heus et al. (2010) for this MPI rank.

    Args:
        nhalo (tuple): number of halo points
        dxi ([type]): grid spacing
        dt ([type]): model time step
        Km ([type]): eddy diffusion coefficient

    Returns:
        double:
    """
    shape = Km.shape

    idx2_sum = (dxi[0] * dxi[0]) + (dxi[1] * dxi[1]) + (dxi[2] * dxi[2])
    diff_max = -1e-8
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                diff_max = max(diff_max, Km[i, j, k] * idx2_sum)

    return diff_max * dt

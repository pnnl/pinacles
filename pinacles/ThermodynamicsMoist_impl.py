import numba
from pinacles import parameters


@numba.njit(fastmath=True)
def s(z, T, ql, qi):
    return (
        T
        + (parameters.G * z - parameters.LV * ql - parameters.LS * qi) * parameters.ICPD
    )


@numba.njit(fastmath=True)
def T(z, s, ql, qi):
    return (
        s
        + (parameters.LV * ql + parameters.LS * qi - parameters.G * z) * parameters.ICPD
    )


@numba.njit(fastmath=True)
def alpha(P, T, qv, ql, qi):
    return parameters.RD * T / P * (1.0 - (qv + ql + qi) + parameters.EPSVI * qv)


@numba.njit(fastmath=True)
def rho(P, T, qv, ql, qi):
    return 1.0 / alpha(P, T, qv, ql, qi)


@numba.njit(fastmath=True)
def buoyancy(alpha0, alpha):
    return parameters.G * (alpha - alpha0) / alpha0


@numba.njit(fastmath=True)
def compute_bvf(n_halo, theta_ref, exner, T, qv, ql, qi, dz, thetav, bvf):

    shape = bvf.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                thetav[i, j, k] = (
                    T[i, j, k]
                    / exner[k]
                    * (1.0 + 0.61 * qv[i, j, k] - ql[i, j, k] - qi[i, j, k])
                )

    for i in range(shape[0]):
        for j in range(shape[1]):
            k = n_halo[2]
            bvf[i, j, k] = (
                parameters.G
                / theta_ref[k]
                * (thetav[i, j, k + 1] - thetav[i, j, k])
                / (dz)
            )
            for k in range(n_halo[2] + 2, shape[2] - n_halo[2]):
                bvf[i, j, k] = (
                    parameters.G
                    / theta_ref[k]
                    * (thetav[i, j, k + 1] - thetav[i, j, k - 1])
                    / (2.0 * dz)
                )
            k = shape[2] - n_halo[2] - 1
            bvf[i, j, k] = (
                parameters.G
                / theta_ref[k]
                * (thetav[i, j, k] - thetav[i, j, k - 1])
                / (dz)
            )

    return


@numba.njit(fastmath=True)
def eos(
    z_in, P_in, alpha0, s_in, qv_in, ql_in, qi_in, T_out, tref, alpha_out, buoyancy_out
):
    shape = s_in.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                T_out[i, j, k] = T(
                    z_in[k], s_in[i, j, k], ql_in[i, j, k], qi_in[i, j, k]
                )
                alpha_out[i, j, k] = alpha(P_in[k], T_out[i, j, k],qv_in[i,j,k], ql_in[i,j,k], qi_in[i,j,k])
                buoyancy_out[i,j,k] = parameters.G * (alpha_out[i,j,k] - alpha0[k])/alpha0[k]


    return


@numba.njit(fastmath=True)
def apply_buoyancy(buoyancy, w_t):
    shape = w_t.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                w_t[i, j, k] += 0.5 * (buoyancy[i, j, k] + buoyancy[i, j, k + 1])
    return

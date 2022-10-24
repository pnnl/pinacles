import numba
from pinacles import parameters
import numpy as np


@numba.njit(fastmath=True)
def compute_sat(temp, rho0, pressure):
    ep2 = 287.0 / 461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15
    _es = 1000.0 * svp1 * np.exp(svp2 * (temp - svpt0) / (temp - svp3))
    # qvs = ep2 * _es / (pressure - _es)
    rhov = _es / (parameters.RV * temp)
    return _es, rhov / rho0


@numba.njit(fastmath=True)
def compute_qvs(temp, rho0, pressure):
    ep2 = 287.0 / 461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15
    _es = 1000.0 * svp1 * np.exp(svp2 * (temp - svpt0) / (temp - svp3))
    rhov = _es / (parameters.RV * temp)

    return rhov / rho0


@numba.njit(fastmath=True)
def s(z, T, ql, qi):
    return (
        T
        + (parameters.G * z - parameters.LV * ql - parameters.LS * qi) * parameters.ICPD
    )

@numba.njit(fastmath=True)
def s_dry(z, T):
    return (
        T
        + (parameters.G * z) * parameters.ICPD
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
def compute_thetav(T, exner, qv, ql, qi, thetav):

    shape = thetav.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                thetav[i, j, k] = (
                    T[i, j, k]
                    / exner[k]
                    * (1.0 + 0.61 * qv[i, j, k] - ql[i, j, k] - qi[i, j, k])
                )

    return


@numba.njit(fastmath=True)
def gamma_m(qv, T):
    """
    Moist-adiabatic lapse rate computed consistent with:
    https://glossary.ametsoc.org/wiki/Adiabatic_lapse_rate

    """

    gm = (
        parameters.G
        * (1 + (parameters.LV * qv) / (parameters.RD * T))
        / (
            parameters.CPD
            + (parameters.LV ** 2.0 * qv * parameters.EPSV) / (parameters.RD * T ** 2.0)
        )
    )
    return gm


@numba.njit(fastmath=True)
def compute_moist_bvf(dz, n_halo, p0, rho0, T, qv, ql, qi, bvf):

    """

    For cloudy conditions we compute equation 5 in Durran and Klemp (1982). Here we have assumed
    the equivalence of mixing ratios and mass-fractions.

    For cloud free conditions we compute equation 1a in Durran and Klemp (1982), with an additional
    term accounting for virtual temperature effects. The virtual temperature term is the last term on the
    righthand side of equation 6 in the Durran and Klemp paper.

    """

    shape = bvf.shape

    for i in range(shape[0]):
        for j in range(shape[1]):

            # Lower boundary
            k = n_halo[2]
            if ql[i, j, k] + qi[i, j, k] < 1e-8:
                bvf[i, j, k] = (
                    parameters.G
                    / (T[i, j, k])
                    * (
                        (T[i, j, k + 1] - T[i, j, k]) / dz
                        + parameters.G / parameters.CPD
                    )
                    - parameters.G
                    / (1.0 + qv[i, j, k])
                    * (qv[i, j, k + 1] - qv[i, j, k])
                    / dz
                )
            else:
                qt_up = qv[i, j, k + 1] + ql[i, j, k + 1] + qi[i, j, k + 1]
                qt = qv[i, j, k] + ql[i, j, k] + qi[i, j, k]
                qs = compute_qvs(T[i, j, k], rho0[k], p0[k])
                bvf[i, j, k] = (
                    parameters.G
                    / (T[i, j, k])
                    * (
                        (T[i, j, k + 1] - T[i, j, k]) / dz
                        + gamma_m(qv[i, j, k], T[i, j, k])
                    )
                    * (1.0 + parameters.LV * qs / (parameters.RD * T[i, j, k]))
                    - parameters.G / (1.0 + qt) * (qt_up - qt) / dz
                )

            # non-boundary
            for k in range(n_halo[2] + 1, shape[2] - n_halo[2]):
                if ql[i, j, k] + qi[i, j, k] < 1e-8:
                    bvf[i, j, k] = (
                        parameters.G
                        / (T[i, j, k])
                        * (
                            0.5 * (T[i, j, k + 1] - T[i, j, k - 1]) / dz
                            + parameters.G / parameters.CPD
                        )
                        - parameters.G
                        / (1.0 + qv[i, j, k])
                        * 0.5
                        * (qv[i, j, k + 1] - qv[i, j, k - 1])
                        / dz
                    )
                else:
                    qt_up = qv[i, j, k + 1] + ql[i, j, k + 1] + qi[i, j, k + 1]
                    qt = qv[i, j, k] + ql[i, j, k] + qi[i, j, k]
                    qt_down = qv[i, j, k - 1] + ql[i, j, k - 1] + qi[i, j, k - 1]
                    qs = compute_qvs(T[i, j, k], rho0[k], p0[k])
                    bvf[i, j, k] = (
                        parameters.G
                        / (T[i, j, k])
                        * (
                            0.5 * (T[i, j, k + 1] - T[i, j, k - 1]) / dz
                            + gamma_m(qv[i, j, k], T[i, j, k])
                        )
                        * (1.0 + parameters.LV * qs / (parameters.RD * T[i, j, k]))
                        - parameters.G / (1.0 + qt) * 0.5 * (qt_up - qt_down) / dz
                    )

            # upper boundary
            k = shape[2] - n_halo[2] - 1
            if ql[i, j, k] + qi[i, j, k] < 1e-8:
                bvf[i, j, k] = (
                    parameters.G
                    / (T[i, j, k])
                    * (
                        (T[i, j, k] - T[i, j, k - 1]) / dz
                        + parameters.G / parameters.CPD
                    )
                    - parameters.G
                    / (1.0 + qv[i, j, k])
                    * (qv[i, j, k] - qv[i, j, k - 1])
                    / dz
                )
            else:
                qt = qv[i, j, k] + ql[i, j, k] + qi[i, j, k]
                qt_down = qv[i, j, k - 1] + ql[i, j, k - 1] + qi[i, j, k - 1]
                qs = compute_qvs(T[i, j, k], rho0[k], p0[k])
                bvf[i, j, k] = (
                    parameters.G
                    / (T[i, j, k])
                    * (
                        (T[i, j, k] - T[i, j, k - 1]) / dz
                        + gamma_m(qv[i, j, k], T[i, j, k])
                    )
                    * (1.0 + parameters.LV * qs / (parameters.RD * T[i, j, k]))
                    - parameters.G / (1.0 + qt) * (qt - qt_down) / dz
                )

    return


@numba.njit(fastmath=True)
def eos(
    z_in, P_in, alpha0, s_in, s_dry_in, qv_in, ql_in, qi_in, T_out, tref, alpha_out, buoyancy_out
):
    shape = s_in.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                T_out[i, j, k] = T(
                    z_in[k], s_in[i, j, k], ql_in[i, j, k], qi_in[i, j, k]
                )
                alpha_out[i, j, k] = alpha(
                    P_in[k],
                    T_out[i, j, k],
                    qv_in[i, j, k],
                    ql_in[i, j, k],
                    qi_in[i, j, k],
                )
                buoyancy_out[i, j, k] = (
                    parameters.G * (alpha_out[i, j, k] - alpha0[k]) / alpha0[k]
                )

                # Compute the dry component of the static energy
                s_dry_in[i,j,k] = s_dry(z_in[k], T_out[i,j,k])

    return


@numba.njit(fastmath=True)
def apply_buoyancy(buoyancy, w_t):
    shape = w_t.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                w_t[i, j, k] += 0.5 * (buoyancy[i, j, k] + buoyancy[i, j, k + 1])
    return

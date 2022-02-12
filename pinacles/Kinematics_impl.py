import numba
import numpy as np


@numba.njit(fastmath=True)
def u_gradients(dxi, u, dudx, dudy, dudz):

    shape = u.shape
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):
                dudx[i, j, k] = (u[i, j, k] - u[i - 1, j, k]) * dxi[0]

                dudy[i, j, k] = (u[i, j + 1, k] - u[i, j, k]) * dxi[1]

                dudz[i, j, k] = (u[i, j, k + 1] - u[i, j, k]) * dxi[2]

    return


@numba.njit(fastmath=True)
def v_gradients(dxi, v, dvdx, dvdy, dvdz):

    shape = v.shape
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                dvdx[i, j, k] = (v[i + 1, j, k] - v[i, j, k]) * dxi[0]

                dvdy[i, j, k] = (v[i, j, k] - v[i, j - 1, k]) * dxi[1]

                dvdz[i, j, k] = (v[i, j, k + 1] - v[i, j, k]) * dxi[2]


@numba.njit(fastmath=True)
def w_gradients(dxi, w, dwdx, dwdy, dwdz):

    shape = w.shape
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                dwdx[i, j, k] = (w[i + 1, j, k] - w[i, j, k]) * dxi[0]

                dwdy[i, j, k] = (w[i, j + 1, k] - w[i, j, k]) * dxi[1]

                dwdz[i, j, k] = (w[i, j, k] - w[i, j, k - 1]) * dxi[2]

    return


@numba.njit(fastmath=True)
def strain_rate(
    dudx,
    dudy,
    dudz,
    dvdx,
    dvdy,
    dvdz,
    dwdx,
    dwdy,
    dwdz,
    s11,
    s22,
    s33,
    s12,
    s13,
    s23,
    strain_rate_mag,
):

    shape = strain_rate_mag.shape
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            for k in range(1, shape[2]):

                s11[i, j, k] = dudx[i, j, k]
                s22[i, j, k] = dvdy[i, j, k]
                s33[i, j, k] = dwdz[i, j, k]

                s12[i, j, k] = 0.5 * ((dvdx[i, j, k] + dudy[i, j, k]))
                s13[i, j, k] = 0.5 * ((dudz[i, j, k] + dwdx[i, j, k]))
                s23[i, j, k] = 0.5 * ((dvdz[i, j, k] + dwdy[i, j, k]))

    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            for k in range(1, shape[2]):

                strain_rate_mag[i, j, k] = np.sqrt(
                    2.0
                    * (
                        s11[i, j, k] ** 2.0
                        + s22[i, j, k] ** 2.0
                        + s33[i, j, k] ** 2.0
                        + +2.0
                        * (
                            0.25
                            * (
                                s12[i, j, k]
                                + s12[i - 1, j, k]
                                + s12[i, j - 1, k]
                                + s12[i - 1, j - 1, k]
                            )
                        )
                        ** 2.0
                        + 2.0
                        * (
                            0.25
                            * (
                                s13[i, j, k]
                                + s13[i - 1, j, k]
                                + s13[i, j, k - 1]
                                + s13[i - 1, j, k - 1]
                            )
                        )
                        ** 2.0
                        + 2.0
                        * (
                            0.25
                            * (
                                s23[i, j, k]
                                + s23[i, j - 1, k]
                                + s23[i, j, k - 1]
                                + s23[i, j - 1, k - 1]
                            )
                        )
                        ** 2.0
                    )
                )
    return


@numba.njit(fastmath=True)
def q_criterion(
    u,
    v,
    w,
    dudx,
    dudy,
    dudz,
    dvdx,
    dvdy,
    dvdz,
    dwdx,
    dwdy,
    dwdz,
    bvf,
    qcrit,
    vertical_vorticity,
    helicity,
    grad_ri,
):
    shape = qcrit.shape
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            for k in range(1, shape[2]):

                dudy_ = 0.25 * (
                    dudy[i, j, k]
                    + dudy[i - 1, j, k]
                    + dudy[i, j - 1, k]
                    + dudy[i - 1, j - 1, k]
                )
                dudz_ = 0.25 * (
                    dudz[i, j, k]
                    + dudz[i - 1, j, k]
                    + dudz[i, j, k - 1]
                    + dudz[i - 1, j, k - 1]
                )

                dvdx_ = 0.25 * (
                    dvdx[i, j, k]
                    + dvdx[i - 1, j, k]
                    + dvdx[i, j - 1, k]
                    + dvdx[i - 1, j - 1, k]
                )
                dvdz_ = 0.25 * (
                    dvdz[i, j, k]
                    + dvdz[i, j - 1, k]
                    + dvdx[i, j, k - 1]
                    + dvdz[i, j - 1, k - 1]
                )

                dwdx_ = 0.25 * (
                    dwdx[i, j, k]
                    + dwdx[i - 1, j, k]
                    + dwdx[i, j, k - 1]
                    + dwdx[i - 1, j, k - 1]
                )
                dwdy_ = 0.25 * (
                    dwdy[i, j, k]
                    + dwdy[i, j - 1, k]
                    + dwdy[i, j, k - 1]
                    + dwdy[i, j - 1, k - 1]
                )

                q12 = (0.5 * (dudy_ - dvdx_)) ** 2.0
                q13 = (0.5 * (dudz_ - dwdx_)) ** 2.0
                q21 = (0.5 * (dvdx_ - dudy_)) ** 2.0
                q23 = (0.5 * (dvdz_ - dwdy_)) ** 2.0
                q31 = (0.5 * (dwdx_ - dudz_)) ** 2.0
                q32 = (0.5 * (dwdy_ - dvdz_)) ** 2.0

                s11 = (dudx[i, j, k]) ** 2.0
                s22 = (dvdy[i, j, k]) ** 2.0
                s33 = (dwdz[i, j, k]) ** 2.0

                s12 = (0.5 * (dvdx_ + dudy_)) ** 2.0
                s21 = s12
                s13 = (0.5 * (dudz_ + dwdx_)) ** 2.0
                s31 = s13
                s23 = (0.5 * (dvdz_ + dwdy_)) ** 2.0
                s32 = s23

                qcrit[i, j, k] = 0.5 * (
                    np.sqrt(q12 + q13 + q21 + q23 + q31 + q32)
                    - np.sqrt(s11 + s22 + s33 + s12 + s21 + s13 + s31 + s23 + s32)
                )

                vertical_vorticity[i, j, k] = dvdx_ - dudy_
                helicity[i, j, k] = 0.5 * (
                    (dwdy_ - dvdz_) * 0.5 * (u[i, j, k] + u[i - 1, j, k])
                    + (dudz_ - dwdx_) * 0.5 * (v[i, j - 1, k] + v[i, j, k])
                    + (dvdx_ - dudy_) * 0.5 * (w[i, j, k] + w[i, j, k - 1])
                )

                grad_ri[i, j, k] = bvf[i, j, k] / (
                    dudz_ * dudz_ + dvdz_ * dvdz_ + 1e-10
                )

    return

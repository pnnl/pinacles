import numba
import numpy as np


@numba.njit(fastmath=True)
def u_gradients(dxi, u, dudx, dudy, dudz):

    shape = u.shape
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):
                dudx[i, j, k] = (u[i, j, k] - u[i - 1, j, k]) * dxi[0]

                # uy_l = 0.25 * (u[i,j,k] + u[i-1,j,k] +
                #                u[i,j-1,k] + u[i-1,j-1,k])

                uy_l = 0.5 * (u[i - 1, j - 1, k] + u[i, j - 1, k])

                # uy_h = 0.25 * (u[i,j+1,k] + u[i-1,j+1,k] +
                #                u[i,j,k] + u[i-1,j,k] )
                uy_h = 0.5 * (u[i - 1, j + 1, k] + u[i, j + 1, k])
                dudy[i, j, k] = 0.5 * (uy_h - uy_l) * (dxi[1])

                # uz_l = 0.25 * (u[i,j,k] + u[i-1,j,k] +
                #               u[i,j,k-1] + u[i-1,j,k-1])

                # uz_h = 0.25 * (u[i,j,k+1] + u[i-1,j,k+1]
                #                + u[i,j,k] + u[i-1,j,k])
                uz_l = 0.5 * (u[i - 1, j, k - 1] + u[i, j, k - 1])

                uz_h = 0.5 * (u[i - 1, j, k + 1] + u[i, j, k + 1])

                dudz[i, j, k] = 0.5 * (uz_h - uz_l) * (dxi[2])

    return


@numba.njit(fastmath=True)
def v_gradients(dxi, v, dvdx, dvdy, dvdz):

    shape = v.shape
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                # vx_l = 0.25 * (v[i,j,k] + v[i,j-1,k] +
                #                v[i-1,j,k] + v[i-1,j-1,k])
                # vx_h = 0.25 * (v[i+1,j-1,k] + v[i+1,j,k] +
                #                v[i,j,k] + v[i,j-1,k] )

                vx_l = 0.5 * (v[i - 1, j - 1, k] + v[i - 1, j, k])
                vx_h = 0.5 * (v[i + 1, j - 1, k] + v[i + 1, j, k])

                dvdx[i, j, k] = 0.5 * (vx_h - vx_l) * (dxi[0])

                dvdy[i, j, k] = (v[i, j, k] - v[i, j - 1, k]) * dxi[1]

                # vz_l = 0.25 * (v[i,j,k] + v[i,j-1,k] +
                #               v[i,j,k-1] + v[i,j-1,k-1])
                #
                # vz_h = 0.25 * (v[i,j,k+1] + v[i-1,j,k+1]
                #                + v[i,j,k] + v[i,j-1,k])
                vz_l = 0.5 * (v[i, j - 1, k - 1] + v[i, j, k - 1])
                vz_h = 0.5 * (v[i, j - 1, k + 1] + v[i, j, k + 1])

                dvdz[i, j, k] = 0.5 * (vz_h - vz_l) * (dxi[2])


@numba.njit(fastmath=True)
def w_gradients(dxi, w, dwdx, dwdy, dwdz):

    shape = w.shape
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):

                # wx_l = 0.25 * (w[i,j,k] + w[i,j,k-1] +
                #                w[i-1,j,k] + w[i-1,j,k-1])
                # wx_h = 0.25 * (w[i+1,j,k-1] + w[i+1,j,k] +
                #                w[i,j,k] + w[i,j,k-1] )

                wx_l = 0.5 * (w[i - 1, j, k - 1] + w[i - 1, j, k])
                wx_h = 0.5 * (w[i + 1, j, k - 1] + w[i + 1, j, k])

                dwdx[i, j, k] = 0.5 * (wx_h - wx_l) * (dxi[0])

                # wy_l = 0.25 * (w[i,j,k] + w[i,j,k-1] +
                #                w[i,j-1,k] + w[i,j-1,k-1])
                # wy_h = 0.25 * (w[i,j+1,k-1] + w[i,j+1,k] +
                #                w[i,j,k] + w[i,j,k-1] )

                wy_l = 0.5 * (w[i, j - 1, k - 1] + w[i, j - 1, k])
                wy_h = 0.5 * (w[i, j + 1, k - 1] + w[i, j + 1, k])

                dwdy[i, j, k] = 0.5 * (wy_h - wy_l) * (dxi[1])
                dwdz[i, j, k] = (w[i, j, k] - w[i, j, k - 1]) * dxi[2]
    return


@numba.njit(fastmath=True)
def strain_rate_max(
    dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, strain_rate_mag
):

    shape = strain_rate_mag.shape
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            for k in range(1, shape[2]):

                s11 = (dudx[i, j, k]) ** 2.0
                s22 = (dvdy[i, j, k]) ** 2.0
                s33 = (dwdz[i, j, k]) ** 2.0

                s12 = (0.5 * (dvdx[i, j, k] + dudy[i, j, k])) ** 2.0
                s21 = s12
                s13 = (0.5 * (dudz[i, j, k] + dwdx[i, j, k])) ** 2.0
                s31 = s13
                s23 = (0.5 * (dvdz[i, j, k] + dwdy[i, j, k])) ** 2.0
                s32 = s23

                strain_rate_mag[i, j, k] = np.sqrt(
                    2.0 * (s11 + s22 + s33 + s12 + s21 + s13 + s31 + s23 + s32)
                )

    return


@numba.njit(fastmath=True)
def q_criterion(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, qcrit):
    shape = qcrit.shape
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            for k in range(1, shape[2]):
                q12 = (0.5 * (dudy[i, j, k] - dvdx[i, j, k])) ** 2.0
                q13 = (0.5 * (dudz[i, j, k] - dwdx[i, j, k])) ** 2.0
                q21 = (0.5 * (dvdx[i, j, k] - dudy[i, j, k])) ** 2.0
                q23 = (0.5 * (dvdz[i, j, k] - dwdy[i, j, k])) ** 2.0
                q31 = (0.5 * (dwdx[i, j, k] - dudz[i, j, k])) ** 2.0
                q32 = (0.5 * (dwdy[i, j, k] - dvdz[i, j, k])) ** 2.0

                s11 = (dudx[i, j, k]) ** 2.0
                s22 = (dvdy[i, j, k]) ** 2.0
                s33 = (dwdz[i, j, k]) ** 2.0

                s12 = (0.5 * (dvdx[i, j, k] + dudy[i, j, k])) ** 2.0
                s21 = s12
                s13 = (0.5 * (dudz[i, j, k] + dwdx[i, j, k])) ** 2.0
                s31 = s13
                s23 = (0.5 * (dvdz[i, j, k] + dwdy[i, j, k])) ** 2.0
                s32 = s23

                qcrit[i, j, k] = 0.5 * (
                    np.sqrt(q12 + q13 + q21 + q23 + q31 + q32)
                    - np.sqrt(s11 + s22 + s33 + s12 + s21 + s13 + s31 + s23 + s32)
                )

    return

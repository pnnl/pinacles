import numba
import numpy as np
@numba.njit()
def u_gradients(dxi, u, dudx, dudy, dudz):

    shape = u.shape
    for i in range(1,shape[0]):
        for j in range(1,shape[1]):
            for k in range(1, shape[2]):
                dudx[i,j,k] = (u[i,j,k] - u[i-1,j,k])*dxi[0]

                uy_l = 0.25 * (u[i,j,k] + u[i-1,j,k] +
                                u[i,j-1,k] + u[i-1,j-1,k])
                uy_h = 0.25 * (u[i,j+1,k] + u[i-1,j+1,k] +
                                u[i,j,k] + u[i-1,j,k] )
                dudy[i,j,k] = (uy_h - uy_l)*dxi[1]



                uz_l = 0.25 * (u[i,j,k] + u[i-1,j,k] +
                               u[i,j,k-1] + u[i-1,j,k-1])

                uz_h = 0.25 * (u[i,j,k+1] + u[i-1,j,k+1]
                                + u[i,j,k] + u[i-1,j,k])
                dudz[i,j,k] = (uz_h - uz_l)*dxi[2]


    return

@numba.njit()
def v_gradients(dxi, v, dvdx, dvdy, dvdz):

    shape = v.shape
    for i in range(1,shape[0]):
        for j in range(1,shape[1]):
            for k in range(1, shape[2]):

                vx_l = 0.25 * (v[i,j,k] + v[i,j-1,k] +
                                v[i-1,j,k] + v[i-1,j-1,k])
                vx_h = 0.25 * (v[i+1,j-1,k] + v[i+1,j,k] +
                                v[i,j,k] + v[i,j-1,k] )

                dvdx[i,j,k] = (vx_h - vx_l)*dxi[0]

                dvdy[i,j,k] = (v[i,j-1,k] - v[i,j,k])*dxi[1]
                
                vz_l = 0.25 * (v[i,j,k] + v[i,j-1,k] +
                               v[i,j,k-1] + v[i,j-1,k-1])

                vz_h = 0.25 * (v[i,j,k+1] + v[i-1,j,k+1]
                                + v[i,j,k] + v[i,j-1,k])
                dvdz[i,j,k] = (vz_h - vz_l)*dxi[2]


@numba.njit()
def w_gradients(dxi, w, dwdx, dwdy, dwdz):

    shape = w.shape
    for i in range(1,shape[0]):
        for j in range(1,shape[1]):
            for k in range(1, shape[2]):

                wx_l = 0.25 * (w[i,j,k] + w[i,j,k-1] +
                                w[i-1,j,k] + w[i-1,j,k-1])
                wx_h = 0.25 * (w[i+1,j,k-1] + w[i+1,j,k] +
                                w[i,j,k] + w[i,j,k-1] )


                dwdx[i,j,k] = (wx_h - wx_l)*dxi[0]

                wy_l = 0.25 * (w[i,j,k] + w[i,j,k-1] +
                                w[i,j-1,k] + w[i,j-1,k-1])
                wy_h = 0.25 * (w[i,j+1,k-1] + w[i,j+1,k] +
                                w[i,j,k] + w[i,j,k-1] )

                dwdy[i,j,k] = (wy_h - wy_l)*dxi[1]

                dwdz[i,j,k] = (w[i,j,k-1] - w[i,j,k])*dxi[2]

    return


@numba.njit()
def strain_rate_max(dudx, dudy, dudz,
                    dvdx, dvdy, dvdz,
                    dwdx, dwdy, dwdz, strain_rate_mag):

    shape = strain_rate_mag.shape
    for i in range(1,shape[0]):
        for j in range(1,shape[1]):
            for k in range(1, shape[2]):

                s11 = dudx[i,j,k]**2.0
                s22 = dvdy[i,j,k]**2.0
                s33 = dwdz[i,j,k]**2.0

                s12 = (0.5*(dvdx[i,j,k] + dudy[i,j,k]))**2.0
                s21 = s12
                s13 = (0.5*(dudz[i,j,k] + dwdx[i,j,k]))**2.0
                s31 = s13
                s23 = (0.5 * (dvdz[i,j,k] + dwdy[i,j,k]))**2.0
                s32 = s23

                strain_rate_mag[i,j,k] =  np.sqrt(4.0 * (s11 + s22 + s33 + s12 + s21 + s13 + s31 + s23 + s32))

    return
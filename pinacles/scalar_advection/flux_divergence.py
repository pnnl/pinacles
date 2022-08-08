import numpy as np
import numba


@numba.njit(fastmath=True)
def flux_divergence(
    nhalo, idx, idy, idzi, alpha0, fluxx, fluxy, fluxz, io_flux, phi_range, phi_t
):
    phi_shape = phi_t.shape
    # TODO Tighten range of loops
    for i in range(nhalo[0], phi_shape[0] - nhalo[0]):
        for j in range(nhalo[1], phi_shape[1] - nhalo[1]):
            for k in range(nhalo[2], phi_shape[2] - nhalo[2]):
                phi_t[i, j, k] -= (
                    alpha0[k]
                    * (
                        (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                        + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                        + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                    )
                    * phi_range
                )
                io_flux[k] += (
                    (fluxz[i, j, k] + fluxz[i, j, k - 1]) * 0.5 * alpha0[k] * phi_range
                )


@numba.njit(fastmath=True)
def flux_divergence_bounded(
    nhalo,
    idx,
    idy,
    idzi,
    alpha0,
    fluxx,
    fluxy,
    fluxz,
    fluxx_low,
    fluxy_low,
    fluxz_low,
    vmin,
    vmax,
    dt,
    phi,
    io_flux,
    phi_range,
    phi_t,
):
    phi_shape = phi_t.shape
    # TODO Tighten range of loops
    for i in range(1, phi_shape[0] - 1):
        for j in range(1, phi_shape[1] - 1):
            for k in range(1, phi_shape[2] - 1):

                tend_tmp = -alpha0[k] * (
                    (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                    + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                    + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                )

                ho_val = phi[i, j, k] + (tend_tmp * dt) * phi_range

                if ho_val < vmin or ho_val > vmax:

                    fluxx[i - 1, j, k] = fluxx_low[i - 1, j, k]
                    fluxx[i, j, k] = fluxx_low[i, j, k]

                    fluxy[i, j - 1, k] = fluxy_low[i, j - 1, k]
                    fluxy[i, j, k] = fluxy_low[i, j, k]

                    fluxz[i, j, k - 1] = fluxz_low[i, j, k - 1]
                    fluxz[i, j, k] = fluxz_low[i, j, k]

    for i in range(nhalo[0], phi_shape[0] - nhalo[0]):
        for j in range(nhalo[1], phi_shape[1] - nhalo[1]):
            for k in range(nhalo[2], phi_shape[2] - nhalo[2]):
                io_flux[k] += (fluxz[i, j, k] + fluxz[i, j, k - 1]) * 0.5 * phi_range
                phi_t[i, j, k] -= (
                    alpha0[k]
                    * (
                        (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                        + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                        + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                    )
                    * phi_range
                )


@numba.njit(fastmath=True)
def flux_divergence_monotone(
    nhalo,
    idx,
    idy,
    idzi,
    alpha0,
    fluxx,
    fluxy,
    fluxz,
    fluxx_low,
    fluxy_low,
    fluxz_low,
    dt,
    phi,
    io_flux,
    phi_range,
    phi_t,
):
    phi_shape = phi_t.shape

    for i in range(1, phi_shape[0] - 1):
        for j in range(1, phi_shape[1] - 1):
            for k in range(1, phi_shape[2] - 1):

                smin = min(1e20, phi[i, j, k])
                smax = max(-1e20, phi[i, j, k])
                for ii in range(-1, 2, 2):

                    smin = min(smin, phi[ii + i, j, k])
                    smax = max(smax, phi[ii + i, j, k])

                    smin = min(smin, phi[i, j + ii, k])
                    smax = max(smax, phi[i, j + ii, k])

                    smin = min(smin, phi[i, j, k + ii])
                    smax = max(smax, phi[i, j, k + ii])

                tend_tmp = -alpha0[k] * (
                    (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                    + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                    + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                )
                tmp = phi[i, j, k] + (tend_tmp * dt)


                if tmp <= smin or tmp >= smax:
                    fluxx[i - 1, j, k] = fluxx_low[i - 1, j, k]
                    fluxx[i, j, k] = fluxx_low[i, j, k]

                    fluxy[i, j - 1, k] = fluxy_low[i, j - 1, k]
                    fluxy[i, j, k] = fluxy_low[i, j, k]

                    fluxz[i, j, k - 1] = fluxz_low[i, j, k - 1]
                    fluxz[i, j, k] = fluxz_low[i, j, k]

                if k == nhalo[2]:
                    fluxz[i, j, k - 1] = fluxz_low[i, j, k - 1]
                    fluxz[i, j, k] = fluxz_low[i, j, k]

    for i in range(nhalo[0], phi_shape[0] - nhalo[0]):
        for j in range(nhalo[1], phi_shape[1] - nhalo[1]):
            for k in range(nhalo[2], phi_shape[2] - nhalo[2]):
                io_flux[k] += (fluxz[i, j, k] + fluxz[i, j, k - 1]) * 0.5 * phi_range
                phi_t[i, j, k] -= (
                    alpha0[k]
                    * (
                        (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                        + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                        + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                    )
                    * phi_range
                )


@numba.njit(fastmath=True)
def flux_divergence_split_monotone(
    nhalo,
    idx,
    idy,
    idzi,
    alpha0,
    fluxx,
    fluxy,
    fluxz,
    fluxx_low,
    fluxy_low,
    fluxz_low,
    dt,
    phi,
    io_flux,
    phi_range,
    phi_t,
):
    phi_shape = phi_t.shape

    for i in range(1, phi_shape[0] - 1):
        for j in range(1, phi_shape[1] - 1):
            for k in range(1, phi_shape[2] - 1):

                smin_x = min(1e20, phi[i, j, k])
                smax_x = max(-1e20, phi[i, j, k])

                smin_z = min(1e20, phi[i, j, k])
                smax_z = max(-1e20, phi[i, j, k])

                smin_y = min(1e20, phi[i, j, k])
                smax_y = max(-1e20, phi[i, j, k])

                for ii in range(-1, 2, 2):

                    smin_x = min(smin_x, phi[ii + i, j, k])
                    smax_x = max(smax_x, phi[ii + i, j, k])

                    smin_y = min(smin_y, phi[i, j + ii, k])
                    smax_y = max(smax_y, phi[i, j + ii, k])

                    smin_z = min(smin_z, phi[i, j, k + ii])
                    smax_z = max(smax_z, phi[i, j, k + ii])

                tmp_x = (
                    phi[i, j, k]
                    - alpha0[k] * ((fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx) * dt
                )

                tmp_y = (
                    phi[i, j, k]
                    - alpha0[k] * ((fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy) * dt
                )

                tmp_z = (
                    phi[i, j, k]
                    - alpha0[k] * ((fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi) * dt
                )

                if tmp_x < smin_x or tmp_x > smax_x:
                    fluxx[i - 1, j, k] = fluxx_low[i - 1, j, k]
                    fluxx[i, j, k] = fluxx_low[i, j, k]

                if tmp_y < smin_y or tmp_y > smax_y:
                    fluxy[i, j - 1, k] = fluxy_low[i, j - 1, k]
                    fluxy[i, j, k] = fluxy_low[i, j, k]

                if tmp_z < smin_z or tmp_z > smax_z:
                    fluxz[i, j, k - 1] = fluxz_low[i, j, k - 1]
                    fluxz[i, j, k] = fluxz_low[i, j, k]

                if k == nhalo[2]:
                    fluxz[i, j, k - 1] = fluxz_low[i, j, k - 1]
                    fluxz[i, j, k] = fluxz_low[i, j, k]

    for i in range(nhalo[0], phi_shape[0] - nhalo[0]):
        for j in range(nhalo[1], phi_shape[1] - nhalo[1]):
            for k in range(nhalo[2], phi_shape[2] - nhalo[2]):
                io_flux[k] += (fluxz[i, j, k] + fluxz[i, j, k - 1]) * 0.5 * phi_range
                phi_t[i, j, k] -= (
                    alpha0[k]
                    * (
                        (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                        + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                        + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                    )
                    * phi_range
                )

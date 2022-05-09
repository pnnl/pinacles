import numba
import numpy as np


@numba.njit
def to_wrf_order(nhalo, our_array, wrf_array):
    shape = our_array.shape
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                i_wrf = i - nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2]  # shape[2] - 1 - k
                wrf_array[i_wrf, k_wrf, j_wrf] = our_array[i, j, k]
    return


@numba.njit
def to_our_order(nhalo, wrf_array, our_array):
    shape = our_array.shape
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                i_wrf = i - nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2]
                our_array[i, j, k] = wrf_array[i_wrf, k_wrf, j_wrf]


@numba.njit
def to_wrf_order_halo(nhalo, our_array, wrf_array):
    shape = our_array.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                wrf_array[i, k, j] = our_array[i, j, k]
    return


@numba.njit
def to_our_order_halo(nhalo, wrf_array, our_array):
    shape = our_array.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                our_array[i, j, k] = wrf_array[i, k, j]


@numba.njit
def to_wrf_order_4d(nhalo, our_array, wrf_array):
    shape = our_array.shape

    for n in range(shape[0]):
        for i in range(nhalo[0], shape[1] - nhalo[0]):
            for j in range(nhalo[1], shape[2] - nhalo[1]):
                for k in range(nhalo[2], shape[3] - nhalo[2]):
                    i_wrf = i - nhalo[0]
                    j_wrf = j - nhalo[1]
                    k_wrf = k - nhalo[2]
                    wrf_array[i_wrf, k_wrf, j_wrf, n] = our_array[n, i, j, k]
    return


@numba.njit
def to_our_order_4d(nhalo, wrf_array, our_array):
    shape = our_array.shape
    for n in range(shape[0]):
        for i in range(nhalo[0], shape[1] - nhalo[0]):
            for j in range(nhalo[1], shape[2] - nhalo[1]):
                for k in range(nhalo[2], shape[3] - nhalo[2]):
                    i_wrf = i - nhalo[0]
                    j_wrf = j - nhalo[1]
                    k_wrf = k - nhalo[2]
                    our_array[n, i, j, k] = wrf_array[i_wrf, k_wrf, j_wrf, n]


@numba.njit
def to_wrf_order_4d_halo(nhalo, our_array, wrf_array):
    shape = our_array.shape
    for n in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    wrf_array[i, k, j, n] = our_array[n, i, j, k]
    return


@numba.njit
def wrf_theta_tend_to_our_tend(nhalo, dt, exner, wrf_out, our_in, stend):
    shape = stend.shape
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                i_wrf = i - nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2]  # shape[2] - 1 - k
                stend[i, j, k] += (
                    wrf_out[i_wrf, k_wrf, j_wrf] * exner[k] - our_in[i, j, k]
                ) / dt

    return


@numba.njit
def wrf_tend_to_our_tend(nhalo, dt, wrf_out, our_in, tend):
    shape = tend.shape
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                i_wrf = i - nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2]  # shape[2] - 1 - k
                tend[i, j, k] += (wrf_out[i_wrf, k_wrf, j_wrf] - our_in[i, j, k]) / dt
    return


@numba.njit
def to_our_order_4d_halo(nhalo, wrf_array, our_array):
    shape = our_array.shape
    for n in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    our_array[n, i, j, k] = wrf_array[i, k, j, n]

@numba.njit
def to_sam_order(nhalo, our_array, sam_array):
    shape = our_array.shape
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                i_sam = i - nhalo[0]
                j_sam = j - nhalo[1]
                k_sam = k - nhalo[2]  # shape[2] - 1 - k
                sam_array[i_sam, j_sam, k_sam] = our_array[i, j, k]
    return


@numba.njit
def sam_to_our_order(nhalo, sam_array, our_array):
    shape = our_array.shape
    for i in range(nhalo[0], shape[0] - nhalo[0]):
        for j in range(nhalo[1], shape[1] - nhalo[1]):
            for k in range(nhalo[2], shape[2] - nhalo[2]):
                i_sam = i - nhalo[0]
                j_sam = j - nhalo[1]
                k_sam = k - nhalo[2]
                our_array[i, j, k] = sam_array[i_sam, j_sam, k_sam]
         
        
@numba.njit
def to_sam_order_4d(nhalo, our_array, sam_array):
    shape = our_array.shape

    for n in range(shape[0]):
        for i in range(nhalo[0], shape[1] - nhalo[0]):
            for j in range(nhalo[1], shape[2] - nhalo[1]):
                for k in range(nhalo[2], shape[3] - nhalo[2]):
                    i_sam = i - nhalo[0]
                    j_sam = j - nhalo[1]
                    k_sam = k - nhalo[2]
                    sam_array[i_sam, j_sam, k_sam, n] = our_array[n, i, j, k]
    return


@numba.njit
def sam_to_our_order_4d(nhalo, sam_array, our_array):
    shape = our_array.shape
    for n in range(shape[0]):
        for i in range(nhalo[0], shape[1] - nhalo[0]):
            for j in range(nhalo[1], shape[2] - nhalo[1]):
                for k in range(nhalo[2], shape[3] - nhalo[2]):
                    i_sam = i - nhalo[0]
                    j_sam = j - nhalo[1]
                    k_sam = k - nhalo[2]
                    our_array[n, i, j, k] = sam_array[i_sam, j_sam, k_sam, n]

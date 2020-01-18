import numba
@numba.njit
def to_wrf_order(nhalo, our_array, wrf_array):
    shape = our_array.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i  -nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2] #shape[2] - 1 - k
                wrf_array[i_wrf,k_wrf,j_wrf] = our_array[i,j,k]
    return

@numba.njit
def wrf_theta_tend_to_our_tend(nhalo, dt, exner, wrf_out, our_in, stend):
    shape = stend.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i  -nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2] #shape[2] - 1 - k
                stend[i,j,k] += (wrf_out[i_wrf,k_wrf,j_wrf]*exner[k]- our_in[i,j,k])/dt

    return

@numba.njit
def wrf_tend_to_our_tend(nhalo, dt, wrf_out, our_in, tend):
    shape = tend.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i  -nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2]#shape[2] - 1 - k
                tend[i,j,k] += (wrf_out[i_wrf,k_wrf,j_wrf] - our_in[i,j,k])/dt
    return

@numba.njit
def to_our_order(nhalo, wrf_array, our_array):
    shape = our_array.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i -nhalo[0]
                j_wrf = j - nhalo[1]
                k_wrf = k - nhalo[2]
                our_array[i,j,k] = wrf_array[i_wrf, k_wrf, j_wrf]
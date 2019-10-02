import numba

@numba.njit
def remove_wmean(w_mean, w):
    shape = w.shape
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                w[i,j,k] = w[i,j,k] - w_mean[k]

    return

@numba.njit
def divergence(dxs, u, v, w, div):

    shape = u.shape
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                div[i,j,k] = 0.0


    return
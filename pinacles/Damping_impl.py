import numba


@numba.njit
def rayleigh(timescale_profile, mean, field, tend):

    shape = field.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                tend[i, j, k] += timescale_profile[k] * (mean[k] - field[i, j, k])

    return

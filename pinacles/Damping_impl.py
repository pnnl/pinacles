import numba


@numba.njit
def rayleigh(timescale_profile, mean, field, tend):

    shape = field.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                tend[i, j, k] += timescale_profile[k] * (mean[k] - field[i, j, k])

    return


@numba.njit
def rayleigh_N2(timescale, N2, field, tend):

    shape = field.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if (N2[i,j,k] + N2[i,j,k+1]) * 0.5 > 0.0:
                    tend[i, j, k] += timescale * (0.0 - field[i, j, k])

    return

   
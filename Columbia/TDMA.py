import numpy as np
import numba

@numba.njit()
def Thomas(x, a, b, c):
    shape = x.shape
    scratch = np.empty(shape[2], dtype=np.double)
    scratch[0] = c[0]/b[0]
    x[0] = x[0]/b[0]

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(1,shape[2]):
                m = 1.0/(b[i] - a[i] * scratch[i-1])
                scratch[i] = c[i] * m
                x[i] = (x[i] - a[i] * x[i-1])*m

            for k in range(shape[2]-2,-1,-1):
                x[i] = x[i] - scratch[i] * x[i+1]
    return
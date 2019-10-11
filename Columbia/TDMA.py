import numpy as np
import numba

@numba.njit()
def Thomas(x, a, b, c):
    shape = x.shape
    scratch = np.empty(shape[2], dtype=np.double)
    for i in range(shape[0]):
        for j in range(shape[1]):
            #Upward sweep
            scratch[0] = c[0]/b[0]
            x[i,j,0] = x[i,j,0]/b[0]
            for k in range(1,shape[2]):
                m = 1.0/(b[k] - a[k] * scratch[k-1])
                scratch[k] = c[k] * m
                x[i,j,k] = (x[i,j,k] - a[k] * x[i,j,k-1])*m
            #Downward sweep
            for k in range(shape[2]-2,-1,-1):
                x[i,j,k] = x[i,j,k] - scratch[k] * x[i,j,k+1]
    return
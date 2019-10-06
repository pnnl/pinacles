import numba

@numba.njit
def divergence(dxs, rho0, rho0_edge, u, v, w, div):

    shape = u.shape
    for k in range(1,shape[0]):
        for j in range(1,shape[1]):
            for i in range(1,shape[2]):
                div[i,j,k] = ((u[i,j,k]-u[i-1,j,k])/dxs[0]/rho0[k]
                    + (v[i,j,k] - v[i,j,k-1])/dxs[1]/rho0[k]
                    + (w[i,j,k]/rho0_edge[k] - w[i,j,k-1]/rho0_edge[k-1])/dxs[2])


    return
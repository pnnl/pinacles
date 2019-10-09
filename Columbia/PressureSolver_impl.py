import numba

@numba.njit
def divergence(n_halo, dxs, rho0, rho0_edge, u, v, w, div):

    shape = u.shape
    for i in range(n_halo[0],n_halo[0]):
        for j in range(n_halo[1],n_halo[1]):
            for k in range(n_halo[2],n_halo[2]):
                div[i,j,k] = ((u[i,j,k]-u[i-1,j,k])/dxs[0]/rho0[k]
                    + (v[i,j,k] - v[i,j,k-1])/dxs[1]/rho0[k]
                    + (w[i,j,k]/rho0_edge[k] - w[i,j,k-1]/rho0_edge[k-1])/dxs[2])

    return
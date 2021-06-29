import numba
from pinacles import parameters




@numba.njit
def buoyancy(Theta_ref, Thetav):
    return parameters.G * (Thetav - Theta_ref) / Theta_ref




@numba.njit()
def compute_bvf(n_halo, theta_ref,  dz, thetav, bvf):

    shape = bvf.shape
   
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            k = n_halo[2]
            bvf[i, j, k] = (
                parameters.G
                / theta_ref[k]
                * (thetav[i, j, k + 1] - thetav[i, j, k])
                / (dz)
            )
            for k in range(n_halo[2] + 2, shape[2] - n_halo[2]):
                bvf[i, j, k] = (
                    parameters.G
                    / theta_ref[k]
                    * (thetav[i, j, k + 1] - thetav[i, j, k - 1])
                    / (2.0 * dz)
                )
            k = shape[2] - n_halo[2] - 1
            bvf[i, j, k] = (
                parameters.G
                / theta_ref[k]
                * (thetav[i, j, k] - thetav[i, j, k - 1])
                / (dz)
            )

    return


@numba.njit
def apply_buoyancy(buoyancy, thetav, theta_ref, w_t):
    shape = w_t.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                buoyancy[i,j,k] =  (
                    parameters.G * (thetav[i,j,k] - theta_ref[k]) / theta_ref[k]
                    )
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                w_t[i, j, k] += 0.5 * (buoyancy[i, j, k] + buoyancy[i, j, k + 1])
    return

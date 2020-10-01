import numpy as numpy
import numba

@numba.njit
def large_scale_pgf(ug, vg, f, u, v, u0, v0, ut, vt):
    shape = ut.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            for k in range(1,shape[2]-1):
                u_at_v = 0.25 * (u[i,j,k] + u[i-1,j,k] + u[i-1,j+1,k] + u[i,j+1,k]) + u0
                v_at_u = 0.25 * (v[i,j,k] + v[i+1,j,k] + v[i+1,j-1,k] + v[i,j-1,k]) + v0
                ut[i,j,k] -= f * (vg[k] - v_at_u)
                vt[i,j,k] += f * (ug[k] - u_at_v)
    return

@numba.njit
def apply_subsidence(subsidence, idz, phi, phi_t):
    phi_shape = phi.shape
    for i in range(1,phi_shape[0] -1):
        for j in range(1,phi_shape[1] -1):
            for k in range(1,phi_shape[2] - 1):
                phi_t[i,j,k] -= subsidence[k] *  (phi[i,j,k+1] - phi[i,j,k])*idz

    return
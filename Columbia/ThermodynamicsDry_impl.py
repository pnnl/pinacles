import numba 
from Columbia import parameters

@numba.njit()
def s(z, T): 
    return parameters.CPD * T + parameters.G*z

@numba.njit()
def T(z, s): 
    return (s - parameters.G * z)*parameters.ICPD

@numba.njit()
def rho(P, T): 
    return P/(parameters.RD*T)

@numba.njit
def alpha(P,T): 
    return 1.0/rho(P,T)

@numba.njit
def buoyancy(alpha0,alpha): 
    return parameters.G * (alpha - alpha0)/alpha0

@numba.njit 
def eos(z_in, P_in, alpha0, s_in, T_out, alpha_out, buoyancy_out):
    shape = s_in.shape
    for k in range(shape[2]): 
        for j in range(shape[1]):
            for i in range(shape[0]):
                 T_out[i,j,k] = T(z_in[k], s_in[i,j,k])
                 alpha_out[i,j,k] = alpha(P_in[k], T_out[i,j,k])
                 buoyancy_out[i,j,k] = buoyancy(alpha0[k], alpha_out[i,j,k])
    return 

@numba.njit
def apply_buoyancy(buoyancy, w_t): 
    shape = w_t.shape
    for k in range(shape[2]): 
        for j in range(shape[1]):
            for i in range(shape[0]):
                w_t[i,j,k] += w_t[i,j,k]
    return 

import numba
from Columbia import parameters

@numba.njit()
def s(z, T, ql, qi):
    return T + (parameters.G*z - parameters.LV*ql - parameters.LS*qi)*parameters.ICPD

@numba.njit()
def T(z, s, ql, qi):
    return s + (parameters.LV*ql + parameters.LS*qi - parameters.G*z)*parameters.ICPD

@numba.njit()
def rho(P, T): 
    return P/(parameters.RD*T)

@numba.njit()
def alpha(P,T): 
    return 1.0/rho(P,T)

@numba.njit()
def buoyancy(alpha0,alpha): 
    return parameters.G * (alpha - alpha0)/alpha0

@numba.njit()
def buoyancy_sam(): 
    return 

@numba.njit()
def eos(z_in, P_in, alpha0, s_in, ql_in, qi_in, T_out, alpha_out, buoyancy_out):
    shape = s_in.shape
    for i in range(shape[0]): 
        for j in range(shape[1]):
            for k in range(shape[2]):
                 T_out[i,j,k] = T(z_in[k], s_in[i,j,k], ql_in[i,j,k], qi_in[i,j,k])
                 alpha_out[i,j,k] = alpha(P_in[k], T_out[i,j,k])
                 buoyancy_out[i,j,k] = buoyancy(alpha0[k], alpha_out[i,j,k])
    return 

@numba.njit()
def eos_sam(z_in, P_in, alpha0, s_in, qv_in, ql_in, qi_in, T_out, tref, alpha_out, buoyancy_out):
    shape = s_in.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                T_out[i,j,k] = T(z_in[k], s_in[i,j,k], ql_in[i,j,k], qi_in[i,j,k])
                alpha_out[i,j,k] = alpha(P_in[k], T_out[i,j,k])
                p_prime = parameters.RD/alpha0[k] * tref[k]
                buoyancy_out[i,j,k] = parameters.G * (T_out[i,j,k]/tref[k] + 0.608 * qv_in[i,j,k] - ql_in[i,j,k] - p_prime/P_in[k])

    return

@numba.njit()
def apply_buoyancy(buoyancy, w_t):
    shape = w_t.shape
    for i in range(shape[0]): 
        for j in range(shape[1]):
            for k in range(shape[2]):
                w_t[i,j,k] += 0.5 * (buoyancy[i,j,k] + buoyancy[i,j,k+1])
    return
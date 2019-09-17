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
def eos(p0, s):


    return 
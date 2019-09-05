import numba 
from Columbia import Thermodynamics
print(dir(Thermodynamics))
from Columbia import parameters 

@numba.njit()
def h(T, z): 
    return parameters.CPD * T + parameters.G*z

@numba.njit()
def T(h,z): 
    return (h - parameters.G * z)*parameters.ICPD

@numba.njit()
def rho(P0, T): 
    return P0/(parameters.RD*T)


class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase): 
    def __init__(self, Grid, PrognosticState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, PrognosticState, DiagnosticState)
        return  

    def update(self): 

        return 


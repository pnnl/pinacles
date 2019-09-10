import numba 
from Columbia import Thermodynamics
print(dir(Thermodynamics))
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

class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase): 
    def __init__(self, Grid, PrognosticState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, PrognosticState, DiagnosticState)
        return  

    def update(self): 

        return 


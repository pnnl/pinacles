import numba 
from Columbia import Thermodynamics, ThermodynamicsDry_impl
print(dir(Thermodynamics))
from Columbia import parameters 

class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase): 
    def __init__(self, Grid, ScalarState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, ScalarState, DiagnosticState)
        return  

    def update(self): 

        return 


import numba 
from Columbia import Thermodynamics, ThermodynamicsDry_impl
from Columbia import parameters 

class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase): 
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState)
        
        return  

    def update(self): 

        z = self._Grid.z_global
        p0 = self._Ref.p0 
        alpha0 = self._Ref.alpha0

        s = self._ScalarState.get_field('h')
        T = self._DiagnosticState.get_field('T')
        alpha = self._DiagnosticState.get_field('alpha')
        buoyancy = self._DiagnosticState.get_field('buoyancy')
        w_t = self._VelocityState.get_tend('w')

        ThermodynamicsDry_impl.eos(z, p0, alpha0, s, T, alpha, buoyancy)
        ThermodynamicsDry_impl.apply_buoyancy(buoyancy, w_t)

        return 

